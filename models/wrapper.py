import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math
import yaml
import copy
import clip
import platform
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from .model_timm import CLIP
# from .model import CLIP
#from .model_old import CLIP


class CLIPWrapper(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 config: dict,
                 minibatch_size: int
                 ):
        """A lightning wrapper for a CLIP model as specified in the paper.

        Args:
            model_name (str): A case sensitive visual model name.
            config (dict): A dictionary containing the CLIP instantiation parameters.
        """
        super().__init__()

        self.model_name = model_name
        self.model = CLIP(**config)
        self.minibatch_size = minibatch_size
        self.isViT = 'ViT' in self.model_name

        self.automatic_optimization = False
    
    # Sourced from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = len(dataset)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    # Training loss: https://github.com/openai/CLIP/issues/83
    # Mini-batching thanks to https://github.com/crowsonkb / https://twitter.com/RiversHaveWings
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image, text = train_batch
        n = math.ceil(len(image) // self.minibatch_size)
        image_mbs = torch.chunk(image, n)
        text_mbs = torch.chunk(text, n)

        # calculate original statistics
        with torch.no_grad():
            ims = [F.normalize(self.model.encode_image(im), dim=1) for im in image_mbs]
            txt = [F.normalize(self.model.encode_text(t), dim=1) for t in text_mbs]
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            txt = self.all_gather(torch.cat(txt))

            if len(ims.shape) == 3:
                ims = list(ims)
                txt = list(txt)
            else:
                ims = [ims]
                txt = [txt]

            image_logits = torch.cat(ims) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            # 将交叉熵损失替换为KL散度 想法：就简单换个torch.nn.function函数:cross_entropy->KLDivLoss 但是KL度量的是两个分布之间的不相似性
            #loss = (F.kl_div(image_logits, ground_truth,reduction='batchmean') + F.kl_div(image_logits.t(), ground_truth,reduction='batchmean')).div(2)
            # loss = (F.kl_div(torch.cat(ims),torch.cat(txt),reduction='batchmean')+F.kl_div(torch.cat(txt),torch.cat(ims),reduction='batchmean')).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            self.log_dict({'loss': loss / len(ims), 'acc': (acc_i + acc_t) / 2 / len(image) / len(ims)}, prog_bar=True)

        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        # 原来在这里进行梯度清零，挪到了计算loss的下面，看看会不会收敛，不收敛应该改回来
        # optimizer.zero_grad()

        # image loss
        for j, mb in enumerate(image_mbs):
            images_tmp = copy.deepcopy(ims)
            images_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.encode_image(mb), dim=1)
            image_logits = torch.cat(images_tmp) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            # loss = (F.kl_div(torch.cat(txt), torch.cat(ims)) + F.kl_div(torch.cat(ims), torch.cat(txt))) / 2
            self.manual_backward(loss)

        # text loss
        for j, mb in enumerate(text_mbs):
            text_tmp = copy.deepcopy(txt)
            text_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.encode_text(mb), dim=1)
            image_logits = torch.cat(ims) @ torch.cat(text_tmp).t() * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            # loss = (F.kl_div(torch.cat(txt), torch.cat(ims)) + F.kl_div(torch.cat(ims), torch.cat(txt))) / 2
            self.manual_backward(loss)

        accumulate = True
        if not accumulate:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()
            self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))

        # grad_accumulation
        else:
            N = 2
            if(idx+1) % N ==0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler =self.lr_schedulers()
                lr_scheduler.step()
                self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))

    def validation_step(self, val_batch, idx):
        image, text = val_batch
        image_logits, text_logits = self.forward(image, text)
        ground_truth = torch.arange(len(image_logits))
        loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(text_logits, ground_truth)).div(2)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        lr = {
            "RN50": 5e-4,
            "RN101": 5e-4,
            "RN50x4": 5e-4,
            "RN50x16": 4e-4,
            "RN50x64": 3.6e-4,
            "ViT-B/32": 5e-4,
            "ViT-B/16": 5e-4,
            "ViT-L/14": 4e-4,
            "ViT-L/14-336px": 2e-5
        }[self.model_name]

        model = self.model
        Rmax = 10
        if self.model_name == "ViT-L/14":
            Rmax = 23

        # no_smaller = [
        #     # 'model.visual.prompt_embeddings',
        #     # 'model.visual.transformer.prompt_embeddings',
        #     'model.visual.class_embedding']+[f"model.visual.transformer.resblocks.{i}.prompt_embeddings" for i in range(0, Rmax)]

        no_smaller = [
            'class_embedding','prompt_embedding'
        ]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_smaller)],
                #"lr": 0.00004,
                "requires_grad": False
                # "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_smaller)],
                # "lr": 0.0001 * 1,
                "lr": {
                    "RN50": 5e-4,
                    "RN101": 5e-4,
                    "RN50x4": 5e-4,
                    "RN50x16": 4e-4,
                    "RN50x64": 3.6e-4,
                    "ViT-B/32": 5e-4,
                    "ViT-B/16": 5e-4,
                    "ViT-L/14": 4e-4,
                    "ViT-L/14-336px": 2e-5
                }[self.model_name]
            }
        ]
        grad_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_smaller)],
                # "lr": 0.0001 * 1,
                "lr": {
                    "RN50": 5e-4,
                    "RN101": 5e-4,
                    "RN50x4": 5e-4,
                    "RN50x16": 4e-4,
                    "RN50x64": 3.6e-4,
                    "ViT-B/32": 5e-4,
                    "ViT-B/16": 5e-4,
                    "ViT-L/14": 4e-4,
                    "ViT-L/14-336px": 2e-5
                }[self.model_name]
            }
        ]

        optimizer = torch.optim.AdamW(
            # 筛选requires_grad ==True
            # filter(lambda p: p.requires_grad, self.model.parameters()),
            # self.model.parameters(),

            optimizer_grouped_parameters,
            # grad_parameters,
            # lr=lr,
            betas=(
                0.9,
                0.98 if self.isViT else 0.999
            ),
            eps=1e-6 if self.isViT else 1e-8,
            weight_decay=0.2
        )

        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            # first_cycle_steps=self.num_training_steps,
            first_cycle_steps=8000,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=0,
            warmup_steps=2000
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class CustomCLIPWrapper(CLIPWrapper):
    def __init__(self,
                 image_encoder,
                 text_encoder,
                 minibatch_size,
                 learning_rate=3e-3,# 此处设置了学习率
                 kl_coeff=1.0,
                 avg_word_embs=False
                 ):
        with open('models/configs/ViT.yaml') as fin:
            config = yaml.safe_load(fin)['ViT-L/14']
        super().__init__('ViT-L/14', config, minibatch_size)
        del self.model.visual
        del self.model.transformer
        self.model.visual = image_encoder
        self.model.transformer = text_encoder
        self.learning_rate = learning_rate
        self.avg_word_embs = avg_word_embs
        self.sink_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # init self-distillation model
        self.teacher = copy.deepcopy(self.model)
        self.kl_coeff = kl_coeff

    # assert image.device == self.model.device, "Image and model are on different devices!"
    # assert text["input_ids"].device == self.model.device, "Text and model are on different devices!"

    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image, text = train_batch
        n = math.ceil(len(image) // self.minibatch_size)
        image_mbs = torch.chunk(image, n)
        text_mbs_ids = torch.chunk(torch.arange(len(image)), n)

        # adjust embedding dictionaries
        text_mbs = []
        for s in text_mbs_ids:
            d = {}
            for key in list(text.keys()):
                d[key] = text[key][s]
            text_mbs.append(d)
        # fc_layer = nn.Linear(1000, 768)
        # fc_layer = fc_layer.to(image_mbs[0].device)
        # calculate original statistics
        with torch.no_grad():
            # image_embedding = self.model.encode_image(image_mbs[0])
            # ims = [F.normalize(fc_layer(image_embedding))]
            ims = [F.normalize(self.model.encode_image(im), dim=1) for im in image_mbs]
            txt = [F.normalize(self.encode_text(t), dim=1) for t in text_mbs]
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            txt = self.all_gather(torch.cat(txt))

            if len(ims.shape) == 3:
                ims = list(ims)
                txt = list(txt)
            else:
                ims = [ims]
                txt = [txt]

            image_logits_notemp = torch.cat(ims) @ torch.cat(txt).t()
            image_logits = image_logits_notemp * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            # calculate teacher
            teacher_ims = [F.normalize(self.teacher.encode_image(im), dim=1) for im in image_mbs]
            teacher_txt = [F.normalize(self.encode_text(t, teacher=True), dim=1) for t in text_mbs]

            teacher_ims = self.all_gather(torch.cat(teacher_ims))
            teacher_txt = self.all_gather(torch.cat(teacher_txt))

            if len(teacher_ims.shape) == 3:
                teacher_ims = list(teacher_ims)
                teacher_txt = list(teacher_txt)
            else:
                teacher_ims = [teacher_ims]
                teacher_txt = [teacher_txt]

            sim_ii, sim_tt, sim_it, sim_ti = self.compute_similarities(torch.cat(teacher_ims), torch.cat(teacher_txt))

            # optimal transport
            img_cost = - (sim_ii + sim_tt + sim_it)
            txt_cost = - (sim_ii + sim_tt + sim_ti)
            img_target = self.sinkhorn(img_cost)
            txt_target = self.sinkhorn(txt_cost)
            loss += (F.kl_div(F.log_softmax(image_logits_notemp * self.sink_temp, dim=-1), img_target, reduction='batchmean') + F.kl_div(F.log_softmax(image_logits_notemp.t() * self.sink_temp, dim=-1), txt_target, reduction='batchmean')) / 2 * self.kl_coeff
            self.log_dict({'loss': loss / len(ims), 'acc': (acc_i + acc_t) / 2 / len(image) / len(ims)}, prog_bar=True)

        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        # image loss
        for j, mb in enumerate(image_mbs):
            images_tmp = copy.deepcopy(ims)
            images_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.encode_image(mb), dim=1)
            image_logits_notemp = torch.cat(images_tmp) @ torch.cat(txt).t()
            image_logits = image_logits_notemp * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            loss += (F.kl_div(F.log_softmax(image_logits_notemp * self.sink_temp, dim=-1), img_target, reduction='batchmean') + F.kl_div(F.log_softmax(image_logits_notemp.t() * self.sink_temp, dim=-1), txt_target, reduction='batchmean')) / 2 * self.kl_coeff
            self.manual_backward(loss)

        # text loss
        for j, mb in enumerate(text_mbs):
            text_tmp = copy.deepcopy(txt)
            text_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.encode_text(mb), dim=1)
            image_logits_notemp = torch.cat(ims) @ torch.cat(text_tmp).t()
            image_logits = image_logits_notemp * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            loss += (F.kl_div(F.log_softmax(image_logits_notemp * self.sink_temp, dim=-1), img_target, reduction='batchmean') + F.kl_div(F.log_softmax(image_logits_notemp.t() * self.sink_temp, dim=-1), txt_target, reduction='batchmean')) / 2 * self.kl_coeff
            self.manual_backward(loss)

        # print(image.device)
        # print(self.model.device)
        # print(text[0].device)
        # assert image.device == self.model.device, "Image and model are on different devices!"
        # assert text["input_ids"].device == self.model.device, "Text and model are on different devices!"

        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))
        self.sink_temp.data.clamp_(-np.log(100), np.log(100))
        self.update_teacher()

    def encode_text(self, inputs, teacher=False):
        if self.avg_word_embs:
            sequence_output = self.teacher.transformer(**inputs)[0] if teacher else self.model.transformer(**inputs)[0]

            embeddings = torch.sum(
                sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
            ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)

            return embeddings
        else:
            return self.teacher.transformer(**inputs)[1] if teacher else self.model.transformer(**inputs)[1]

    def compute_similarities(self, I_emb, T_emb):
        sim_ii, sim_tt = I_emb @ I_emb.t(), T_emb @ T_emb.t()
        sim_it, sim_ti = I_emb @ T_emb.t(), T_emb @ I_emb.t()
        return sim_ii, sim_tt, sim_it, sim_ti

    def update_teacher(self):
        for teacher, student in zip(self.teacher.parameters(), self.model.parameters()):
            teacher.data.copy_(self.ema(student.data, teacher.data))

    def ema(self, s, t):
        return s * (1 - 0.999) + t * 0.999

    def forward(self, images, text):
        logits = F.normalize(self.model.encode_image(images), dim=1) @ F.normalize(self.encode_text(text), dim=1).t() * self.model.logit_scale.exp()
        return logits, logits.t()

    # Sourced from: https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/main_swav.py#L354
    def sinkhorn(self, out):
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def configure_optimizers(self):
        lr = self.learning_rate
        model = self.model
        # Rmax = 10
        # if self.model_name=="ViT-L/14":
        #     Rmax = 23

        no_smaller = ['class_embedding','prompt_embedding']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_smaller)],
                "lr": 0.000,
                "requires_grad": False
                # "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_smaller)],
                # "lr": 0.0001 * 1,
                "lr" : {
                    "RN50": 5e-4,
                    "RN101": 5e-4,
                    "RN50x4": 5e-4,
                    "RN50x16": 4e-4,
                    "RN50x64": 3.6e-4,
                    "ViT-B/32": 5e-4,
                    "ViT-B/16": 5e-4,
                    "ViT-L/14": 4e-4,
                    "ViT-L/16": 4e-4,
                    "ViT-L/14-336px": 2e-5
                }[self.model_name]
            }
        ]


        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,

            # self.parameters(),
            # lr=lr,
            momentum=0.9
        )

        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            # first_cycle_steps=self.num_training_steps,
            first_cycle_steps=20000,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=0,
            warmup_steps=2000
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class CLIPWrapper2(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 config: dict,
                 minibatch_size: int,
                 model_path: str =None # 默认为空
                 ):
        """A lightning wrapper for a CLIP model as specified in the paper.

        Args:
            model_name (str): A case sensitive visual model name.
            config (dict): A dictionary containing the CLIP instantiation parameters.
        """
        super().__init__()

        self.model_name = model_name
        self.model = CLIP(**config)

        self.model_path = model_path
        if (model_path is not None) and platform.system() == 'Linux':
            self.model_path = '/workspace/DATA/xpj/model/ViT-L-14.pt'

        elif model_path is not None:
            lst = self.model_path.split(".")
            if lst[-1] == 'pt':
                pretrained_model = torch.jit.load(self.model_path,map_location="cpu")
                checkpoint = torch.load(r'E:\xpj\models\rsvit\vit-b-checkpoint-1599.pth', map_location='cpu')['model']
                
                # self.model.load_state_dict(checkpoint, strict=False)
                # self.model, process = clip.load('ckpt/ViT-L-14.pt')
                # self.model.load_state_dict(pretrained_model.state_dict(), strict=False)
                self.model.load_state_dict(pretrained_model.state_dict(),strict=False)
                # self.model.visual.load_state_dict(checkpoint,strict=False)
            elif lst[-1] == 'ckpt':
                raise ValueError("ckpt 保存的是clipwrapper模型，请到train.py/load.py中加载")
            else:
                if lst[-1] not in ['pt', 'ckpt']:
                    raise ValueError(f"意外的文件扩展名: {lst[-1]}。期望是 'pt' 或 'ckpt'。")
        self.minibatch_size = minibatch_size
        self.isViT = 'ViT' in self.model_name

        self.automatic_optimization = False

    # Sourced from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = len(dataset)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    # Training loss: https://github.com/openai/CLIP/issues/83
    # Mini-batching thanks to https://github.com/crowsonkb / https://twitter.com/RiversHaveWings
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image, text = train_batch
        n = math.ceil(len(image) // self.minibatch_size)
        image_mbs = torch.chunk(image, n)
        text_mbs = torch.chunk(text, n)

        # calculate original statistics
        with torch.no_grad():
            ims = [F.normalize(self.model.encode_image(im), dim=1) for im in image_mbs]
            txt = [F.normalize(self.model.encode_text(t), dim=1) for t in text_mbs]
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            txt = self.all_gather(torch.cat(txt))

            if len(ims.shape) == 3:
                ims = list(ims)
                txt = list(txt)
            else:
                ims = [ims]
                txt = [txt]

            image_logits = torch.cat(ims) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(
                2)
            # 将交叉熵损失替换为KL散度 想法：就简单换个torch.nn.function函数:cross_entropy->KLDivLoss 但是KL度量的是两个分布之间的不相似性
            # loss = (F.kl_div(image_logits, ground_truth,reduction='batchmean') + F.kl_div(image_logits.t(), ground_truth,reduction='batchmean')).div(2)
            # loss = (F.kl_div(torch.cat(ims),torch.cat(txt),reduction='batchmean')+F.kl_div(torch.cat(txt),torch.cat(ims),reduction='batchmean')).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            self.log_dict({'loss': loss / len(ims), 'acc': (acc_i + acc_t) / 2 / len(image) / len(ims)}, prog_bar=True)

        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        # 原来在这里进行梯度清零，挪到了计算loss的下面，看看会不会收敛，不收敛应该改回来
        # optimizer.zero_grad()

        # image loss
        for j, mb in enumerate(image_mbs):
            images_tmp = copy.deepcopy(ims)
            images_tmp[self.global_rank][j * self.minibatch_size:(j + 1) * self.minibatch_size] = F.normalize(
                self.model.encode_image(mb), dim=1)
            image_logits = torch.cat(images_tmp) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            print(image_logits.requires_grad)
            print(ground_truth.requires_grad)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)) / 2
            # loss = (F.kl_div(torch.cat(txt), torch.cat(ims)) + F.kl_div(torch.cat(ims), torch.cat(txt))) / 2
            self.manual_backward(loss)

        # text loss
        for j, mb in enumerate(text_mbs):
            text_tmp = copy.deepcopy(txt)
            text_tmp[self.global_rank][j * self.minibatch_size:(j + 1) * self.minibatch_size] = F.normalize(
                self.model.encode_text(mb), dim=1)
            image_logits = torch.cat(ims) @ torch.cat(text_tmp).t() * self.model.logit_scale.exp()
            image_logits.requires_grad = True
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)) / 2
            # loss.requires_grad = True
            # loss = (F.kl_div(torch.cat(txt), torch.cat(ims)) + F.kl_div(torch.cat(ims), torch.cat(txt))) / 2
            self.manual_backward(loss)

        accumulate = False
        if not accumulate:
            optimizer.step()
            optimizer.zero_grad()
            # 关闭学习率调度器
            # lr_scheduler = self.lr_schedulers()
            # lr_scheduler.step()
            self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))

        # grad_accumulation
        else:
            N = 2
            if (idx + 1) % N == 0:
                optimizer.step()
                optimizer.zero_grad()
                # lr_scheduler = self.lr_schedulers()
                # lr_scheduler.step()
                self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))

    def validation_step(self, val_batch, idx):
        self.model.eval()
        with torch.no_grad():
            image, text = val_batch
            image_logits, text_logits = self.forward(image, text)
            ground_truth = torch.arange(len(image_logits)).to('cuda')
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(text_logits, ground_truth)).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()

            # self.log('val_loss', loss)
            self.log_dict({'val_loss': loss, 'val_acc': (acc_i + acc_t) / 2 / len(image) }, prog_bar=True)

    def forward(self, images, text):
        logits = F.normalize(self.model.encode_image(images), dim=1) @ F.normalize(self.model.encode_text(text), dim=1).t() * self.model.logit_scale.exp()
        return logits, logits.t()

    def configure_optimizers(self):
        lr = {
            "RN50": 5e-4,
            "RN101": 5e-4,
            "RN50x4": 5e-4,
            "RN50x16": 4e-4,
            "RN50x64": 3.6e-4,
            "ViT-B/32": 5e-4,
            "ViT-B/16": 5e-4,
            "ViT-L/14": 4e-4,
            "ViT-L/14-336px": 2e-5
        }[self.model_name]
        lr =lr

        model = self.model

        no_smaller = [
            'class_embedding', 'prompt_embedding'
        ]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.visual.named_parameters() if not any(nd in n for nd in no_smaller)],
                "lr": lr /100,
                # "requires_grad": False
                # "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.visual.named_parameters() if any(nd in n for nd in no_smaller)],
                "lr": lr,

            }
        ]
        grad_parameters = [
            {
                "params": [p for n, p in model.visual.named_parameters() if any(nd in n for nd in no_smaller)],
                "lr": lr,

            }
        ]
        all_parameters = [
            {
                "params": model.named_parameters(),
                "lr": lr/100,
            }
        ]

        optimizer = torch.optim.AdamW(
            # 筛选requires_grad ==True
            # filter(lambda p: p.requires_grad, self.model.parameters()),
            # self.model.parameters(),
            optimizer_grouped_parameters,
            # grad_parameters,
            # lr=lr,
            betas=(
                0.9,
                0.98 if self.isViT else 0.999
            ),
            eps=1e-6 if self.isViT else 1e-8,
            weight_decay=0.2
        )

        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            # first_cycle_steps=self.num_training_steps,
            first_cycle_steps=20000,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=0,
            warmup_steps=2000
        )

        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        return {'optimizer': optimizer}

