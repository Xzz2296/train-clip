import torch
import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper, CLIPWrapper2
from torchvision.models import resnet50,vit_l_16
from transformers import AutoTokenizer, AutoModel
import clip

def main(hparams):
    img_encoder = resnet50(pretrained=True)
    img_encoder.fc = torch.nn.Linear(2048, 768)
    # clip_model, process =clip.load('chek/ViT-L-14.pt')
    # clip_model =clip_model.cuda()
    # img_encoder = clip_model.visual.cuda()
    # for name,param in img_encoder.named_parameters():
    #     print (param.device)
    # print()
    # print(clip_model.visual)
    # tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
    config_dir = 'models/configs/ViT.yaml' if 'ViT' in hparams.model_name else 'models/configs/RN.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]
    # tokenizer = AutoTokenizer.from_pretrained(r"E:\xpj\models\declutr-sci-base")
    # txt_encoder = AutoModel.from_pretrained(r"E:\xpj\models\declutr-sci-base")

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    # model = CustomCLIPWrapper(img_encoder, txt_encoder, hparams.minibatch_size, avg_word_embs=True)
    model = CLIPWrapper2(hparams.model_name, config, hparams.minibatch_size)
    dm = TextImageDataModule.from_argparse_args(hparams)

    trainer = Trainer.from_argparse_args(hparams,gpus=1 ,precision=32, max_epochs=32)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
