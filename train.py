import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CLIPWrapper


def main(hparams):
    config_dir = 'models/configs/ViT.yaml' if 'ViT' in hparams.model_name else 'models/configs/RN.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    model = CLIPWrapper(hparams.model_name, config, hparams.minibatch_size)
    params =list(model.named_parameters())
    # 枚举参数名称和 形状
    for name,param in model.named_parameters():
        #print(i)
        print(name)
        print(param.size())
    # 共3个参数需要训练：model.visual.prompt_embeddings model.visual.transformer.prompt_embeddings model.visual.class_embedding
    print(params)
    # no_smaller = ['model.visual.prompt_embeddings', 'model.visual.transformer.prompt_embeddings', 'model.visual.class_embedding']
    #
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_smaller)],
    #         "lr": 0.0001
    #         # "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_smaller)],
    #         "lr": 0.0001
    #     }
    # ]

    # 冻结全部参数
    # for param in model.parameters():
    #     param.requires_grad = False
    
    del hparams.model_name


    dm = TextImageDataModule.from_argparse_args(hparams)
    trainer = Trainer.from_argparse_args(hparams, gpus=[0], precision=16, max_epochs=32)
    # 添加gpus=x参数
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    #parser.add_argument('--batchsize', type=int)
    parser.add_argument('--minibatch_size', type=int, default=0)
    # parser.add_argument('--gpus', default=0, type=int)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
