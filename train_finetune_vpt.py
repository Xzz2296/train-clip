import torch
import platform
import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm_val import TextImageDataModule
from models import CustomCLIPWrapper, CLIPWrapper2
from torchvision.models import resnet50,vit_l_16
from transformers import AutoTokenizer, AutoModel
import clip

def main(hparams):

    config_dir = 'models/configs/ViT.yaml' if 'ViT' in hparams.model_name else 'models/configs/RN.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    # model = CustomCLIPWrapper(img_encoder, txt_encoder, hparams.minibatch_size, avg_word_embs=True)
    model = CLIPWrapper2(hparams.model_name, config, hparams.minibatch_size)
    dm = TextImageDataModule.from_argparse_args(hparams)

    gpu_nums = 7
    if platform.system() == 'Windows':
        gpu_nums = 1

    trainer = Trainer.from_argparse_args(hparams,gpus=gpu_nums ,precision=16, max_epochs=12)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
