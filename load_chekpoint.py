import platform
import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CLIPWrapper

MODEL_NAME ='ViT-L/16'

DEVICE ='cuda'

CHECKPOINT ='chek/vit_l_16-852ce7e3.pth'

config_dir = 'models/configs/ViT.yaml' if 'ViT' in MODEL_NAME else 'models/configs/RN.yaml'
with open(config_dir) as fin:
    config = yaml.safe_load(fin)[MODEL_NAME ]

model = CLIPWrapper.load_from_checkpoint(CHECKPOINT, model_name=MODEL_NAME, config=config, minibatch_size=1).model.to(DEVICE)

print("1")