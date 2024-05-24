# Description: load vision transformer model and use it to encode images
# from Visualizer_main.visualizer import get_local
from functools import partial
import torch
import torch.nn as nn
import yaml
import torchvision
from torchvision import transforms

# 修改了timm库的vision_transformer，增加了可学习参数，使用时需在工作目录下拷贝timm文件夹
from models.wrapper_timm import CLIPWrapper2


config_dir = 'models/configs/ViT.yaml'
with open(config_dir) as fin:
    config = yaml.safe_load(fin)['ViT-L/14']

# 初始化一个CLIP模型，视觉编码器调整为了ViT-B/16并丢掉了project投影，但其他参数仍然是ViT-L/14对应的CLIP模型的参数
clip_model = CLIPWrapper2('ViT-L/14', config, 2)
checkpoint = torch.load(r"E:\xpj\research\VPT\checkpoint\rs_vit_b_vpt.ckpt")['state_dict']
clip_model.load_state_dict(checkpoint, strict=True)
clip_model.eval().to('cuda')
# 获取模型的视觉编码器即VPT微调后的ViT-B模型
model = clip_model.model.visual


# 数据文件夹，一个子文件夹对应一个类别
image_folder = r'E:\xpj\dataset\clip_test'

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 使用dataset和dataloader读取数据
dataset = torchvision.datasets.ImageFolder(image_folder, transform=transform)
# linux下可使用多线程读取数据
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)


with torch.no_grad():
    inputs, labels = next(iter(dataloader))
    inputs = inputs.to('cuda')
    # 使用模型对输入进行编码,输出形状为（batch_size, embedding_dim） 即batch_size个 cls token
    outputs = model(inputs)
    # 将outputs 切分成batch_size个样本
    outputs = torch.split(outputs, 1, dim=0)


print(outputs)
print("load successfully!")

