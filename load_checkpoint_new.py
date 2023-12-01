import torch
import torch.nn as nn
from torchvision import transforms as T
import yaml
from PIL import Image
from models import CLIPWrapper2
from data.text_image_dm_new import TextImageDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    config_dir = 'models/configs/ViT.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)['ViT-L/14']
    # 对输入图像预处理的过程
    def fix_img(img):
        return img.convert('RGB') if img.mode != 'RGB' else img
    transform = T.Compose([
        T.Lambda(fix_img),
        T.RandomResizedCrop(224,
                            scale=(0.75, 1.),
                            ratio=(1., 1.)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    # 第三个参数8为batch大小，服务器上可以设置大一点，处理速度会更快一些
    # 加载预训练的VIT模型请在CLIPWrapper2的参数中进行设置 如CLIPWrapper2('ViT-L/14', config, 8，model_path='ckpt/ViT-L-14.pt')
    model = CLIPWrapper2('ViT-L/14', config, minibatch_size=8).to(device)
    # 加载微调后的模型请在下方checkpoint处加载 model为封装后的clip模型, model.model为CLIP模型，model.model.model为VIT编码器
    checkpoint = torch.load('ckpt/epoch=31-step=4959.ckpt')['state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    linear_layer = nn.Linear(768, 512).to(device)  # 全连接层
    with torch.no_grad():
        image = Image.open('test.jpg')
        image = transform(image).to(device)
        # 仅使用CLIP模型的视觉部分 即ViT
        feature_origin = model.model.encode_image(image)
        # 特征形状为n*768 ,n为送入图像的数量
        feature = linear_layer(feature_origin)
        # 经过全连接 从1*768 变成 1*512
        print(feature)


if __name__ == '__main__':
    main()
