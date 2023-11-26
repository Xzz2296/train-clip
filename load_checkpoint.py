import torch
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
    # 指定ckpt路径即可
    model = CLIPWrapper2('ViT-L/14', config, 8).to(device)
    checkpoint = torch.load('ckpt/epoch=31-step=4959.ckpt')['state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        image = Image.open('test.jpg')
        image = transform(image).to(device)
        # 仅使用CLIP模型的视觉部分 即ViT
        feature = model.model.encode_image(image)
        # 特征形状为n*768 ,n为送入图像的数量
        print(feature)


if __name__ == '__main__':
    main()
