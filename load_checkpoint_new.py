import torch
import torch.nn as nn
from torchvision import transforms as T
import yaml
from PIL import Image
from models import CLIPWrapper2
from data.text_image_dm_new import TextImageDataset
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"

image_folder = r'E:\xpj\dataset\clip_test'

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
    checkpoint = torch.load('ckpt/gdp_270w.ckpt')['state_dict']
    model.load_state_dict(checkpoint)
    model.eval()

    dataset = torchvision.datasets.ImageFolder(image_folder, transform=transform)
    # linux下可使用多线程读取数据
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # linear_layer = nn.Linear(768, 512).to(device)  # 全连接层
    with torch.no_grad():
        inputs, labels = next(iter(dataloader))
        inputs = inputs.to(device)
        # 使用模型对输入进行编码,输出形状为（batch_size, embedding_dim） 即batch_size个 cls token
        outputs = model(inputs)
        # 将outputs 切分成batch_size个样本
        outputs = torch.split(outputs, 1, dim=0)


if __name__ == '__main__':
    main()
