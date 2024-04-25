# Description: load vision transformer model and use it to encode images
from Visualizer_main.visualizer import get_local
get_local.activate()
from functools import partial
import torch
import torch.nn as nn

import timm.models.vision_transformer
# import Visualizer_main.timm.models.vision_transformer
import torchvision
from torchvision import transforms


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0,**kwargs)
    return model



model = vit_base_patch16()
checkpoint = torch.load(r'E:\xpj\models\rsvit\vit-b-checkpoint-1599.pth', map_location='cpu')['model']
model.load_state_dict(checkpoint, strict=False)
model.eval().to('cuda')

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
get_local.clear()
with torch.no_grad():
    # 读取一个batch的数据
    inputs, labels = next(iter(dataloader))
    inputs = inputs.to('cuda')
    # 使用模型对输入进行编码,输出形状为（batch_size, embedding_dim） 即batch_size个 cls token
    outputs = model(inputs)
    # 将outputs 切分成batch_size个样本
    outputs = torch.split(outputs, 1, dim=0)
    cache = get_local.cache
    print(list(cache.keys()))
    print(cache)


print(outputs)
print("load success!")
# print(model)
