import torch

# 加载checkpoint
# checkpoint = torch.load('ckpt/epoch=31-step=5823.ckpt')
checkpoint = torch.load('ckpt/ViT-L-14.pt')
# model = checkpoint['state_dict']
model = checkpoint
# 查看checkpoint中的键
print(model)
# print(model.keys())
# print(checkpoint.keys())
