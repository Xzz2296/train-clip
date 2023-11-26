import torch

# 加载checkpoint
checkpoint = torch.load('ckpt/epoch=31-step=5823.ckpt')
model = checkpoint['state_dict']
# 查看checkpoint中的键
print(model)
print(checkpoint.keys())
