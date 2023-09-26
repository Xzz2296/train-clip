import argparse
from pytorch_lightning import Trainer

parser = argparse.ArgumentParser(description="Training script")

# 把trainer 中的参数加到parser中
parser = Trainer.add_argparse_args(parser)

# 添加自定义的命令行参数
parser.add_argument('--custom_param', type=int, default=10, help='Custom hyperparameter')

# 进行参数解析 并存储在args中
args = parser.parse_args()

# 使用 args 来访问命令行参数
print(args.gpus)  # 访问 Trainer 中的参数
print(args.custom_param)  # 访问自定义的参数
