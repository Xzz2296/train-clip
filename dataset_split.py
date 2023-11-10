import os
import random
from sklearn.model_selection import train_test_split

def split_dataset(data_folder, image_folder, text_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    # 获取所有文件
    image_files = [file for file in os.listdir(os.path.join(data_folder, image_folder)) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    text_files = [file for file in os.listdir(os.path.join(data_folder, text_folder)) if file.endswith('.txt')]

    # 根据文件名进行分组
    image_file_groups = {}
    for file in image_files:
        key = os.path.splitext(file)[0]  # 移除文件后缀
        image_file_groups.setdefault(key, []).append(file)

    text_file_groups = {}
    for file in text_files:
        key = os.path.splitext(file)[0]  # 移除文件后缀
        text_file_groups.setdefault(key, []).append(file)

    # 获取共同的文件名
    keys = set(image_file_groups.keys()) & set(text_file_groups.keys())

    # 随机化分组顺序
    random.seed(random_seed)
    keys = list(keys)
    random.shuffle(keys)

    # 计算划分的数量
    total_samples = len(keys)
    num_train = int(total_samples * train_ratio)
    num_val = int(total_samples * val_ratio)
    num_test = total_samples - num_train - num_val

    # 划分数据集
    train_keys, test_keys = train_test_split(keys, test_size=num_test, random_state=random_seed)
    train_keys, val_keys = train_test_split(train_keys, test_size=num_val, random_state=random_seed)

    # 将文件名转换为完整的文件路径
    train_files = [os.path.join(file.split('.')[0]) for key in train_keys for file in text_file_groups[key]]
    val_files = [os.path.join(file.split('.')[0]) for key in val_keys for file in text_file_groups[key]]
    test_files = [ os.path.join(file.split('.')[0]) for key in test_keys for file in text_file_groups[key]]

    # 保存结果到文件
    save_to_file(train_files, 'train_set.txt')
    save_to_file(val_files, 'val_set.txt')
    save_to_file(test_files, 'test_set.txt')

def save_to_file(file_lists, filename):
    with open(filename, 'w') as file:
        for text_file in file_lists:
            file.write(f"{text_file}\n")

# 示例用法
data_folder = r'E:\xpj\dataset\clip\train'
image_folder = 'image'
text_folder = 'text'
split_dataset(data_folder, image_folder, text_folder)
