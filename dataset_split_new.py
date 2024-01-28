import os
import random
# import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(data_folder, image_folder, text_folder, per_class_samples=300000, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    # 获取所有文件
    class_folders = [class_folder for class_folder in os.listdir(os.path.join(data_folder, text_folder)) if os.path.isdir(os.path.join(data_folder, text_folder, class_folder))]
    npz_files = []
    for class_folder in class_folders:
        class_folder_path = os.path.join(data_folder, text_folder, class_folder)
        path =Path(class_folder_path)
        # path_img = Path(os.path.join(data_folder,image_folder))
        text_files = [*path.glob('**/*.txt')]
        
        # text_files = set(os.listdir(os.path.join(data_folder, text_folder, class_folder)))
        # image_files = set(os.listdir(os.path.join(data_folder, image_folder, class_folder)))
        # common_files = text_files.intersection(image_files)
        if len(text_files) < per_class_samples:
            raise ValueError(f"Not enough samples in class {class_folder}. Only {len(text_files)} samples found.")
        npz_files.extend(random.sample(text_files, per_class_samples))

    # 计算划分的数量
    total_samples = len(npz_files)
    num_train = int(total_samples * train_ratio)
    num_val = int(total_samples * val_ratio)
    num_test = total_samples - num_train - num_val

    # 划分数据集
    train_files, test_files = train_test_split(npz_files, test_size=num_test, random_state=random_seed)
    train_files, val_files = train_test_split(train_files, test_size=num_val, random_state=random_seed)

    # 保存结果到文件
    save_to_file(train_files, 'dataset/90w_c/train_set.txt')
    save_to_file(val_files, 'dataset/90w_c/val_set.txt')
    save_to_file(test_files, 'dataset/90w_c/test_set.txt')

def save_to_file(file_lists, filename):
    with open(filename, 'w') as file:
        for npz_file in file_lists:
            file.write(f"{npz_file}\n")

# 示例用法
data_folder = '/workspace/DATA/xpj/dataset/'
image_folder = 'img_data'
text_folder = 'prompt_text'
t1 =time.time()
split_dataset(data_folder, image_folder, text_folder)

t2 =time.time()
print(t2-t1)

