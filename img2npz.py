import os

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T


def fix_img(img):
    return img.convert('RGB') if img.mode != 'RGB' else img


def preprocess_image(image_path):
    transform = T.Compose([
        T.Lambda(fix_img),
        T.RandomResizedCrop(224,
                            scale=(0.75, 1.),
                            ratio=(1., 1.)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    # 读取图像
    image = Image.open(image_path)
    image_tensor = transform(image)

    # 将张量转换为 NumPy 数组，指定dtype为object
    image_array = np.array(image_tensor, dtype=np.float16)

    return image_array


def process_images_in_folder(folder_path, output_folder):
    # data = []
    # keys = []

    # 遍历文件夹中的所有图像文件
    for filename in tqdm(os.listdir(folder_path), desc="Processing images"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 构造图像文件的完整路径
            image_path = os.path.join(folder_path, filename)
            # 获取去掉后缀的文件名作为key
            key = os.path.splitext(filename)[0]
            output_npz_file =os.path.join(output_folder, f"{key}.npz")

            # 图像预处理
            preprocessed_image = preprocess_image(image_path)
            # 将数据和keys保存到npz文件中，指定dtype为object
            np.savez(output_npz_file, data=preprocessed_image,  dtype=np.float16)


if __name__ == "__main__":
    # 设置文件夹路径和输出npz文件名
    input_folder = r"E:\xpj\dataset\clip\train\image"
    output_folder = r"E:\xpj\dataset\clip\train\npz"

    # 处理图像并保存到npz文件
    process_images_in_folder(input_folder, output_folder)



