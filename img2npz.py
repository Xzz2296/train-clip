import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import clip

from torchvision import transforms as T


def fix_img(img):
    img.convert('RGB') if img.mode != 'RGB' else img

def preprocess_image(image_path):
    transform = T.Compose([
        # T.Lambda(fix_img()),
        T.RandomResizedCrop(224,
                            scale=(0.75, 1.),
                            ratio=(1., 1.)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    # 读取图像
    image = Image.open(image_path)
    image_tensor = transform(image)
    return image_tensor

# def preprocess_text(text_path):


def process_images_in_folder(folder_path, output_npz_file):
    data = []
    text = []
    keys = []

    # 遍历文件夹中的所有图像文件
    for filename in tqdm(os.listdir(folder_path), desc="Processing images"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 构造图像文件的完整路径
            image_path = os.path.join(folder_path, filename)

            # 获取去掉后缀的文件名作为key
            key = os.path.splitext(filename)[0]
            keys.append(key)

            # 图像预处理
            preprocessed_image = preprocess_image(image_path)

            # 将预处理后的图像添加到数据列表中
            data.append(preprocessed_image)

    # 将数据和keys保存到npz文件中
    np.savez(output_npz_file, data=data, keys=keys)


if __name__ == "__main__":
    # 设置文件夹路径和输出npz文件名
    input_folder = r"E:\xpj\dataset\clip\train\image"
    output_npz_file = "output.npz"

    # 处理图像并保存到npz文件
    process_images_in_folder(input_folder, output_npz_file)


