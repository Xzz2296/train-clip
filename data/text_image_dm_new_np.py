# Originally found in https://github.com/lucidrains/DALLE-pytorch
import os.path
from pathlib import Path
from random import randint, choice
import numpy as np
import PIL
import argparse
import clip
import torch

import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule


class TextImageDataset(Dataset):
    def __init__(self,
                 folder: str,
                 train_val_file:str,
                 image_size=224,
                 resize_ratio=0.75,
                 shuffle=False,
                 custom_tokenizer=False
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)
        self.path = path
        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]
        with open(train_val_file, 'r') as file:
            train_list = [line.strip() for line in file.readlines()]
        text_files = {text_file.stem: text_file for text_file in text_files if text_file.stem in train_list}
        image_files = {image_file.stem: image_file for image_file in image_files if image_file.stem in train_list}

        keys = (image_files.keys() & text_files.keys())
        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        # self.image_files = data_dict
        self.resize_ratio = resize_ratio
        self.image_transform = T.Compose([
            T.Lambda(self.fix_img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.custom_tokenizer = custom_tokenizer

    def __len__(self):
        return len(self.keys)
    
    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)


    def __getitem__(self, ind):
        path = os.path.join(self.path,"npz")
        key = self.keys[ind]

        text_file = self.text_files[key]
        # image_file = self.image_files[key]
        image_file_npz_path = os.path.join(path, f"{key}.npz")
        # 添加 按utf-8方式编码
        t1 = time.time()
        descriptions = text_file.read_text(encoding='utf-8').split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        tokenized_text = description if self.custom_tokenizer else clip.tokenize(description, truncate=True)[0]
        t2 = time.time()

        try:
            image_np = np.load(image_file_npz_path)['data']
            image_tensor = torch.from_numpy(image_np)
            # image_tensor = self.lazy_load_image(image_file)
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            # print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        t3 = time.time()

        print(f"text:{1000*(t2 -t1)}s")
        print(f"image:{1000*(t3- t2)}s")
        # Success
        return image_tensor, tokenized_text

class TextImageDataModule(LightningDataModule):
    def __init__(self,
                 folder: str,
                 # folder2: str,
                 batch_size: int,
                 num_workers=0,
                 image_size=224,
                 resize_ratio=0.75,
                 shuffle=False,
                 custom_tokenizer=None
                 ):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (transformers.AutoTokenizer, optional): The tokenizer to use on the text. Defaults to None.
        """
        super().__init__()
        self.folder =folder
        # self.folder2 =folder2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.shuffle = shuffle
        self.custom_tokenizer = custom_tokenizer
    
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--folder', type=str, required=True, help='directory of your training folder')
        # parser.add_argument('--folder2', type=str, help='directory of your val folder')
        parser.add_argument('--batch_size', type=int, help='size of the batch')
        parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
        parser.add_argument('--image_size', type=int, default=224, help='size of the images')
        parser.add_argument('--resize_ratio', type=float, default=0.75, help='minimum size of images during random crop')
        parser.add_argument('--shuffle', type=bool, default=False, help='whether to use shuffling during sampling')
        return parser
    
    def setup(self, stage=None):
        self.dataset = TextImageDataset(self.folder,train_val_file='train_set.txt', image_size=self.image_size, resize_ratio=self.resize_ratio, shuffle=self.shuffle, custom_tokenizer=not self.custom_tokenizer is None)
        self.dataset2 = TextImageDataset(self.folder,train_val_file='val_set.txt', image_size=self.image_size,resize_ratio=self.resize_ratio,shuffle=self.shuffle,custom_tokenizer=not  self.custom_tokenizer is None)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True , collate_fn=self.dl_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset2,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,drop_last=True , collate_fn=self.dl_collate_fn)
    
    def dl_collate_fn(self, batch):
        if self.custom_tokenizer is None:
            return torch.stack([row[0] for row in batch]), torch.stack([row[1] for row in batch])
        else:
            return torch.stack([row[0] for row in batch]), self.custom_tokenizer([row[1] for row in batch], padding=True, truncation=True, return_tensors="pt")
