import os
import concurrent.futures

train_val_file ="/workspace/DATA/xpj/github/train-clip-1101/train-clip-main/dataset/3w/train_set.txt"
img_dataset_folder = r"/workspace/DATA/xpj/dataset/img_data"

def check_path(text_file_path):
    basename = os.path.basename(text_file_path).split('.')[0]+'.npz'
    city =text_file_path.split('/')[-2]
    city_npz =city+"_npz" 
    img_path = os.path.join(img_dataset_folder,city_npz,basename)
    if os.path.exists(img_path):
        return img_path
    return None

def get_img_path_from_txt(train_val_file,img_dataset_folder="/workspace/DATA/xpj/dataset/img_data"):
    with open(train_val_file, 'r') as file:
        train_list = [line.strip() for line in file.readlines()]

    train_list_img = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(check_path, text_file_path): text_file_path for text_file_path in train_list}
        for future in concurrent.futures.as_completed(future_to_path):
            img_path = future.result()
            if img_path is not None:
                train_list_img.append(img_path)
    return train_list_img    

if __name__ == "__main__":
    train_list_img = get_img_path_from_txt(train_val_file,img_dataset_folder)
    print(train_list_img[0])        

# print("hello")