import os
train_val_file =r"E:\github\train-CLIP\dataset\train_set.txt"
img_dataset_folder = r"workspace/DATA/xpj/dataset/img_data"

def get_img_path_from_txt(train_val_file,img_dataset_folder):
    with open(train_val_file, 'r') as file:
        train_list = [line.strip() for line in file.readlines()]
    #train_list 是存放训练文本位置的列表，我要根据这个列表找到对应的图片，将图片的位置存放到一个新的列表中
        train_list_img = []
        for text_file_path in train_list:
            basename = os.path.basename(text_file_path)
            city =text_file_path.split('/')[-2]
            city_npz =city+"_npz" 
            img_path = os.path.join(img_dataset_folder,city_npz,basename)
            if os.path.exists(img_path):
                train_list_img.append(img_path)
    return train_list_img    

if __name__ == "__main__":
    train_list_img = get_img_path_from_txt(train_val_file,img_dataset_folder)
    print(train_list_img[0])        

# print("hello")