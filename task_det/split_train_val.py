import os
import shutil
import random
from tqdm import tqdm


def split_data_from_txt(input_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split data into train, val, and test sets using file paths from train.txt and organize them into a new folder structure.
    :param input_folder: Path to the input folder (e.g., merge_training with images, labels, and train.txt).
    :param output_folder: Path to the output folder for split data.
    :param train_ratio: Ratio of data for training (default 80%).
    :param val_ratio: Ratio of data for validation (default 10%).
    :param test_ratio: Ratio of data for testing (default 10%).
    """
    
    images_folder = os.path.join(input_folder, 'images')
    labels_folder = os.path.join(input_folder, 'new_format_labels')
    train_txt_file = os.path.join(input_folder, 'train.txt')

    with open(train_txt_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]

    
    random.shuffle(image_paths)

    
    total = len(image_paths)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = 1

    print("TRAIN SIZE: ", train_size)
    print("VAL SIZE: ", val_size)
    print("TEST SIZE: ", test_size)
    
    
    train_data = image_paths[:train_size]
    val_data = image_paths[train_size:train_size + val_size]
    test_data = image_paths[-1]

    
    os.makedirs(output_folder, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_folder, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, split, 'labels'), exist_ok=True)

    
    def copy_and_create_txt(split, data):
        split_images_folder = os.path.join(output_folder, split, 'images')
        split_labels_folder = os.path.join(output_folder, split, 'labels')
        split_txt = os.path.join(output_folder, f"{split}.txt")

        with open(split_txt, 'w') as txt_file:
            for img_path in data:
                print(img_path)
                
                image_name = os.path.basename(img_path)
                label_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                
                
                src_image = os.path.join(input_folder, img_path)
                src_label = os.path.join(labels_folder, label_name)
                
                
                dst_image = os.path.join(split_images_folder, image_name)
                dst_label = os.path.join(split_labels_folder, label_name)

                
                shutil.copy2(src_image, dst_image)
                shutil.copy2(src_label, dst_label)

                
                
                txt_file.write(f"./images/{image_name}\n")

    
    print("MAKE DATA TRAIN ....")
    copy_and_create_txt('train', train_data)
    print("MAKE DATA VAL ....")
    copy_and_create_txt('val', val_data)
    print("MAKE DATA TEST ....")
    copy_and_create_txt('test', test_data)

    print(f"Data successfully split and saved to {output_folder}")


output_folder = "../obj_det_cls/data_training"  # source data training for single shot or 4cls with folder labels or new_format_labels
output_folder_split = "./split_data_training_4cls"  
split_data_from_txt(output_folder, output_folder_split, train_ratio=0.65, val_ratio=0.35, test_ratio=0)

