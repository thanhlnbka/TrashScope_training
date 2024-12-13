import os
import shutil
import random
from tqdm import tqdm

def merge_multiple_folders(input_folders, output_folder):
    """
    Merge multiple folders into a single folder with images, labels, and a combined train.txt file.
    :param input_folders: List of folders to merge (each containing images, labels, and a train.txt).
    :param output_folder: Target folder to store the merged structure.
    """
    images_folder = os.path.join(output_folder, 'images')
    labels_folder = os.path.join(output_folder, 'labels')
    train_file = os.path.join(output_folder, 'train.txt')

    
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    with open(train_file, 'w') as train_out:
        for input_folder in input_folders:
            print("INPUT FOLDER: ", input_folder)
            input_images_folder = os.path.join(input_folder, 'images')
            input_labels_folder = os.path.join(input_folder, 'labels')
            input_train_file = os.path.join(input_folder, 'train.txt')

            
            if not os.path.exists(input_images_folder) or not os.path.exists(input_labels_folder):
                print(f"Skipping folder {input_folder}: missing images or labels folder.")
                continue

            
            
            for file_name in tqdm(os.listdir(input_images_folder)):                
                src_image = os.path.join(input_images_folder, file_name)
                src_label = os.path.join(input_labels_folder, file_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
                
                
                if os.path.exists(src_image) and os.path.exists(src_label):
                    
                    shutil.copy2(src_image, os.path.join(images_folder, file_name))
                    
                    shutil.copy2(src_label, os.path.join(labels_folder, file_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')))
                    
                    
                    train_out.write(os.path.join("./images",file_name) + '\n')
                else:
                    print(src_image)

    
    print(f"All data merged into {output_folder}")

input_root = "./data_format"
input_folders = ["input3_part1_batch123", "input3_part2_batch_1", "input3_part2_batch_2", "input3_part2_batch_3", "input3_part4_unknow"]  
input_folders_join = [f"{input_root}/{i}" for i in input_folders]

output_folder = "./data_training"  

merge_multiple_folders(input_folders_join, output_folder)


