
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import os
from collections import defaultdict
import random
import csv 
import numpy as np

CLS1_LABELS = ["Clean", "Not_clean", "Partial_mix", "Total_mix", "Empty", "Unknown"]  
CLS2_LABELS = ["Covered", "Uncorved", "Unknown"]

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.cls1 = CLS1_LABELS
        self.cls2 = CLS2_LABELS  
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, cls1, cls2 = self.data[index]
        sample = Image.open(path).convert("RGB")
        if self.transform:
            sample = self.transform(sample)
        
        label1 = torch.tensor(self.cls1.index(cls1), dtype=torch.long)
        label2 = torch.tensor(self.cls2.index(cls2), dtype=torch.long)
        
        return sample, label1, label2
    
def create_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize the image
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(
            mean=timm.data.IMAGENET_DEFAULT_MEAN,  # Normalize using ImageNet default mean
            std=timm.data.IMAGENET_DEFAULT_STD   # Normalize using ImageNet default std
        )
    ])

def group_images_by_id(data_path):
    image_groups = defaultdict(list)
    for cls1 in os.listdir(data_path):
        class_path = os.path.join(data_path, cls1)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if os.path.isfile(file_path):
                    _filenames = file_name.split('_')
                    cls2 = _filenames[-1][:-4]
                    image_id = "_".join(_filenames[:4])
                    image_groups[image_id].append((file_path, cls1, cls2))
    return image_groups

def split_images_into_folds(image_groups, k=5):
    image_ids = list(image_groups.keys())
    random.shuffle(image_ids)

    folds = [[] for _ in range(k)]
    fold_sizes = [0] * k
    fold_class_counts_cls1 = [defaultdict(int) for _ in range(k)]  
    fold_class_counts_cls2 = [defaultdict(int) for _ in range(k)]  

    context_fold = []  

    for image_id in image_ids:
        group = image_groups[image_id]
        min_fold_idx = min(range(k), key=lambda i: fold_sizes[i])
        
        folds[min_fold_idx].extend(group)
        fold_sizes[min_fold_idx] += len(group)
        
        # Count num class (cls1 và cls2)
        for _, cls1, cls2 in group:
            fold_class_counts_cls1[min_fold_idx][cls1] += 1
            fold_class_counts_cls2[min_fold_idx][cls2] += 1

    
    total_samples = sum(fold_sizes)
    context_fold.append(f"\nTotal Samples: {total_samples}")
    for i, size in enumerate(fold_sizes):
        context_fold.append(f"\nFold {i + 1}: {size} samples")
        
        context_fold.append("  Class Distribution for cls1 (Cleanliness Status):")
        for cls1, count in sorted(fold_class_counts_cls1[i].items()): 
            context_fold.append(f"    Class {cls1}: {count}")
        
 
        context_fold.append("  Class Distribution for cls2 (Cover Status):")
        for cls2, count in sorted(fold_class_counts_cls2[i].items()): 
            context_fold.append(f"    Class {cls2}: {count}")
    
    return folds, context_fold


def save_context_to_file(context_fold, filename="training_fold.txt"):
    with open(filename, "w") as f:
        for line in context_fold:
            f.write(line + "\n")

def plot_confusion_matrix(cm, labels, title, save_path):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(save_path)
    plt.close()

def save_metrics_to_csv(fold_dir, metrics, fold_idx, epoch):
    # Define directory and file path
    metrics_dir = f"{fold_dir}/metrics_results"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "metrics.csv")
    
    # Use DictWriter to handle the writing of metrics
    fieldnames = ["Fold", "Epoch", "Train_Acc1", "Train_Acc2", "Val_Acc1", "Val_Acc2", "Val_F1_Cls1", "Val_F1_Cls2", "Best_Loss"]

    # Open the CSV file for appending
    with open(metrics_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if the file is empty
        if f.tell() == 0:
            writer.writeheader()

        # Write the metrics for the current epoch
        writer.writerow({
            "Fold": fold_idx,
            "Epoch": epoch,
            "Train_Acc1": metrics["train_acc1"],
            "Train_Acc2": metrics["train_acc2"],
            "Val_Acc1": metrics["val_acc1"],
            "Val_Acc2": metrics["val_acc2"],
            "Val_F1_Cls1": metrics["val_f1_cls1"],
            "Val_F1_Cls2": metrics["val_f1_cls2"],
            "Best_Loss": metrics["best_loss"]
        })




def save_final_results_to_file(fold_results, best_fold_results, filename="final_results.txt"):
    with open(filename, "w") as f:
        # Ghi kết quả chung (Mean Loss và Std Dev)
        f.write("\nK-Fold Cross Validation Results:\n")
        f.write(f"Mean Loss: {np.mean(fold_results):.4f}\n")
        f.write(f"Std Dev: {np.std(fold_results):.4f}\n")
        
        # Ghi kết quả tốt nhất cho mỗi fold
        f.write("\nBest Results for Each Fold:\n")
        for result in best_fold_results:
            f.write(f"Fold {result['fold']}:\n")
            f.write(f"  Best Train Acc1: {result['best_train_acc1']:.2f}%\n")
            f.write(f"  Best Train Acc2: {result['best_train_acc2']:.2f}%\n")
            f.write(f"  Best Val Acc1: {result['best_val_acc1']:.2f}%\n")
            f.write(f"  Best Val Acc2: {result['best_val_acc2']:.2f}%\n")
            f.write(f"  Best Validation Loss: {result['best_loss']:.4f}\n")

