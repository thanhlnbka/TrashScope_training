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
from tqdm import tqdm
import numpy as np
from model_multihead import MultiHeadTimmModel
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import csv 
from utils import *


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma  
        self.reduction = reduction  
    
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        
        
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        p_t = (inputs * targets_one_hot).sum(dim=1)  
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



def train_fold_focalloss(
    output_folder_training, model, train_loader, val_loader, optimizer, device, num_epochs=50, fold_idx=1, cls2_weight=0.5, patience=5
):
    criterion_cls1 = FocalLoss(alpha=0.25, gamma=2.0)  
    criterion_cls2 = FocalLoss(alpha=0.25, gamma=2.0)  

    best_val_loss = float("inf")
    best_val_acc1, best_val_acc2 = 0, 0
    best_train_acc1, best_train_acc2 = 0, 0  
    best_preds_cls1, best_preds_cls2 = [], []
    best_labels_cls1, best_labels_cls2 = [], []
    
    fold_dir = f"{output_folder_training}/fold_results/fold_{fold_idx}"
    os.makedirs(fold_dir, exist_ok=True)  
    
    
    epochs_since_improvement = 0  

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} (Fold {fold_idx})")

        
        model.train()
        train_loss, correct_head1, correct_head2, total_samples = 0.0, 0, 0, 0

        for batch_images, batch_labels1, batch_labels2 in tqdm(train_loader, desc="Training", leave=False):
            batch_images, batch_labels1, batch_labels2 = (
                batch_images.to(device),
                batch_labels1.to(device),
                batch_labels2.to(device),
            )
            optimizer.zero_grad()

            
            head1_outputs, head2_outputs = model(batch_images)
            loss1 = criterion_cls1(head1_outputs, batch_labels1)
            loss2 = criterion_cls2(head2_outputs, batch_labels2)
            total_loss = loss1 + cls2_weight * loss2
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            _, pred_head1 = torch.max(head1_outputs, 1)
            _, pred_head2 = torch.max(head2_outputs, 1)
            correct_head1 += (pred_head1 == batch_labels1).sum().item()
            correct_head2 += (pred_head2 == batch_labels2).sum().item()
            total_samples += batch_labels1.size(0)

        train_acc1 = correct_head1 / total_samples * 100
        train_acc2 = correct_head2 / total_samples * 100

        
        best_train_acc1 = max(best_train_acc1, train_acc1)
        best_train_acc2 = max(best_train_acc2, train_acc2)

        
        model.eval()
        val_loss, correct_head1, correct_head2, total_samples = 0.0, 0, 0, 0
        all_preds_cls1, all_preds_cls2 = [], []
        all_labels_cls1, all_labels_cls2 = [], []

        with torch.no_grad():
            for batch_images, batch_labels1, batch_labels2 in tqdm(val_loader, desc="Validation", leave=False):
                batch_images, batch_labels1, batch_labels2 = (
                    batch_images.to(device),
                    batch_labels1.to(device),
                    batch_labels2.to(device),
                )
                head1_outputs, head2_outputs = model(batch_images)
                loss1 = criterion_cls1(head1_outputs, batch_labels1)
                loss2 = criterion_cls2(head2_outputs, batch_labels2)
                total_loss = loss1 + cls2_weight * loss2
                val_loss += total_loss.item()

                _, pred_head1 = torch.max(head1_outputs, 1)
                _, pred_head2 = torch.max(head2_outputs, 1)
                correct_head1 += (pred_head1 == batch_labels1).sum().item()
                correct_head2 += (pred_head2 == batch_labels2).sum().item()
                total_samples += batch_labels1.size(0)

                all_preds_cls1.extend(pred_head1.cpu().tolist())
                all_preds_cls2.extend(pred_head2.cpu().tolist())
                all_labels_cls1.extend(batch_labels1.cpu().tolist())
                all_labels_cls2.extend(batch_labels2.cpu().tolist())

        val_acc1 = correct_head1 / total_samples * 100
        val_acc2 = correct_head2 / total_samples * 100
        val_f1_cls1 = f1_score(all_labels_cls1, all_preds_cls1, average="weighted")
        val_f1_cls2 = f1_score(all_labels_cls2, all_preds_cls2, average="weighted")

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc1, best_val_acc2 = val_acc1, val_acc2
            best_preds_cls1, best_preds_cls2 = all_preds_cls1, all_preds_cls2
            best_labels_cls1, best_labels_cls2 = all_labels_cls1, all_labels_cls2
            epochs_since_improvement = 0  

            model_save_path = os.path.join(fold_dir, f"best_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at {model_save_path}")
        else:
            epochs_since_improvement += 1

        
        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        print(f"Epoch {epoch + 1}: Val Acc1 = {val_acc1:.2f}%, Val F1_cls1 = {val_f1_cls1:.4f}")
        print(f"Val Acc2 = {val_acc2:.2f}%, Val F1_cls2 = {val_f1_cls2:.4f}")

        
        metrics = {
            "train_acc1": train_acc1,
            "train_acc2": train_acc2,
            "val_acc1": val_acc1,
            "val_acc2": val_acc2,
            "val_f1_cls1": val_f1_cls1,
            "val_f1_cls2": val_f1_cls2,
            "best_loss": best_val_loss,
        }
        save_metrics_to_csv(fold_dir, metrics, fold_idx, epoch + 1)

    
    cm_cls1 = confusion_matrix(best_labels_cls1, best_preds_cls1)
    cm_cls2 = confusion_matrix(best_labels_cls2, best_preds_cls2)

    
    cm1_path = os.path.join(fold_dir, "confusion_matrix_cls1_final.png")
    cm2_path = os.path.join(fold_dir, "confusion_matrix_cls2_final.png")
    plot_confusion_matrix(cm_cls1, CLS1_LABELS, "Cleanliness Status", cm1_path)
    plot_confusion_matrix(cm_cls2, CLS2_LABELS, "Cover Status", cm2_path)

    
    print(f"Best Validation Acc1: {best_val_acc1:.2f}%, Best Validation Acc2: {best_val_acc2:.2f}%")

    
    return {
        "best_train_acc1": best_train_acc1,
        "best_train_acc2": best_train_acc2,
        "best_val_acc1": best_val_acc1,
        "best_val_acc2": best_val_acc2,
        "best_loss": best_val_loss
    }



def main():
    
    DATA_PATH = './output_images'
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    IMG_SIZE = 224
    K_FOLDS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_NAME = 'efficientnet_b0'

    OUTPUT_TRAINING = "./training_multihead_folcalloss"
    os.makedirs(OUTPUT_TRAINING, exist_ok=True)
    

    
    image_groups = group_images_by_id(DATA_PATH)
    folds, context_fold = split_images_into_folds(image_groups, k=K_FOLDS)

    save_context_to_file(context_fold, f"{OUTPUT_TRAINING}/description_folds.txt")
    

    
    transform = create_transforms(img_size=IMG_SIZE)

    
    fold_results = []
    best_fold_results = []
    
    for fold in range(K_FOLDS):
        print(f"\nTraining Fold {fold + 1}/{K_FOLDS}")  
        
        
        val_data = folds[fold]  
        train_data = [item for f in folds[:fold] + folds[fold+1:] for item in f]  

        
        print(f"Validation Fold: {fold + 1} ({len(val_data)} samples)")  
        print(f"Training Folds: {[i + 1 for i in range(K_FOLDS) if i != fold]} ({len(train_data)} samples)")  
        
        
        train_dataset = CustomDataset(train_data, transform=transform)  
        val_dataset = CustomDataset(val_data, transform=transform)  
        
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)  
        
        
        model = MultiHeadTimmModel(
            model_name=MODEL_NAME,  
            num_classes1=6,  
            num_classes2=3   
        ).to(DEVICE)  
        
        optimizer = optim.AdamW(
            model.parameters(),  
            lr=LEARNING_RATE,  
            weight_decay=1e-5  
        )
        
        
        fold_loss = train_fold_focalloss(
            OUTPUT_TRAINING,
            model,  
            train_loader,  
            val_loader,  
            optimizer,  
            DEVICE,  
            num_epochs=NUM_EPOCHS,  
            fold_idx=fold + 1,  
            cls2_weight=0.1
        )
        
        fold_results.append(fold_loss["best_loss"])  
        best_fold_results.append({
            "fold": fold + 1,
            "best_train_acc1": fold_loss["best_train_acc1"],
            "best_train_acc2": fold_loss["best_train_acc2"],
            "best_val_acc1": fold_loss["best_val_acc1"],
            "best_val_acc2": fold_loss["best_val_acc2"],
            "best_loss": fold_loss["best_loss"]
        })
        
        
        print(f"Fold {fold + 1} - Best Results:")
        print(f"  Best Train Acc1 (Cleanliness): {fold_loss['best_train_acc1']:.2f}%")
        print(f"  Best Train Acc2 (Cover): {fold_loss['best_train_acc2']:.2f}%")
        print(f"  Best Val Acc1 (Cleanliness): {fold_loss['best_val_acc1']:.2f}%")
        print(f"  Best Val Acc2 (Cover): {fold_loss['best_val_acc2']:.2f}%")
        print(f"  Best Val Loss: {fold_loss['best_loss']:.4f}")
    
    save_final_results_to_file(fold_results, best_fold_results, f"{OUTPUT_TRAINING}/final_results.txt")



if __name__ == '__main__':
    main()