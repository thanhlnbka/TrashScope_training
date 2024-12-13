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
from utils import *



def train_fold(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, fold_idx=1, cls2_weight=0.5):
    best_val_loss = float('inf')
    best_val_acc1, best_val_acc2 = 0, 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs} (Fold {fold_idx})')
        model.train()
        train_loss = 0.0
        correct_head1, correct_head2 = 0, 0
        total_samples = 0
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_images, batch_labels1, batch_labels2 in train_bar:
            batch_images = batch_images.to(device)
            batch_labels1 = batch_labels1.to(device)
            batch_labels2 = batch_labels2.to(device)
            
            optimizer.zero_grad()
            
            
            head1_outputs, head2_outputs = model(batch_images)
            
            
            loss1 = criterion(head1_outputs, batch_labels1)
            loss2 = criterion(head2_outputs, batch_labels2)
            total_loss = loss1 + cls2_weight * loss2  
            
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            
            _, pred_head1 = torch.max(head1_outputs, 1)
            _, pred_head2 = torch.max(head2_outputs, 1)
            correct_head1 += (pred_head1 == batch_labels1).sum().item()
            correct_head2 += (pred_head2 == batch_labels2).sum().item()
            total_samples += batch_labels1.size(0)
            
            train_bar.set_postfix({"Loss": total_loss.item()})
        
        train_acc1 = correct_head1 / total_samples * 100
        train_acc2 = correct_head2 / total_samples * 100
        
        
        model.eval()
        val_loss = 0.0
        correct_head1, correct_head2 = 0, 0
        total_samples = 0
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_images, batch_labels1, batch_labels2 in val_bar:
                batch_images = batch_images.to(device)
                batch_labels1 = batch_labels1.to(device)
                batch_labels2 = batch_labels2.to(device)
                
                head1_outputs, head2_outputs = model(batch_images)
                
                loss1 = criterion(head1_outputs, batch_labels1)
                loss2 = criterion(head2_outputs, batch_labels2)
                total_loss = loss1 + cls2_weight * loss2  
                
                val_loss += total_loss.item()
                
                
                _, pred_head1 = torch.max(head1_outputs, 1)
                _, pred_head2 = torch.max(head2_outputs, 1)
                correct_head1 += (pred_head1 == batch_labels1).sum().item()
                correct_head2 += (pred_head2 == batch_labels2).sum().item()
                total_samples += batch_labels1.size(0)
                
                val_bar.set_postfix({"Loss": total_loss.item()})
        
        val_acc1 = correct_head1 / total_samples * 100
        val_acc2 = correct_head2 / total_samples * 100
        
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        print(f"Train Acc1 (Cleanliness Status) = {train_acc1:.2f}%, Train Acc2 (Cover Status) = {train_acc2:.2f}%")
        print(f"Val Acc1 (Cleanliness Status) = {val_acc1:.2f}%, Val Acc2 (Cover Status) = {val_acc2:.2f}%")
        
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc1 = val_acc1
            best_val_acc2 = val_acc2
            os.makedirs("pretrain_normal", exist_ok=True)
            model_save_path = f'pretrain_normal/best_model_fold_{fold_idx}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at {model_save_path}")
    
    print(f"Best Val Acc1 (Cleanliness Status) = {best_val_acc1:.2f}%, Best Val Acc2 (Cover Status) = {best_val_acc2:.2f}%")
    return {
        "best_train_acc1": train_acc1,      
        "best_train_acc2": train_acc2,      
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

    image_groups = group_images_by_id(DATA_PATH)
    folds = split_images_into_folds(image_groups, k=K_FOLDS)

    
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
        
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)  
        
        
        model = MultiHeadTimmModel(
            model_name=MODEL_NAME,  
            num_classes1=6,  
            num_classes2=3   
        ).to(DEVICE)  
        
        
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.AdamW(
            model.parameters(),  
            lr=LEARNING_RATE,  
            weight_decay=1e-5  
        )
        
        
        fold_loss = train_fold(
            model,  
            train_loader,  
            val_loader,  
            criterion,  
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
    
    
    print("\nK-Fold Cross Validation Results:")
    print(f"Mean Loss: {np.mean(fold_results):.4f}")
    print(f"Std Dev: {np.std(fold_results):.4f}")
    
    
    print("\nBest Results for Each Fold:")
    for result in best_fold_results:
        print(f"Fold {result['fold']}:")
        print(f"  Best Train Acc1: {result['best_train_acc1']:.2f}%")
        print(f"  Best Train Acc2: {result['best_train_acc2']:.2f}%")
        print(f"  Best Val Acc1: {result['best_val_acc1']:.2f}%")
        print(f"  Best Val Acc2: {result['best_val_acc2']:.2f}%")
        print(f"  Best Validation Loss: {result['best_loss']:.4f}")



if __name__ == '__main__':
    main()