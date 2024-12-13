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
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score


def calculate_class_weights(dataset):
    
    cls1_count = {cls: 0 for cls in dataset.cls1}
    cls2_count = {cls: 0 for cls in dataset.cls2}
    

    for _, cls1, cls2 in dataset.data:
        cls1_count[cls1] += 1
        cls2_count[cls2] += 1

    total_cls1 = len(dataset.data)
    total_cls2 = len(dataset.data)
    
    cls1_weights = {cls: total_cls1 / count for cls, count in cls1_count.items()}
    cls2_weights = {cls: total_cls2 / count for cls, count in cls2_count.items()}

    max_cls1_weight = max(cls1_weights.values())
    max_cls2_weight = max(cls2_weights.values())

    
    normalized_cls1_weights = {cls: weight / max_cls1_weight for cls, weight in cls1_weights.items()}
    normalized_cls2_weights = {cls: weight / max_cls2_weight for cls, weight in cls2_weights.items()}

    return normalized_cls1_weights, normalized_cls2_weights



def train_fold_clsweight(output_folder_training, model, train_loader, val_loader, 
                        criterion, optimizer, device, num_epochs=50, fold_idx=1, 
                        cls2_weight=0.5, patience=5):
    criterion_cls1, criterion_cls2 = criterion
    best_val_loss = float('inf')
    best_val_acc1, best_val_acc2 = 0, 0
    epochs_without_improvement = 0  

    best_preds_cls1, best_preds_cls2 = [], []
    best_labels_cls1, best_labels_cls2 = [], []

    fold_dir = f"{output_folder_training}/fold_results/fold_{fold_idx}"
    os.makedirs(fold_dir, exist_ok=True)  

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
            
            train_bar.set_postfix({"Loss": total_loss.item()})
        
        train_acc1 = correct_head1 / total_samples * 100
        train_acc2 = correct_head2 / total_samples * 100
        
        
        model.eval()
        val_loss = 0.0
        correct_head1, correct_head2 = 0, 0
        total_samples = 0
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        all_preds_cls1, all_preds_cls2 = [], []
        all_labels_cls1, all_labels_cls2 = [], []

        with torch.no_grad():
            for batch_images, batch_labels1, batch_labels2 in val_bar:
                batch_images = batch_images.to(device)
                batch_labels1 = batch_labels1.to(device)
                batch_labels2 = batch_labels2.to(device)
                
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
                
                val_bar.set_postfix({"Loss": total_loss.item()})
        
        val_acc1 = correct_head1 / total_samples * 100
        val_acc2 = correct_head2 / total_samples * 100
        
        
        val_f1_cls1 = f1_score(all_labels_cls1, all_preds_cls1, average="weighted")
        val_f1_cls2 = f1_score(all_labels_cls2, all_preds_cls2, average="weighted")
        
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        print(f"Train Acc1 (Cleanliness Status) = {train_acc1:.2f}%, Train Acc2 (Cover Status) = {train_acc2:.2f}%")
        print(f"Val Acc1 (Cleanliness Status) = {val_acc1:.2f}%, Val Acc2 (Cover Status) = {val_acc2:.2f}%")

        
        if val_acc1 > best_val_acc1:
            best_val_loss = avg_val_loss
            best_val_acc1 = val_acc1
            best_val_acc2 = val_acc2
            best_preds_cls1, best_preds_cls2 = all_preds_cls1, all_preds_cls2
            best_labels_cls1, best_labels_cls2 = all_labels_cls1, all_labels_cls2
            os.makedirs(f"{fold_dir}/best_model", exist_ok=True)
            model_save_path = f'{fold_dir}/best_model/best_model_fold_{fold_idx}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at {model_save_path} (Val Acc1 improved)")

            
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        
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


    OUTPUT_TRAINING = "./training_multihead_clsweight"
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

        cls1_weights, cls2_weights = calculate_class_weights(train_dataset)
        print("WEIGHT CLS1: ", cls1_weights)
        print("WEIGHT CLS2: ", cls2_weights)

        cls1_weight_tensor = torch.tensor(list(cls1_weights.values()), dtype=torch.float).to(DEVICE)
        cls2_weight_tensor = torch.tensor(list(cls2_weights.values()), dtype=torch.float).to(DEVICE)

        
        
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)  
        
        
        model = MultiHeadTimmModel(
            model_name=MODEL_NAME,  
            num_classes1=6,  
            num_classes2=3   
        ).to(DEVICE)  
        
        
        criterion_cls1 = nn.CrossEntropyLoss(weight=cls1_weight_tensor)
        criterion_cls2 = nn.CrossEntropyLoss(weight=cls2_weight_tensor)
        optimizer = optim.AdamW(
            model.parameters(),  
            lr=LEARNING_RATE,  
            weight_decay=1e-5  
        )
        
        
        fold_loss = train_fold_clsweight(
            OUTPUT_TRAINING,
            model,  
            train_loader,  
            val_loader,  
            (criterion_cls1, criterion_cls2),  
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