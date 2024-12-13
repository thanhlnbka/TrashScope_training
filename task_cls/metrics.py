import os
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from model_multihead import MultiHeadTimmModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from utils import *



def evaluate_model(model, folds, transform, device):
    model.eval()
    all_preds_cls1, all_preds_cls2 = [], []
    all_labels_cls1, all_labels_cls2 = [], []

    with torch.no_grad():
        for fold_idx, fold in enumerate(folds):
            print(f"\nEvaluating Fold {fold_idx + 1}/{len(folds)}:")
            for file_path, cls1, cls2 in tqdm(fold, desc=f"Processing Fold {fold_idx + 1}"):
                
                image = Image.open(file_path).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)

                
                head1_output, head2_output = model(image)
                prob1 = torch.softmax(head1_output, dim=1)
                prob2 = torch.softmax(head2_output, dim=1)

                
                pred_cls1 = torch.argmax(prob1).item()
                pred_cls2 = torch.argmax(prob2).item()

                
                all_preds_cls1.append(pred_cls1)
                all_preds_cls2.append(pred_cls2)
                all_labels_cls1.append(CLS1_LABELS.index(cls1))
                all_labels_cls2.append(CLS2_LABELS.index(cls2))

    
    acc_cls1 = accuracy_score(all_labels_cls1, all_preds_cls1)
    f1_cls1 = f1_score(all_labels_cls1, all_preds_cls1, average='weighted')
    cm_cls1 = confusion_matrix(all_labels_cls1, all_preds_cls1, labels=list(range(len(CLS1_LABELS))))

    
    acc_cls2 = accuracy_score(all_labels_cls2, all_preds_cls2)
    f1_cls2 = f1_score(all_labels_cls2, all_preds_cls2, average='weighted')
    cm_cls2 = confusion_matrix(all_labels_cls2, all_preds_cls2, labels=list(range(len(CLS2_LABELS))))

    
    print("\nMetrics for Cleanliness Status (Head 1):")
    print(f"Accuracy: {acc_cls1:.4f}")
    print(f"F1 Score: {f1_cls1:.4f}")
    plot_confusion_matrix(
        cm_cls1,
        CLS1_LABELS,
        "Confusion Matrix for Head 1 (Cleanliness)",
        save_path="confusion_matrix_cls1.png"
    )

    print("\nMetrics for Cover Status (Head 2):")
    print(f"Accuracy: {acc_cls2:.4f}")
    print(f"F1 Score: {f1_cls2:.4f}")
    plot_confusion_matrix(
        cm_cls2,
        CLS2_LABELS,
        "Confusion Matrix for Head 2 (Cover Status)",
        save_path="confusion_matrix_cls2.png"
    )

    return {
        "accuracy_cls1": acc_cls1,
        "f1_cls1": f1_cls1,
        "confusion_matrix_cls1": cm_cls1,
        "accuracy_cls2": acc_cls2,
        "f1_cls2": f1_cls2,
        "confusion_matrix_cls2": cm_cls2
    }




def main():
    
    MODEL_PATH = "pretrain_focallost.bak/best_model_fold_2.pth"  
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    DATA_PATH = "./test_images"  
    IMG_SIZE = 224  
    
    
    model = MultiHeadTimmModel(
        model_name='efficientnet_b0',  
        num_classes1=len(CLS1_LABELS),
        num_classes2=len(CLS2_LABELS)
    ).to(DEVICE)
    
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded successfully.")
    
    
    transform = create_transforms(img_size=IMG_SIZE)
    
    
    print("Grouping images by ID...")
    image_groups = group_images_by_id(DATA_PATH)

    folds = split_images_into_folds(image_groups, k=1)
    
    
    print("Evaluating model on the grouped dataset...")
    metrics = evaluate_model(model, folds, transform, DEVICE)
    print("\nFinal Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
