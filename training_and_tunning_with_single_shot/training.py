
import os
from ultralytics import YOLO
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


MODEL_NAME = "yolov8s.pt"  
DATA_CONFIG = "./configs/data.yaml"


EPOCHS = 50
NUM_WORKER = 8
BATCH_SIZE = 64
IMG_SIZE = 512  
PROJECT_NAME = "yolo_training"
EXPERIMENT_NAME = "yolov8s_512"




model = YOLO(MODEL_NAME)

results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,  
    batch=BATCH_SIZE,
    workers=NUM_WORKER,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    lr0=0.01,  
    device=device,
    patience=20,  
    save=True,  
    save_period=5,  
    cache=False,  
    
    degrees=0.2,  
    translate=0.2,  
    scale=0.2,  
    fliplr=0.2,  
    mosaic=0.0,  
    mixup=0.0,  
    copy_paste=0.1,  
    
    optimizer='auto',  
    cos_lr=True,  
    warmup_epochs=3,  
    warmup_momentum=0.8,  
    warmup_bias_lr=0.1,  
    
    box=5.0,  
    cls=0.5,  
    dfl=1.5,  
    overlap_mask=False,  
    mask_ratio=4,  
    single_cls=False,  
)


