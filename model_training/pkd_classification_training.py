# -*- coding: utf-8 -*-
"""
================================================================================
PKD (Polycystic Kidney Disease) Classification System
CT Kidney Image Classification using Deep Learning
================================================================================

Project: Polycystic Kidney Disease Detection System
Guide: Prof. Kanchan Dabre
Team: Aniket Waghela, Bhavya Jappi, Kerul Kidecha

This script is structured for Google Colab execution with T4 GPU.
Run each section (cell) sequentially.

Dataset: CT-KIDNEY-DATASET (Normal, Cyst, Tumor, Stone)
Model: EfficientNet-B0 with Custom Classification Head
Training: 5-Fold Stratified Cross-Validation, 5 Epochs per fold
"""

# ============================================================================
# SECTION 1: ENVIRONMENT SETUP & GPU MEMORY OPTIMIZATION
# ============================================================================
# Run this cell first to set up the environment

# Install required packages (uncomment if running on Colab for the first time)
# !pip install albumentations timm seaborn scikit-learn

import os
import gc
import json
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# CPU/GPU compatible mixed precision imports
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = torch.cuda.is_available()
except ImportError:
    AMP_AVAILABLE = False
    GradScaler = None
    autocast = None

import torchvision.transforms as T
from torchvision import models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# GPU MEMORY OPTIMIZATION FOR T4
# ============================================================================

def setup_device():
    """Configure device (GPU or CPU) with appropriate optimizations"""
    if torch.cuda.is_available():
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Enable TF32 for faster computation on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
    else:
        device = torch.device('cpu')
        print(f"✓ Using CPU for training")
        print(f"✓ PyTorch Version: {torch.__version__}")
        print(f"  Note: Training on CPU will be slower than GPU")
    
    return device

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(42)
DEVICE = setup_device()
IS_CUDA = DEVICE.type == 'cuda'

# ============================================================================
# SECTION 2: CONFIGURATION & HYPERPARAMETERS
# ============================================================================

class Config:
    """Training configuration optimized for T4 GPU"""
    
    # Paths - Configured for local execution
    # BASE_DIR is the model_training folder, DATA_DIR is ../data
    BASE_DIR = Path(__file__).resolve().parent  # model_training folder
    DATA_DIR = BASE_DIR.parent / "data"  # Points to pkd/data folder
    CSV_PATH = DATA_DIR / "kidneyData_fixed.csv"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # Model settings
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 4
    CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
    
    # Training hyperparameters - Optimized for T4 GPU
    IMAGE_SIZE = 224
    BATCH_SIZE = 16  # Conservative for T4
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32
    NUM_EPOCHS = 3  # Reduced for CPU training
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Cross-validation
    NUM_FOLDS = 2  # Reduced for CPU training
    
    # Memory optimization - AMP only works on CUDA
    USE_AMP = IS_CUDA and AMP_AVAILABLE  # Mixed precision only on GPU
    PIN_MEMORY = IS_CUDA  # Pin memory only beneficial with CUDA
    NUM_WORKERS = 0 if os.name == 'nt' else 2  # 0 workers on Windows to avoid issues
    
    # Early stopping
    PATIENCE = 3
    
    # Visualization
    PLOT_DPI = 300
    FIGSIZE = (12, 8)
    
    @classmethod
    def create_output_dirs(cls):
        """Create output directories"""
        dirs = [
            cls.OUTPUT_DIR,
            cls.OUTPUT_DIR / "models",
            cls.OUTPUT_DIR / "plots",
            cls.OUTPUT_DIR / "metrics"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directories created at: {cls.OUTPUT_DIR}")

Config.create_output_dirs()

# ============================================================================
# SECTION 3: DATASET CLASS
# ============================================================================

class KidneyDataset(Dataset):
    """
    Custom Dataset for CT Kidney Classification
    
    Handles:
    - Loading images from CSV paths
    - Preprocessing (resize, normalize)
    - Data augmentation (via albumentations)
    """
    
    def __init__(self, df, base_dir, transform=None, is_training=True):
        """
        Args:
            df: DataFrame with 'path' and 'target' columns
            base_dir: Base directory for image paths
            transform: Albumentations transform pipeline
            is_training: Whether this is training data
        """
        self.df = df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Normalization values (ImageNet)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Construct image path
        img_path = self.base_dir / row['path']
        
        # Load and convert image
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            # Fallback: return a blank image if loading fails
            print(f"Warning: Could not load {img_path}: {e}")
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default transform if none provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Get label
        label = torch.tensor(row['target'], dtype=torch.long)
        
        return image, label

# ============================================================================
# SECTION 4: DATA AUGMENTATION PIPELINE
# ============================================================================

def get_train_transforms():
    """
    Training augmentation pipeline following the architecture diagram:
    - Spatial Augmentations
    - Color Augmentations  
    - Noise Augmentations
    """
    return A.Compose([
        # Preprocessing
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        
        # === SPATIAL AUGMENTATIONS ===
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        A.RandomRotate90(p=0.3),
        
        # === COLOR AUGMENTATIONS ===
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.5),
        
        # === NOISE AUGMENTATIONS ===
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        
        # === NORMALIZATION ===
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_valid_transforms():
    """Validation/Test transforms - only resize and normalize"""
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

# ============================================================================
# SECTION 5: MODEL ARCHITECTURE
# ============================================================================

class KidneyClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier for kidney CT images
    
    Features:
    - Pretrained EfficientNet-B0 encoder
    - Custom classification head with attention
    - Dropout for regularization
    """
    
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.3):
        super(KidneyClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize custom layers
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for custom layers"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output

def create_model():
    """Create and initialize the model"""
    model = KidneyClassifier(
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        dropout_rate=0.3
    )
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: {Config.MODEL_NAME}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model

# ============================================================================
# SECTION 6: TRAINING UTILITIES
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch):
    """Train for one epoch with mixed precision and gradient accumulation"""
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    optimizer.zero_grad(set_to_none=True)
    
    for step, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=IS_CUDA)
        labels = labels.to(DEVICE, non_blocking=IS_CUDA)
        
        # Mixed precision forward pass (only on CUDA)
        if Config.USE_AMP and autocast is not None:
            with autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
        
        # Gradient accumulation
        if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            if Config.USE_AMP and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        
        # Update meters
        loss_meter.update(loss.item() * Config.GRADIENT_ACCUMULATION_STEPS, images.size(0))
        acc_meter.update(acc.item(), images.size(0))
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def validate(model, dataloader, criterion):
    """Validate the model"""
    model.eval()
    
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc="Validating")
    
    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=IS_CUDA)
        labels = labels.to(DEVICE, non_blocking=IS_CUDA)
        
        if Config.USE_AMP and autocast is not None:
            with autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        loss_meter.update(loss.item(), images.size(0))
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'loss': loss_meter.avg,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics, all_preds, all_labels, all_probs

# ============================================================================
# SECTION 7: TRAINING WITH 5-FOLD CROSS-VALIDATION
# ============================================================================

def train_fold(fold, train_df, valid_df, all_metrics):
    """Train a single fold"""
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{Config.NUM_FOLDS}")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
    
    # Create datasets
    train_dataset = KidneyDataset(
        train_df,
        Config.DATA_DIR,
        transform=get_train_transforms(),
        is_training=True
    )
    valid_dataset = KidneyDataset(
        valid_df,
        Config.DATA_DIR,
        transform=get_valid_transforms(),
        is_training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Create model, optimizer, scheduler
    model = create_model()
    
    # Class weights for imbalanced data
    class_counts = train_df['target'].value_counts().sort_index()
    class_weights = 1.0 / class_counts.values
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=Config.NUM_EPOCHS,
        eta_min=Config.LEARNING_RATE / 100
    )
    # GradScaler only for AMP on CUDA
    if Config.USE_AMP and GradScaler is not None:
        scaler = GradScaler(enabled=True)
    else:
        scaler = None
    
    # Training history for this fold
    history = {
        'train_loss': [], 'train_acc': [],
        'valid_loss': [], 'valid_acc': [],
        'valid_f1': []
    }
    
    best_f1 = 0
    best_metrics = None
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch
        )
        
        # Validate
        valid_metrics, valid_preds, valid_labels, valid_probs = validate(
            model, valid_loader, criterion
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_metrics['loss'])
        history['valid_acc'].append(valid_metrics['accuracy'])
        history['valid_f1'].append(valid_metrics['f1'])
        
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Valid Loss: {valid_metrics['loss']:.4f}, Valid Acc: {valid_metrics['accuracy']:.4f}")
        print(f"  Valid F1: {valid_metrics['f1']:.4f}, Valid P: {valid_metrics['precision']:.4f}, Valid R: {valid_metrics['recall']:.4f}")
        
        # Save best model
        if valid_metrics['f1'] > best_f1:
            best_f1 = valid_metrics['f1']
            best_metrics = valid_metrics.copy()
            best_metrics['preds'] = valid_preds
            best_metrics['labels'] = valid_labels
            best_metrics['probs'] = valid_probs
            patience_counter = 0
            
            # Save model
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': best_metrics,
            }, Config.OUTPUT_DIR / "models" / f"best_model_fold_{fold+1}.pth")
            print(f"  ★ New best model saved! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        # Clear GPU cache after each epoch (only if CUDA)
        if IS_CUDA:
            torch.cuda.empty_cache()
        gc.collect()
    
    # Store fold results
    all_metrics['fold_histories'].append(history)
    all_metrics['fold_best_metrics'].append(best_metrics)
    
    # Clean up
    del model, optimizer, scheduler
    if scaler is not None:
        del scaler
    if IS_CUDA:
        torch.cuda.empty_cache()
    gc.collect()
    
    return best_metrics

def run_training():
    """Run complete 5-fold cross-validation training"""
    print("\n" + "="*60)
    print("STARTING 5-FOLD CROSS-VALIDATION TRAINING")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {Config.CSV_PATH}")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df['Class'].value_counts())
    
    # Create target mapping
    class_to_idx = {c: i for i, c in enumerate(sorted(df['Class'].unique()))}
    df['target'] = df['Class'].map(class_to_idx)
    
    print(f"\nClass mapping: {class_to_idx}")
    
    # Initialize metrics storage
    all_metrics = {
        'fold_histories': [],
        'fold_best_metrics': [],
        'class_names': Config.CLASS_NAMES,
        'config': {
            'model': Config.MODEL_NAME,
            'image_size': Config.IMAGE_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.NUM_EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
            'num_folds': Config.NUM_FOLDS
        }
    }
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df['target'])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
        
        fold_metrics = train_fold(fold, train_df, valid_df, all_metrics)
    
    # Calculate average metrics across folds
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in all_metrics['fold_best_metrics']]),
        'precision': np.mean([m['precision'] for m in all_metrics['fold_best_metrics']]),
        'recall': np.mean([m['recall'] for m in all_metrics['fold_best_metrics']]),
        'f1': np.mean([m['f1'] for m in all_metrics['fold_best_metrics']])
    }
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE - AVERAGE RESULTS ACROSS 5 FOLDS")
    print("="*60)
    print(f"  Average Accuracy:  {avg_metrics['accuracy']:.4f}")
    print(f"  Average Precision: {avg_metrics['precision']:.4f}")
    print(f"  Average Recall:    {avg_metrics['recall']:.4f}")
    print(f"  Average F1 Score:  {avg_metrics['f1']:.4f}")
    
    all_metrics['average_metrics'] = avg_metrics
    
    return all_metrics

# ============================================================================
# SECTION 8: PUBLICATION-QUALITY VISUALIZATIONS
# ============================================================================

def plot_training_curves(all_metrics):
    """Plot training and validation curves for all folds"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, Config.NUM_FOLDS))
    
    # Training Loss
    ax = axes[0, 0]
    for fold, history in enumerate(all_metrics['fold_histories']):
        ax.plot(history['train_loss'], color=colors[fold], 
                label=f'Fold {fold+1}', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss per Fold', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[0, 1]
    for fold, history in enumerate(all_metrics['fold_histories']):
        ax.plot(history['valid_loss'], color=colors[fold],
                label=f'Fold {fold+1}', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Validation Loss per Fold', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax = axes[1, 0]
    for fold, history in enumerate(all_metrics['fold_histories']):
        ax.plot(history['train_acc'], color=colors[fold],
                label=f'Fold {fold+1}', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training Accuracy per Fold', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax = axes[1, 1]
    for fold, history in enumerate(all_metrics['fold_histories']):
        ax.plot(history['valid_acc'], color=colors[fold],
                label=f'Fold {fold+1}', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Validation Accuracy per Fold', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "training_curves.png",
                dpi=Config.PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✓ Training curves saved")

def plot_confusion_matrices(all_metrics):
    """Plot confusion matrices for all folds"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for fold, metrics in enumerate(all_metrics['fold_best_metrics']):
        ax = axes[fold]
        cm = confusion_matrix(metrics['labels'], metrics['preds'])
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=Config.CLASS_NAMES,
                    yticklabels=Config.CLASS_NAMES,
                    ax=ax, cbar=False)
        ax.set_title(f'Fold {fold+1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Average confusion matrix
    ax = axes[5]
    all_labels = np.concatenate([m['labels'] for m in all_metrics['fold_best_metrics']])
    all_preds = np.concatenate([m['preds'] for m in all_metrics['fold_best_metrics']])
    cm_avg = confusion_matrix(all_labels, all_preds)
    cm_avg_normalized = cm_avg.astype('float') / cm_avg.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_avg_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES,
                ax=ax, cbar=True)
    ax.set_title('Average (All Folds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    plt.suptitle('Confusion Matrices - 5-Fold Cross-Validation',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "confusion_matrices.png",
                dpi=Config.PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✓ Confusion matrices saved")

def plot_roc_curves(all_metrics):
    """Plot ROC curves for multi-class classification"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Combine all predictions
    all_labels = np.concatenate([m['labels'] for m in all_metrics['fold_best_metrics']])
    all_probs = np.concatenate([m['probs'] for m in all_metrics['fold_best_metrics']])
    
    # Binarize labels
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    # Per-class ROC curves
    ax = axes[0]
    for i, (class_name, color) in enumerate(zip(Config.CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves per Class', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Macro and Micro average ROC
    ax = axes[1]
    
    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(all_labels_bin.ravel(), all_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color='deeppink', linewidth=2,
            label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
    
    # Macro-average
    fpr_macro = np.unique(np.concatenate([
        roc_curve(all_labels_bin[:, i], all_probs[:, i])[0]
        for i in range(Config.NUM_CLASSES)
    ]))
    tpr_macro = np.zeros_like(fpr_macro)
    for i in range(Config.NUM_CLASSES):
        fpr_i, tpr_i, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        tpr_macro += np.interp(fpr_macro, fpr_i, tpr_i)
    tpr_macro /= Config.NUM_CLASSES
    roc_auc_macro = auc(fpr_macro, tpr_macro)
    ax.plot(fpr_macro, tpr_macro, color='navy', linewidth=2,
            label=f'Macro-average (AUC = {roc_auc_macro:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Average ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "roc_curves.png",
                dpi=Config.PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✓ ROC curves saved")

def plot_class_performance(all_metrics):
    """Plot per-class performance metrics"""
    # Calculate per-class metrics from all folds
    all_labels = np.concatenate([m['labels'] for m in all_metrics['fold_best_metrics']])
    all_preds = np.concatenate([m['preds'] for m in all_metrics['fold_best_metrics']])
    
    # Get per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    ax = axes[0]
    x = np.arange(len(Config.CLASS_NAMES))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall_per_class, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(Config.CLASS_NAMES)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    # Radar chart
    ax = axes[1]
    ax.set_aspect('equal')
    
    # Create radar chart data
    metrics = np.array([precision_per_class, recall_per_class, f1_per_class])
    
    angles = np.linspace(0, 2 * np.pi, len(Config.CLASS_NAMES), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    labels = ['Precision', 'Recall', 'F1-Score']
    
    ax = plt.subplot(122, polar=True)
    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        values = np.concatenate([metric, [metric[0]]])
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(Config.CLASS_NAMES)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Per-Class Radar Chart', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "class_performance.png",
                dpi=Config.PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✓ Class performance plots saved")

def plot_fold_comparison(all_metrics):
    """Plot comparison across folds"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    folds = [f'Fold {i+1}' for i in range(Config.NUM_FOLDS)]
    
    # Extract metrics per fold
    accs = [m['accuracy'] for m in all_metrics['fold_best_metrics']]
    f1s = [m['f1'] for m in all_metrics['fold_best_metrics']]
    precisions = [m['precision'] for m in all_metrics['fold_best_metrics']]
    recalls = [m['recall'] for m in all_metrics['fold_best_metrics']]
    
    # Grouped bar chart
    ax = axes[0]
    x = np.arange(len(folds))
    width = 0.2
    
    ax.bar(x - 1.5*width, accs, width, label='Accuracy', color='#3498db')
    ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#2ecc71')
    ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#f39c12')
    ax.bar(x + 1.5*width, f1s, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics Comparison Across Folds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax = axes[1]
    data = [accs, precisions, recalls, f1s]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metric Distribution Across Folds', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    for i, d in enumerate(data):
        ax.hlines(np.mean(d), i + 0.75, i + 1.25, colors='red', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "fold_comparison.png",
                dpi=Config.PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✓ Fold comparison plot saved")

def plot_class_distribution(df):
    """Plot class distribution in the dataset"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    class_counts = df['Class'].value_counts()
    
    # Bar chart
    ax = axes[0]
    colors = ['#e74c3c', '#3498db', '#95a5a6', '#f39c12']
    bars = ax.bar(class_counts.index, class_counts.values, color=colors)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Pie chart
    ax = axes[1]
    ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
           colors=colors, explode=[0.02]*4, shadow=True)
    ax.set_title('Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "class_distribution.png",
                dpi=Config.PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✓ Class distribution plot saved")

def generate_all_plots(all_metrics):
    """Generate all publication-quality plots"""
    print("\n" + "="*60)
    print("GENERATING PUBLICATION-QUALITY PLOTS")
    print("="*60)
    
    # Load data for class distribution
    df = pd.read_csv(Config.CSV_PATH)
    
    # Generate all plots
    plot_class_distribution(df)
    plot_training_curves(all_metrics)
    plot_confusion_matrices(all_metrics)
    plot_roc_curves(all_metrics)
    plot_class_performance(all_metrics)
    plot_fold_comparison(all_metrics)
    
    print("\n✓ All plots saved to:", Config.OUTPUT_DIR / "plots")

# ============================================================================
# SECTION 9: SAVE RESULTS
# ============================================================================

def save_results(all_metrics):
    """Save all training results and metrics"""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Prepare metrics for JSON serialization
    metrics_to_save = {
        'config': all_metrics['config'],
        'class_names': all_metrics['class_names'],
        'average_metrics': all_metrics['average_metrics'],
        'fold_metrics': []
    }
    
    for fold, metrics in enumerate(all_metrics['fold_best_metrics']):
        fold_data = {
            'fold': fold + 1,
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'loss': float(metrics['loss'])
        }
        metrics_to_save['fold_metrics'].append(fold_data)
    
    # Save metrics JSON
    metrics_path = Config.OUTPUT_DIR / "metrics" / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Generate and save classification report
    all_labels = np.concatenate([m['labels'] for m in all_metrics['fold_best_metrics']])
    all_preds = np.concatenate([m['preds'] for m in all_metrics['fold_best_metrics']])
    
    report = classification_report(
        all_labels, all_preds,
        target_names=Config.CLASS_NAMES,
        digits=4
    )
    
    report_path = Config.OUTPUT_DIR / "metrics" / "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PKD Classification - 5-Fold Cross-Validation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {Config.MODEL_NAME}\n")
        f.write(f"Image Size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}\n")
        f.write(f"Epochs per Fold: {Config.NUM_EPOCHS}\n")
        f.write(f"Batch Size: {Config.BATCH_SIZE} (Effective: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS})\n")
        f.write(f"Learning Rate: {Config.LEARNING_RATE}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("CLASSIFICATION REPORT (All Folds Combined)\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
        f.write("AVERAGE METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:  {all_metrics['average_metrics']['accuracy']:.4f}\n")
        f.write(f"Precision: {all_metrics['average_metrics']['precision']:.4f}\n")
        f.write(f"Recall:    {all_metrics['average_metrics']['recall']:.4f}\n")
        f.write(f"F1-Score:  {all_metrics['average_metrics']['f1']:.4f}\n")
    
    print(f"✓ Classification report saved to: {report_path}")
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {Config.OUTPUT_DIR}")
    print("\nFiles saved:")
    print("  ├── models/")
    print("  │   └── best_model_fold_X.pth (5 models)")
    print("  ├── plots/")
    print("  │   ├── class_distribution.png")
    print("  │   ├── training_curves.png")
    print("  │   ├── confusion_matrices.png")
    print("  │   ├── roc_curves.png")
    print("  │   ├── class_performance.png")
    print("  │   └── fold_comparison.png")
    print("  └── metrics/")
    print("      ├── training_metrics.json")
    print("      └── classification_report.txt")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run training
    all_metrics = run_training()
    
    # Generate plots
    generate_all_plots(all_metrics)
    
    # Save results
    save_results(all_metrics)
