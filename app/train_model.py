"""
train_model.py — Drowsiness Detection Model Training (FIXED VERSION)
====================================================================
Fixes:
- Windows multiprocessing pickling error
- TransformSubset moved outside function
- num_workers = 0 for stability
"""

import os
import cv2
import copy
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import shutil
import time

# ─── Configuration ─────────────────────────────────────────────
DATA_DIR      = Path("data/drowsiness_dataset")
CLEANED_DIR   = Path("data/cleaned")
MODEL_PATH    = Path("models/drowsy_model.pth")
NUM_CLASSES   = 2
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
IMG_SIZE      = 224
BLUR_THRESHOLD = 80.0
EARLY_STOP_PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")

# ─── FIXED: TransformSubset moved OUTSIDE ─────────────────────
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return self.transform(img), label

# ─── Outlier Removal ──────────────────────────────────────────
def is_blurry(image_path: Path, threshold: float = BLUR_THRESHOLD) -> bool:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold

def clean_dataset(src_dir: Path, dst_dir: Path):
    if dst_dir.exists():
        print(f"[INFO] Cleaned dataset already exists at {dst_dir}, skipping.")
        return

    total, kept = 0, 0
    for class_dir in sorted(src_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        out_class = dst_dir / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            total += 1
            if is_blurry(img_path):
                continue
            shutil.copy(img_path, out_class / img_path.name)
            kept += 1

    print(f"[CLEAN] Total: {total} | Kept: {kept}")

# ─── Transforms ───────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ─── Data Loaders (FIXED) ─────────────────────────────────────
def build_dataloaders(data_dir: Path):
    full_dataset = datasets.ImageFolder(root=str(data_dir))
    print(f"[DATA] Classes: {full_dataset.classes}")

    indices = list(range(len(full_dataset)))
    labels  = full_dataset.targets

    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_labels = [labels[i] for i in train_idx]

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )

    sample_weights = [class_weights[l] for l in train_labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dataset = TransformSubset(full_dataset, train_idx, train_transforms)
    val_dataset   = TransformSubset(full_dataset, val_idx, val_transforms)

    # 🔥 FIX: num_workers = 0 (important for Windows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    return train_loader, val_loader

# ─── Model ────────────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model.to(DEVICE)

# ─── Training ─────────────────────────────────────────────────
def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total, correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total

        print(f"Epoch {epoch+1}: Train Accuracy {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            MODEL_PATH.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"[DONE] Best Accuracy: {best_acc:.4f}")

# ─── MAIN ─────────────────────────────────────────────────────
def main():
    print("Cleaning dataset...")
    clean_dataset(DATA_DIR, CLEANED_DIR)

    print("Loading data...")
    train_loader, val_loader = build_dataloaders(CLEANED_DIR)

    print("Building model...")
    model = build_model()

    print("Training...")
    train(model, train_loader, val_loader)

    print("Model saved!")

if __name__ == "__main__":
    main()