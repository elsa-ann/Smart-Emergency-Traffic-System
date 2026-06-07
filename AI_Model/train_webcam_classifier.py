"""
Train a lightweight image classifier for webcam ambulance detection.

Folder structure expected:
  Dataset/webcam/train/ambulance
  Dataset/webcam/train/non_ambulance
  Dataset/webcam/val/ambulance
  Dataset/webcam/val/non_ambulance

Run from repository root:
  python AI_Model/train_webcam_classifier.py

The script saves:
  Trained_Model/webcam_classifier.pth
  Trained_Model/webcam_classes.json
"""

import json
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ── CONFIG ─────────────────────────────────────────────────────────
DATA_ROOT = Path("Dataset/webcam")
OUT_MODEL = Path("Trained_Model/webcam_classifier.pth")
OUT_CLASSES = Path("Trained_Model/webcam_classes.json")
BATCH_SIZE = 16
IMAGE_SIZE = 224
EPOCHS = 6
LR = 1e-4
NUM_WORKERS = 2


def build_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms


def build_datasets(train_tf, val_tf):
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Dataset folders not found. Create Dataset/webcam/train/<class> and Dataset/webcam/val/<class>."
        )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    return train_ds, val_ds


def build_model(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()

    loss = running_loss / len(loader.dataset)
    acc = running_corrects / len(loader.dataset)
    return loss, acc


def main():
    print("\n=== WEBCAM CLASSIFIER TRAINING ===\n")
    train_tf, val_tf = build_transforms()
    train_ds, val_ds = build_datasets(train_tf, val_tf)

    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model, OUT_MODEL)
            OUT_CLASSES.parent.mkdir(parents=True, exist_ok=True)
            json.dump({str(i): c for i, c in enumerate(train_ds.classes)}, OUT_CLASSES, indent=2)

    print("\nTraining complete")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Saved model: {OUT_MODEL}")
    print(f"Saved class map: {OUT_CLASSES}")

    print("\nNext: buka Web_Prototype/app.py dan pilih view Webcam AI")

if __name__ == '__main__':
    main()
