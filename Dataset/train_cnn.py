import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# --------------------------------------------------
# Konfigurasi
# --------------------------------------------------
DATA_DIR = "baru"           # folder yang berisi train/ dan val/
BATCH_SIZE = 16             # kurangi jadi 8 jika memori terbatas
EPOCHS = 30
LR = 0.001
NUM_CLASSES = 3             # ambulance, non_emergency, pemadam_kebakaran
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Transformasi gambar
# --------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------
# Fungsi training dan validasi
# --------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")

    # Cek apakah folder train/val ada
    train_path = os.path.join(DATA_DIR, "train")
    val_path = os.path.join(DATA_DIR, "val")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Folder train tidak ditemukan: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Folder val tidak ditemukan: {val_path}")

    # Load dataset
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    class_names = train_dataset.classes
    print("Kelas:", class_names)
    print(f"Jumlah train: {len(train_dataset)} gambar")
    print(f"Jumlah val: {len(val_dataset)} gambar")

    # Tampilkan jumlah per kelas
    for cls in class_names:
        idx = class_names.index(cls)
        train_count = sum(1 for _, label in train_dataset.samples if label == idx)
        val_count = sum(1 for _, label in val_dataset.samples if label == idx)
        print(f"  {cls}: train={train_count}, val={val_count}")

    # DataLoader dengan num_workers=0 untuk Windows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model ResNet18 dengan pretrained
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cnn_model.pth")
            print(f"*** Model saved with val_acc: {val_acc:.4f} ***")

        scheduler.step()

    # Simpan class names
    with open("cnn_classes.json", "w") as f:
        json.dump({"classes": class_names}, f)
    print("Class names saved to cnn_classes.json")

    # Plot kurva training
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Akurasi")
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig("training_curves.png")
    print("Training curves saved to training_curves.png")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()