import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Folder dataset baru
DATA_ROOT = Path("Dataset/baru")  # sesuaikan dengan folder kamu
MODEL_DIR = Path("Trained_Model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def extract_features(image, size=(128, 128)):
    """Ekstraksi fitur dari gambar: histogram RGB + statistik edge sederhana"""
    image = image.convert('RGB').resize(size)
    arr = np.array(image)
    # Histogram 32 bins per channel
    hist = []
    for i in range(3):
        h = np.histogram(arr[:,:,i], bins=32, range=(0,255))[0].astype(float)
        hist.extend(h)
    # Simple edge stats menggunakan gradient
    gray = np.mean(arr, axis=2).astype(np.uint8)
    edge_stats = [gray.mean(), gray.std(), gray.max()]
    return np.concatenate([hist, edge_stats])

def load_data():
    X, y = [], []
    for split in ['train', 'val']:
        split_path = DATA_ROOT / split
        if not split_path.exists():
            continue
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            label = class_dir.name  # 'ambulance', 'pemadam_kebakaran', 'non_emergency'
            # Map ke emergency / non_emergency
            if label in ['ambulance', 'pemadam_kebakaran']:
                target = 'emergency'
            else:
                target = 'non_emergency'
            for img_path in class_dir.glob('*'):
                try:
                    img = Image.open(img_path)
                    feat = extract_features(img)
                    X.append(feat)
                    y.append(target)
                except Exception as e:
                    print(f"Error {img_path}: {e}")
    return np.array(X), np.array(y)

print("Loading dataset...")
X, y = load_data()
print(f"Total samples: {len(X)}")
print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

if len(X) == 0:
    raise RuntimeError("Tidak ada data! Pastikan folder Dataset/baru/train dan val ada isinya.")

# Split train/val (walau sudah ada val, kita tetap split untuk aman)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Training Random Forest
print("Training Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluasi
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation accuracy: {acc:.2%}")
print(classification_report(y_val, y_pred))

# Simpan model
joblib.dump(clf, MODEL_DIR / "webcam_classifier.pkl")
with open(MODEL_DIR / "webcam_classes.json", "w") as f:
    json.dump({"classes": ['emergency', 'non_emergency']}, f)

print(f"Model saved to {MODEL_DIR / 'webcam_classifier.pkl'}")