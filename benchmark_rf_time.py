import joblib
import time
import numpy as np
from PIL import Image
from pathlib import Path

# Load model
clf = joblib.load('Trained_Model/webcam_classifier.pkl')

# Fungsi ekstraksi fitur (salin dari app.py)
def extract_features(image, size=(128, 128)):
    image = image.convert('RGB').resize(size)
    arr = np.array(image)
    hist = []
    for i in range(3):
        h = np.histogram(arr[:,:,i], bins=32, range=(0,255))[0].astype(float)
        hist.extend(h)
    gray = np.mean(arr, axis=2).astype(np.uint8)
    edge_stats = [gray.mean(), gray.std(), gray.max()]
    return np.concatenate([hist, edge_stats])

# Ambil satu gambar contoh dari dataset baru
sample_img = next(Path('Dataset/baru/train/ambulance').glob('*'))
img = Image.open(sample_img)
feat = extract_features(img).reshape(1, -1)

# Warmup
_ = clf.predict(feat)
_ = clf.predict_proba(feat)

# Ukur 100 kali
times = []
for _ in range(100):
    start = time.time()
    _ = clf.predict(feat)
    _ = clf.predict_proba(feat)
    times.append(time.time() - start)

avg_sec = np.mean(times)
print(f'Rata-rata waktu deteksi (ekstraksi fitur + predict + predict_proba): {avg_sec:.4f} detik ({avg_sec*1000:.1f} ms)')