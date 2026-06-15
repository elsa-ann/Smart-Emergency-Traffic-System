# semua_akurasi.py
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json
import time
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path.cwd()

# ==================================================
# 1. Tentukan data evaluasi
# ==================================================
TEST_DIR = PROJECT_ROOT / "test_set"
VAL_DIR = PROJECT_ROOT / "Dataset" / "baru" / "val"

if TEST_DIR.exists() and any(TEST_DIR.iterdir()):
    DATA_DIR = TEST_DIR
    print("Menggunakan test_set untuk evaluasi.")
else:
    DATA_DIR = VAL_DIR
    print("Menggunakan Dataset/baru/val untuk evaluasi (catatan: data val pernah dilihat CNN).")

# ==================================================
# 2. Load CNN
# ==================================================
print("Loading CNN...")
cnn_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, 3)
cnn_model.load_state_dict(torch.load(PROJECT_ROOT / "Trained_Model" / "best_cnn_model.pth", map_location='cpu'))
cnn_model.eval()
with open(PROJECT_ROOT / "Trained_Model" / "cnn_classes.json", "r") as f:
    cnn_classes = json.load(f)["classes"]
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================================================
# 3. Load YOLO
# ==================================================
print("Loading YOLO...")
yolo_model = YOLO(PROJECT_ROOT / "Trained_Model" / "best.pt")

# ==================================================
# 4. Load Random Forest
# ==================================================
print("Loading Random Forest...")
rf_model = joblib.load(PROJECT_ROOT / "Trained_Model" / "decision_model.pkl")
rf_encoders = joblib.load(PROJECT_ROOT / "Trained_Model" / "label_encoders.pkl")

# ==================================================
# 5. Fungsi prediksi dengan waktu
# ==================================================
def predict_cnn(img_path):
    img = Image.open(img_path).convert('RGB')
    start = time.perf_counter()
    inp = cnn_transform(img).unsqueeze(0)
    with torch.no_grad():
        out = cnn_model(inp)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        idx = torch.argmax(probs).item()
        conf = probs[idx].item()
    elapsed = (time.perf_counter() - start) * 1000
    return cnn_classes[idx], conf, elapsed

def predict_yolo(img_path, conf_thresh=0.25):
    img_bgr = cv2.imread(str(img_path))
    start = time.perf_counter()
    results = yolo_model(img_bgr, conf=conf_thresh)[0]
    emergency = False
    max_conf = 0.0
    if results.boxes:
        for box in results.boxes:
            cls_name = results.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            if cls_name in ["emergency_vehicle", "ambulance", "firetruck", "fire truck"]:
                emergency = True
                max_conf = max(max_conf, conf)
    elapsed = (time.perf_counter() - start) * 1000
    return emergency, max_conf, elapsed

def predict_rf(vehicle_type, traffic_density, time_of_day, weather, distance, confidence, vehicle_count):
    start = time.perf_counter()
    # Mapping aman (ganti nilai yang tidak dikenal dengan default)
    try:
        vt = rf_encoders['vehicle_type'].transform([vehicle_type])[0]
    except:
        vt = rf_encoders['vehicle_type'].transform(['car'])[0]
    try:
        td = rf_encoders['traffic_density'].transform([traffic_density])[0]
    except:
        td = rf_encoders['traffic_density'].transform(['medium'])[0]
    try:
        tod = rf_encoders['time_of_day'].transform([time_of_day])[0]
    except:
        tod = rf_encoders['time_of_day'].transform(['afternoon'])[0]
    try:
        w = rf_encoders['weather'].transform([weather])[0]
    except:
        w = rf_encoders['weather'].transform(['clear'])[0]
    features = {
        'vehicle_type_enc': vt,
        'traffic_density_enc': td,
        'time_of_day_enc': tod,
        'weather_enc': w,
        'distance_m': distance,
        'confidence_pct': confidence,
        'vehicle_count': vehicle_count,
    }
    X = pd.DataFrame([features])
    pred_enc = rf_model.predict(X)[0]
    decision = rf_encoders['decision'].inverse_transform([pred_enc])[0]
    elapsed = (time.perf_counter() - start) * 1000
    return decision, elapsed

# ==================================================
# 6. Evaluasi gambar (CNN + YOLO + Ensemble)
# ==================================================
print("\nEvaluasi gambar...")
y_true = []
cnn_preds = []
yolo_preds = []
ensemble_preds = []
cnn_times = []
yolo_times = []

for class_folder in DATA_DIR.iterdir():
    if not class_folder.is_dir():
        continue
    true_emergency = (class_folder.name != "non_emergency")
    print(f"\nKelas: {class_folder.name} (emergency={true_emergency})")
    for img_path in class_folder.glob("*.*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        # CNN
        cnn_label, cnn_conf, t_cnn = predict_cnn(img_path)
        cnn_emergency = (cnn_label in ["ambulance", "pemadam_kebakaran"])
        # YOLO
        yolo_emergency, yolo_conf, t_yolo = predict_yolo(img_path)
        # Ensemble OR (threshold 85%)
        ensemble_emergency = (cnn_emergency and cnn_conf >= 0.85) or (yolo_emergency and yolo_conf >= 0.85)
        
        y_true.append(true_emergency)
        cnn_preds.append(cnn_emergency)
        yolo_preds.append(yolo_emergency)
        ensemble_preds.append(ensemble_emergency)
        cnn_times.append(t_cnn)
        yolo_times.append(t_yolo)
        
        print(f"  {img_path.name}: CNN={cnn_label}({cnn_conf:.2f}) YOLO={yolo_emergency}({yolo_conf:.2f}) Ens={ensemble_emergency} | t_cnn={t_cnn:.1f}ms t_yolo={t_yolo:.1f}ms")

# Akurasi
acc_cnn = accuracy_score(y_true, cnn_preds) * 100
acc_yolo = accuracy_score(y_true, yolo_preds) * 100
acc_ensemble = accuracy_score(y_true, ensemble_preds) * 100
avg_time_cnn = sum(cnn_times)/len(cnn_times) if cnn_times else 0
avg_time_yolo = sum(yolo_times)/len(yolo_times) if yolo_times else 0
avg_time_e2e = (avg_time_cnn + avg_time_yolo) / 2  # end-to-end gambar

# ==================================================
# 7. Evaluasi Random Forest (skenario sintetis)
# ==================================================
print("\nEvaluasi Random Forest...")
rf_scenarios = [
    ("ambulance", "high", "afternoon", "clear", 300, 95, 8, "OPEN_CORRIDOR"),
    ("car", "low", "morning", "clear", 500, 60, 3, "NO_ACTION"),
    ("firetruck", "medium", "evening", "rain", 400, 90, 15, "OPEN_CORRIDOR"),
    ("bus", "jam", "night", "fog", 200, 80, 25, "CAUTION"),
    ("ambulance", "clear", "afternoon", "clear", 100, 99, 2, "OPEN_CORRIDOR"),
    ("motorcycle", "high", "night", "fog", 600, 70, 30, "NO_ACTION"),
]
rf_correct = 0
rf_times = []
for vt, td, tod, w, dist, conf, vc, expected in rf_scenarios:
    dec, t_rf = predict_rf(vt, td, tod, w, dist, conf, vc)
    rf_times.append(t_rf)
    if dec == expected:
        rf_correct += 1
    print(f"  {vt:10} {td:6} -> {dec:14} (expected {expected:14}) time={t_rf:.2f}ms")
acc_rf = (rf_correct / len(rf_scenarios)) * 100
avg_time_rf = sum(rf_times)/len(rf_times)

# ==================================================
# 8. Rata-rata keempat akurasi
# ==================================================
rata2_akurasi = (acc_cnn + acc_yolo + acc_rf + acc_ensemble) / 4

# ==================================================
# 9. Cetak hasil
# ==================================================
print("\n" + "="*60)
print("📊 HASIL AKURASI DAN WAKTU INFERENSI")
print("="*60)
print(f"| Komponen               | Akurasi (%) | Rata-rata Waktu (ms) |")
print(f"|------------------------|-------------|----------------------|")
print(f"| CNN (ResNet18)         | {acc_cnn:9.2f} | {avg_time_cnn:18.2f} |")
print(f"| YOLO (Global)          | {acc_yolo:9.2f} | {avg_time_yolo:18.2f} |")
print(f"| Random Forest          | {acc_rf:9.2f} | {avg_time_rf:18.2f} |")
print(f"| Ensemble OR (CNN+YOLO) | {acc_ensemble:9.2f} | {avg_time_e2e:18.2f} |")
print(f"\n✨ Rata-rata keempat akurasi = {rata2_akurasi:.2f}%")
print(f"⏱️ Estimasi waktu deteksi end-to-end (gambar) : {avg_time_e2e:.0f} ms ({avg_time_e2e/1000:.3f} detik)")
print(f"⏱️ Rata-rata waktu inferensi Random Forest   : {avg_time_rf:.0f} ms ({avg_time_rf/1000:.3f} detik)")


# Simpan hasil ke JSON
hasil = {
    "cnn_accuracy": acc_cnn,
    "yolo_accuracy": acc_yolo,
    "rf_accuracy": acc_rf,
    "ensemble_accuracy": acc_ensemble,
    "avg_accuracy": rata2_akurasi,
    "inference_time_ms": avg_time_e2e,
    "rf_time_ms": avg_time_rf,
    "date": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open("hasil_akurasi.json", "w") as f:
    json.dump(hasil, f, indent=4)
print("\n📁 Hasil disimpan ke hasil_akurasi.json")