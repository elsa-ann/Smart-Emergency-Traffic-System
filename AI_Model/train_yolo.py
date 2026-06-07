"""
Training YOLOv11 — Emergency Vehicle Detection
Jalankan dari folder: Smart-Emergency-Traffic-System/
"""

from ultralytics import YOLO
from pathlib import Path
import time

# ── CONFIG ──────────────────────────────────────────────────────
DATA_YAML   = Path("Dataset/emergency-vehicles/yolo/data.yaml")
MODEL_SIZE  = "yolo11n.pt"       # nano = paling ringan, cocok untuk laptop
EPOCHS      = 30                 # cukup untuk demo, bisa dinaikkan kalau mau lebih akurat
IMG_SIZE    = 416                # lebih kecil = lebih cepat
BATCH_SIZE  = 8                  # sesuai RAM 16GB
PROJECT_DIR = Path("Trained_Model")
RUN_NAME    = "emergency-detector"

def main():
    print("=" * 55)
    print("  TRAINING YOLO — EMERGENCY VEHICLE DETECTOR")
    print("=" * 55)
    print(f"\n  Model     : {MODEL_SIZE}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Img size  : {IMG_SIZE}px")
    print(f"  Batch     : {BATCH_SIZE}")
    print(f"  Data      : {DATA_YAML}")
    print(f"  Output    : {PROJECT_DIR}/{RUN_NAME}/")
    print("\n  Estimasi waktu: 10-20 menit (CPU)")
    print("=" * 55)

    if not DATA_YAML.exists():
        print(f"\n[ERROR] data.yaml tidak ditemukan: {DATA_YAML}")
        print("  Jalankan dulu: python AI_Model/convert_dataset.py")
        return

    # Load model
    print("\n[1/3] Loading model YOLO...")
    model = YOLO(MODEL_SIZE)

    # Training
    print("[2/3] Mulai training...\n")
    start = time.time()
    results = model.train(
        data    = str(DATA_YAML),
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        project = str(PROJECT_DIR),
        name    = RUN_NAME,
        exist_ok= True,
        verbose = True,
        # Augmentasi ringan biar model lebih robust
        flipud  = 0.1,
        fliplr  = 0.5,
        hsv_h   = 0.015,
        hsv_s   = 0.4,
        hsv_v   = 0.4,
    )
    elapsed = time.time() - start

    # Validasi
    print("\n[3/3] Validasi model...")
    metrics = model.val()

    # Simpan model terbaik
    best_model = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"

    print("\n" + "=" * 55)
    print("  TRAINING SELESAI!")
    print("=" * 55)
    print(f"  Waktu training : {elapsed/60:.1f} menit")
    print(f"  mAP50          : {metrics.box.map50:.3f}")
    print(f"  mAP50-95       : {metrics.box.map:.3f}")
    print(f"  Model tersimpan: {best_model}")
    print("\n  Next step: jalankan test_model.py")

if __name__ == "__main__":
    main()