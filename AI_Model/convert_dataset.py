"""
Convert emergency-vehicles Kaggle dataset (CSV format) → YOLO format
Jalankan dari folder: Smart-Emergency-Traffic-System/
"""

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from PIL import Image

# ── CONFIG ──────────────────────────────────────────────────────
BASE_DIR    = Path("Dataset/emergency-vehicles")
TRAIN_IMG   = BASE_DIR / "train"
TRAIN_CSV   = BASE_DIR / "train.csv"
OUT_DIR     = BASE_DIR / "yolo"          # hasil konversi disimpan di sini

# Berapa gambar yang mau dipakai (biar ga kebanyakan)
MAX_SAMPLES = 200   # ← ganti angka ini kalau mau lebih/kurang

# ── LABEL MAP ───────────────────────────────────────────────────
# Cek kolom 'emergency_or_not' di CSV:
#   1 = emergency (ambulans / damkar)
#   0 = non-emergency (kendaraan biasa)
LABEL_MAP = {
    1: 0,   # emergency vehicle  → class 0
    0: 1,   # non-emergency      → class 1
}
CLASS_NAMES = ["emergency_vehicle", "non_emergency_vehicle"]

def main():
    print("=" * 55)
    print("  CONVERT KAGGLE CSV → YOLO FORMAT")
    print("=" * 55)

    # 1. Baca CSV
    if not TRAIN_CSV.exists():
        print(f"[ERROR] File tidak ditemukan: {TRAIN_CSV}")
        return
    df = pd.read_csv(TRAIN_CSV)
    print(f"\n[INFO] Total data di CSV : {len(df)} baris")
    print(f"[INFO] Kolom             : {list(df.columns)}")
    print(f"\n[INFO] Distribusi label :")
    print(df['emergency_or_not'].value_counts())

    # 2. Sampling — ambil MAX_SAMPLES data, balanced kalau bisa
    emergency     = df[df['emergency_or_not'] == 1]
    non_emergency = df[df['emergency_or_not'] == 0]

    n_emg = min(len(emergency),     MAX_SAMPLES // 2)
    n_non = min(len(non_emergency), MAX_SAMPLES - n_emg)

    sampled = pd.concat([
        emergency.sample(n=n_emg,     random_state=42),
        non_emergency.sample(n=n_non, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n[INFO] Setelah sampling  : {len(sampled)} gambar")
    print(f"       Emergency         : {n_emg}")
    print(f"       Non-emergency     : {n_non}")

    # 3. Split train / valid (80:20)
    split_idx  = int(len(sampled) * 0.8)
    train_df   = sampled.iloc[:split_idx]
    valid_df   = sampled.iloc[split_idx:]
    print(f"\n[INFO] Train : {len(train_df)} | Valid : {len(valid_df)}")

    # 4. Buat folder output YOLO
    for split in ["train", "valid"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # 5. Konversi tiap gambar
    def process_split(subset_df, split_name):
        ok, skip = 0, 0
        for _, row in subset_df.iterrows():
            img_name  = str(row['image_names'])
            label_val = int(row['emergency_or_not'])
            class_id  = LABEL_MAP[label_val]

            src_img = TRAIN_IMG / img_name
            if not src_img.exists():
                skip += 1
                continue

            # Salin gambar
            dst_img = OUT_DIR / split_name / "images" / img_name
            shutil.copy2(src_img, dst_img)

            # Buat label YOLO
            # Format: class_id cx cy w h (normalized 0-1)
            # Karena CSV tidak punya bounding box, kita pakai full-image box
            label_str = f"{class_id} 0.5 0.5 1.0 1.0\n"
            label_file = OUT_DIR / split_name / "labels" / (Path(img_name).stem + ".txt")
            label_file.write_text(label_str)
            ok += 1

        print(f"  [{split_name}] Berhasil: {ok} | Dilewati: {skip}")

    print("\n[INFO] Memproses gambar...")
    process_split(train_df, "train")
    process_split(valid_df, "valid")

    # 6. Buat data.yaml
    yaml_content = f"""path: {OUT_DIR.resolve()}
train: train/images
val:   valid/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (OUT_DIR / "data.yaml").write_text(yaml_content)
    print(f"\n[INFO] data.yaml dibuat di: {OUT_DIR / 'data.yaml'}")

    # 7. Ringkasan
    print("\n" + "=" * 55)
    print("  KONVERSI SELESAI!")
    print("=" * 55)
    print(f"  Output folder : Dataset/emergency-vehicles/yolo/")
    print(f"  Train images  : {len(list((OUT_DIR/'train'/'images').glob('*')))}")
    print(f"  Valid images  : {len(list((OUT_DIR/'valid'/'images').glob('*')))}")
    print(f"  data.yaml     : ✓")
    print("\n  Next step: jalankan train_yolo.py")

if __name__ == "__main__":
    main()