"""
Generate synthetic dataset + Train Random Forest
untuk Decision Making AI (kapan buka/tutup lampu darurat)
Jalankan dari folder: Smart-Emergency-Traffic-System/
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────
OUT_MODEL   = Path("Trained_Model/decision_model.pkl")
OUT_ENCODER = Path("Trained_Model/label_encoders.pkl")
OUT_DATASET = Path("Dataset/traffic_decisions.csv")
N_SAMPLES   = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ── 1. GENERATE DATASET ─────────────────────────────────────────
def generate_dataset(n):
    print(f"[1/4] Generating {n} samples dataset...")

    vehicle_types  = ["ambulance", "firetruck", "car", "bus", "truck", "motorcycle"]
    traffic_levels = ["clear", "low", "medium", "high", "jam"]
    time_of_day    = ["morning", "afternoon", "evening", "night"]
    weather        = ["clear", "rain", "fog"]

    rows = []
    for _ in range(n):
        v_type    = np.random.choice(vehicle_types,  p=[0.15, 0.10, 0.40, 0.15, 0.10, 0.10])
        traffic   = np.random.choice(traffic_levels, p=[0.10, 0.20, 0.30, 0.25, 0.15])
        tod       = np.random.choice(time_of_day)
        w         = np.random.choice(weather,        p=[0.70, 0.20, 0.10])
        distance  = np.random.randint(50, 1001)       # meter
        confidence= round(np.random.uniform(60, 99), 1)
        v_count   = np.random.randint(0, 25)          # jumlah kendaraan di persimpangan

        is_emergency = v_type in ["ambulance", "firetruck"]
        conf_ok      = confidence >= 85
        dist_ok      = distance <= 500

        # ── DECISION RULES ──────────────────────────────────────
        # OPEN_CORRIDOR = buka jalur darurat
        # NO_ACTION     = tidak perlu tindakan
        # CAUTION       = perlu konfirmasi manual

        if is_emergency and conf_ok and dist_ok:
            if traffic in ["high", "jam"]:
                decision = "OPEN_CORRIDOR"
            elif traffic in ["medium"]:
                decision = "OPEN_CORRIDOR"
            else:
                decision = "OPEN_CORRIDOR"
        elif is_emergency and conf_ok and not dist_ok:
            decision = "CAUTION"       # masih jauh, pantau dulu
        elif is_emergency and not conf_ok:
            decision = "CAUTION"       # confidence rendah
        else:
            decision = "NO_ACTION"

        # Tambah sedikit noise biar model ga terlalu perfect
        if np.random.random() < 0.03:
            decision = np.random.choice(["OPEN_CORRIDOR", "NO_ACTION", "CAUTION"])

        rows.append({
            "vehicle_type"   : v_type,
            "traffic_density": traffic,
            "time_of_day"    : tod,
            "weather"        : w,
            "distance_m"     : distance,
            "confidence_pct" : confidence,
            "vehicle_count"  : v_count,
            "decision"       : decision,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATASET, index=False)
    print(f"  Dataset tersimpan: {OUT_DATASET}")
    print(f"  Distribusi keputusan:")
    print(df["decision"].value_counts().to_string(index=True))
    return df

# ── 2. PREPROCESSING ────────────────────────────────────────────
def preprocess(df):
    print("\n[2/4] Preprocessing data...")
    encoders = {}
    cat_cols = ["vehicle_type", "traffic_density", "time_of_day", "weather"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    le_target = LabelEncoder()
    df["decision_enc"] = le_target.fit_transform(df["decision"])
    encoders["decision"] = le_target

    feature_cols = [c + "_enc" for c in cat_cols] + ["distance_m", "confidence_pct", "vehicle_count"]
    X = df[feature_cols]
    y = df["decision_enc"]
    print(f"  Features : {list(X.columns)}")
    print(f"  Classes  : {list(le_target.classes_)}")
    return X, y, encoders

# ── 3. TRAINING ─────────────────────────────────────────────────
def train(X, y):
    print("\n[3/4] Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy : {acc*100:.2f}%")
    return model, X_test, y_test, y_pred

# ── 4. EVALUATION ───────────────────────────────────────────────
def evaluate(model, X_test, y_test, y_pred, encoders):
    print("\n[4/4] Evaluasi model...")
    le = encoders["decision"]
    labels = list(le.classes_)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    fi = pd.Series(model.feature_importances_, index=X_test.columns)
    fi = fi.sort_values(ascending=False)
    print("\n  Feature Importance:")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<22} {bar} {imp:.3f}")

    return accuracy_score(y_test, y_pred)

# ── MAIN ────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  DECISION MAKING AI — RANDOM FOREST TRAINING")
    print("=" * 55)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    df                        = generate_dataset(N_SAMPLES)
    X, y, encoders            = preprocess(df)
    model, X_test, y_test, yp = train(X, y)
    acc                       = evaluate(model, X_test, y_test, yp, encoders)

    # Simpan model & encoder
    joblib.dump(model,    OUT_MODEL)
    joblib.dump(encoders, OUT_ENCODER)

    print("\n" + "=" * 55)
    print("  TRAINING SELESAI!")
    print("=" * 55)
    print(f"  Accuracy         : {acc*100:.2f}%")
    print(f"  Model tersimpan  : {OUT_MODEL}")
    print(f"  Encoder tersimpan: {OUT_ENCODER}")
    print(f"  Dataset tersimpan: {OUT_DATASET}")
    print("\n  Next step: jalankan app.py (Streamlit)")

if __name__ == "__main__":
    main()