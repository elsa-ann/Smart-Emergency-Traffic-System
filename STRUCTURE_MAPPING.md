# Struktur Folder & Mapping File

## File Existing → Lokasi Baru

### Training Scripts (AI_Model/)
```
AI_Model/train_yolo.py              → tetap di AI_Model/ (atau copy ke scripts/)
AI_Model/train_decision_model.py    → tetap di AI_Model/ (atau copy ke scripts/)
AI_Model/train_webcam_classifier.py → tetap di AI_Model/ (atau copy ke scripts/)
AI_Model/convert_dataset.py         → tetap di AI_Model/ (atau copy ke scripts/)
```

### Web & UI (Web_Prototype/)
```
Web_Prototype/app.py                → src/web/dashboard.py (dengan symlink di root)
Web_Prototype/app_sim.py            → src/web/simulator.py (dengan symlink di root)
Web_Prototype/app.html              → src/web/static/app.html
```

### Dataset (Dataset/)
```
Dataset/traffic_decisions.csv       → tetap di Dataset/
Dataset/bandung-cctv/               → tetap di Dataset/
Dataset/emergency-vehicles/         → tetap di Dataset/
```

### Trained Models (Trained_Model/)
```
Trained_Model/best.pt               → tetap di Trained_Model/
Trained_Model/decision_model.pkl    → tetap di Trained_Model/
Trained_Model/label_encoders.pkl    → tetap di Trained_Model/
```

### Documentation (Documentation/)
```
Documentation/GUARDRAILS.md         → tetap (atau copy ke docs/)
Documentation/USER_GUIDE.md         → tetap (atau copy ke docs/)
Documentation/BIAS_ANALYSIS.md      → tetap (atau copy ke docs/)
```

### Baru dibuat
```
src/models/yolo_detector.py         ← BARU (wrapper YOLO)
src/models/decision_engine.py       ← BARU (wrapper decision model)
src/data/loader.py                  ← BARU (data loading helpers)
configs/model_config.yaml           ← BARU (model parameters)
configs/safety_constraints.yaml     ← BARU (ethical rules)
notebooks/01_webcam_classifier_training.ipynb ← BARU (training notebook)
```

## Struktur Akhir
```
Smart-Emergency-Traffic-System/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py       # YOLO wrapper
│   │   ├── decision_engine.py     # Decision model wrapper
│   │   └── utils.py               # Helpers (jika perlu)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Data loading
│   │   ├── preprocessor.py        # Data cleaning (jika perlu)
│   │   └── encoders.py            # Label encoding (jika perlu)
│   └── web/
│       ├── __init__.py
│       ├── dashboard.py           # app.py dipindah ke sini
│       ├── simulator.py           # app_sim.py dipindah ke sini
│       └── static/
│           └── app.html           # dari Web_Prototype/app.html
├── notebooks/
│   └── 01_webcam_classifier_training.ipynb  ← BARU
├── configs/
│   ├── model_config.yaml
│   └── safety_constraints.yaml
├── AI_Model/                       # Training scripts tetap di sini
├── Dataset/                        # Data tetap di sini
├── Trained_Model/                  # Models tetap di sini
├── Documentation/                  # Docs tetap di sini
├── Web_Prototype/                  # (legacy, bisa dihapus setelah migrasi)
├── requirements.txt
└── README.md
```
