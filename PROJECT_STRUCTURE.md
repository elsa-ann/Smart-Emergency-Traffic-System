# SECS Project Structure — Complete Overview

Struktur final yang sudah dibuat beserta mapping file existing.

## ✅ Structure Created

```
Smart-Emergency-Traffic-System/
│
├── 📁 src/                           [NEW - Modular code structure]
│   ├── models/
│   │   ├── __init__.py               [NEW]
│   │   ├── yolo_detector.py          [NEW - YOLO wrapper]
│   │   ├── decision_engine.py        [NEW - Decision model wrapper]
│   │   └── utils.py                  [placeholder - helpers]
│   │
│   ├── data/
│   │   ├── __init__.py               [NEW]
│   │   ├── loader.py                 [NEW - Data loading helpers]
│   │   ├── preprocessor.py           [placeholder]
│   │   └── encoders.py               [placeholder]
│   │
│   └── web/
│       ├── __init__.py               [NEW]
│       ├── dashboard.py              [TODO - copy from Web_Prototype/app.py]
│       ├── simulator.py              [TODO - copy from Web_Prototype/app_sim.py]
│       └── static/
│           └── app.html              [TODO - copy from Web_Prototype/app.html]
│
├── 📁 configs/                       [NEW - Configuration files]
│   ├── model_config.yaml             [NEW - Model parameters]
│   └── safety_constraints.yaml       [NEW - Ethical guardrails]
│
├── 📁 notebooks/                     [NEW - Jupyter notebooks for training]
│   └── 01_webcam_classifier_training.ipynb  [NEW - Webcam classifier training]
│
├── 📁 docs/                          [NEW - Documentation]
│   ├── API.md                        [NEW - API reference]
│   ├── SIMULATOR_GUIDE.md            [NEW - app_sim.py usage]
│   └── WEBCAM_AI_GUIDE.md            [NEW - Webcam AI implementation]
│
├── 📁 AI_Model/                      [EXISTING]
│   ├── train_yolo.py
│   ├── train_decision_model.py
│   ├── train_webcam_classifier.py
│   ├── convert_dataset.py
│   └── README.md
│
├── 📁 Dataset/                       [EXISTING]
│   ├── traffic_decisions.csv
│   ├── bandung-cctv/
│   └── emergency-vehicles/
│
├── 📁 Trained_Model/                 [EXISTING + NEW]
│   ├── best.pt                       [YOLO]
│   ├── decision_model.pkl            [Decision model]
│   ├── label_encoders.pkl            [Encoders]
│   └── webcam_classifier.pkl         [NEW - generated dari notebook]
│
├── 📁 Documentation/                 [EXISTING]
│   ├── GUARDRAILS.md
│   ├── BIAS_ANALYSIS.md
│   ├── USER_GUIDE.md
│   └── POSTER_A1.md
│
├── 📁 Web_Prototype/                 [EXISTING - legacy after migration]
│   ├── app.html
│   ├── app.py
│   └── app_sim.py
│
├── 📁 runs/                          [EXISTING]
├── 📁 scripts/                       [EXISTING]
├── STRUCTURE_MAPPING.md              [NEW - file mapping reference]
├── CODE_REVIEW.md                    [EXISTING]
├── CONTRIBUTING.md                   [EXISTING]
├── requirements.txt                  [EXISTING + updated]
├── README.md                         [EXISTING + updated]
└── .git/                             [EXISTING]
```

## 📋 File Mapping & Migration Plan

### Option A: Keep Both (Current State)
`Web_Prototype/` tetap ada, sementara `src/web/` juga ada.
- ✅ No breaking changes
- ⚠️ Duplicate code
- 📍 Use `Web_Prototype/` untuk now, `src/web/` untuk future

### Option B: Migrate (Recommended)
1. Copy `Web_Prototype/app.py` → `src/web/dashboard.py`
2. Copy `Web_Prototype/app_sim.py` → `src/web/simulator.py`
3. Copy `Web_Prototype/app.html` → `src/web/static/app.html`
4. Update imports dalam dashboard/simulator
5. Update `streamlit run` command ke new location
6. Delete `Web_Prototype/` setelah verified working

```bash
# Setelah migration:
streamlit run src/web/dashboard.py
# atau dari root:
streamlit run Web_Prototype/app.py  (akan broken, butuh update)
```

## 🚀 Next Steps

### 1. Train Webcam Classifier (5 min)
```bash
jupyter notebook notebooks/01_webcam_classifier_training.ipynb
# Run semua cells
# Output: Trained_Model/webcam_classifier.pkl
```

### 2. Update `app.py` Webcam AI View (10 min)
Copy code dari `docs/WEBCAM_AI_GUIDE.md` ke `Web_Prototype/app.py`:
```python
elif view == "Webcam AI":
    # ... replace placeholder with full implementation
```

### 3. (Optional) Migrate to src/ Structure (15 min)
```bash
# Copy files
cp Web_Prototype/app.py src/web/dashboard.py
cp Web_Prototype/app_sim.py src/web/simulator.py
cp Web_Prototype/app.html src/web/static/app.html

# Test
streamlit run src/web/dashboard.py

# Update imports if needed
```

### 4. Test All Views
```bash
streamlit run Web_Prototype/app.py
# Test:
# - Prototype (HTML) ✓
# - Simulator ✓
# - Streamlit Dashboard ✓
# - Webcam AI ← should work now
```

## 📚 Documentation Map

| Nama | File | Untuk |
|------|------|-------|
| **Getting Started** | `Documentation/USER_GUIDE.md` | Setup & installation |
| **API Reference** | `docs/API.md` | Developer API |
| **Simulator Usage** | `docs/SIMULATOR_GUIDE.md` | How to use app_sim.py |
| **Webcam AI** | `docs/WEBCAM_AI_GUIDE.md` | How to enable webcam detection |
| **Ethical Constraints** | `Documentation/GUARDRAILS.md` | Safety rules & audit |
| **Dataset Bias** | `Documentation/BIAS_ANALYSIS.md` | Data fairness analysis |
| **Code Review** | `CODE_REVIEW.md` | Quality checklist |
| **Contributing** | `CONTRIBUTING.md` | How to contribute |
| **Poster Template** | `Documentation/POSTER_A1.md` | A1 poster draft |

## 🎯 Current Status by Deliverable

| Deliverable | Status | File(s) | Action |
|---|---|---|---|
| **Prototype** | ✅ Done | `Web_Prototype/app.html` + `app.py` | Ready to demo |
| **Dashboard (Streamlit)** | ✅ Done | `Web_Prototype/app.py` | Ready |
| **Simulator** | ✅ Done | `Web_Prototype/app_sim.py` | Ready + guide in `docs/SIMULATOR_GUIDE.md` |
| **Webcam AI** | ⏳ WIP | `app.py` (placeholder) + notebook | Run notebook → update app.py |
| **Source Code** | ✅ Modular | `src/` + `configs/` | Ready |
| **Dataset** | ✅ Complete | `Dataset/` + `docs/BIAS_ANALYSIS.md` | Done |
| **Documentation** | ✅ Complete | `docs/` + `Documentation/` | All guides written |
| **User Guide (PDF)** | ✅ Ready | `Documentation/USER_GUIDE.md` → convert to PDF | Run pandoc |
| **Poster (A1)** | ✅ Draft | `Documentation/POSTER_A1.md` | Edit in PowerPoint |
| **Repository** | ✅ Ready | `.git` + `CONTRIBUTING.md` + `CODE_REVIEW.md` | Push to GitHub |

## 🔧 Commands Summary

### Setup
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Train Webcam Classifier
```bash
jupyter notebook notebooks/01_webcam_classifier_training.ipynb
```

### Run Application
```bash
streamlit run Web_Prototype/app.py
```

### Convert Docs to PDF
```bash
python -c "import pypandoc; pypandoc.convert_file('Documentation/USER_GUIDE.md','pdf','--output=Documentation/USER_GUIDE.pdf')"
```

### Code Quality Check
```bash
flake8 src/
black src/
```

## 📁 Storage Breakdown

| Location | Purpose | Files |
|----------|---------|-------|
| `src/` | Core application logic | models, data, web |
| `configs/` | Configuration parameters | model_config.yaml, safety_constraints.yaml |
| `notebooks/` | Training & experimentation | Jupyter notebooks |
| `docs/` | Developer documentation | API, guides |
| `Documentation/` | User documentation | User guide, guardrails, poster |
| `AI_Model/` | Training scripts | train_*.py |
| `Dataset/` | Data files | CSV, images |
| `Trained_Model/` | Serialized models | .pkl, .pt |
| `Web_Prototype/` | Streamlit apps (legacy) | app.py, app_sim.py, app.html |

---

## ✨ Key Features Now Available

1. **Modular Code** (`src/`)
   - Easy to import: `from src.models import DecisionEngine`
   - Separation of concerns: models, data, web

2. **Configuration Management** (`configs/`)
   - Model parameters in one place
   - Safety constraints version-controlled

3. **Training Notebooks** (`notebooks/`)
   - Webcam classifier training (Jupyter)
   - Reproducible & interactive

4. **Comprehensive Docs** (`docs/`)
   - API reference
   - Usage guides for each view
   - Implementation steps for Webcam AI

5. **Ready-to-Deploy**
   - All components can run
   - Just need to train webcam classifier
   - Then enable Webcam AI view

---

## 🎓 Learning Path

1. **Understand Project**: Read `README.md` + `STRUCTURE_MAPPING.md`
2. **Setup Environment**: Run setup commands above
3. **Explore Prototype**: `streamlit run Web_Prototype/app.py` → test views
4. **Train Webcam**: Run notebook `01_webcam_classifier_training.ipynb`
5. **Enable Webcam AI**: Copy code from `docs/WEBCAM_AI_GUIDE.md` to `app.py`
6. **Test Again**: `streamlit run Web_Prototype/app.py` → test Webcam AI view
7. **Deploy**: Push to GitHub, ready for production

Good luck! 🚀
