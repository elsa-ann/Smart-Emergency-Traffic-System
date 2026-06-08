# User Guide — SECS

This guide explains how to install, run, and evaluate the Smart Emergency Corridor System.

## 1. Requirements
- OS: Windows / Linux
- Python 3.10+ recommended
- Virtual environment (venv)

## 2. Install
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # PowerShell (Windows)
pip install -r requirements.txt
```

## 3. Running the Streamlit app
```bash
streamlit run Web_Prototype/app.py
```

Views available (sidebar `View`):
- Prototype (HTML) — opens `Web_Prototype/app.html` embedded
- Simulator — lightweight dataset-driven scenario player
- Streamlit Dashboard — main AI dashboard
- Webcam AI — demo webcam capture

## 4. Training models (outline)
- YOLO training: `python AI_Model/train_yolo.py` (see `AI_Model` for dataset yaml)
- Decision model: `python AI_Model/train_decision_model.py`
- Webcam classifier: `python AI_Model/train_webcam_classifier.py`

## 5. Audit & Logs
- Decision logs: `logs/decisions.csv` (create if missing)
- Model artifacts: `Trained_Model/` (place `best.pt`, `decision_model.pkl`, `label_encoders.pkl`)

## 6. Ethical Impact Assessment (short)
- Primary harms: false positives (unnecessary traffic disruption), false negatives (failure to prioritize)
- Mitigations: high confidence threshold (85%), human override, audit trail

## 7. Converting docs to PDF
We provide Markdown files under `Documentation/`. Convert to PDF using Pandoc:
```bash
pip install pandoc pypandoc
pypandoc --from markdown --to pdf -o Documentation/USER_GUIDE.pdf Documentation/USER_GUIDE.md
```

## 8. Contact & Support
Create a GitHub issue with label `support` or email project maintainer.
