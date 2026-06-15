# 🚑 Smart Emergency Traffic System (SETS)

**AI‑powered traffic management system** that detects emergency vehicles and dynamically controls traffic lights using ensemble deep learning.

## 👥 Authors
- **Alifah Fai'zah Rufaidah** (140810240006)  
- **Elsa Rizki Utami** (140810240040)  
- **Adella Safitri Akmaliyah** (140810240094)  

📌 Project submitted for AI course – Smart Emergency Traffic System (SETS).

---

## ✨ Features

| Feature                     | Description                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| **CNN (ResNet18)**          | 84.1% accuracy on emergency vehicle classification (trained on our own dataset)               |
| **YOLOv11** (global)        | 88.9% accuracy on emergency detection (pre‑trained on public dataset)                         |
| **Ensemble OR**             | Combines CNN + YOLO using OR rule → final decision (OPEN_CORRIDOR / CAUTION / NO_ACTION)     |
| **Random Forest**           | 83.3% accuracy for final traffic control decision (based on distance, confidence, density)    |
| **Interactive Prototype**   | HTML canvas simulation with traffic lights, CCTV, and real‑time vehicle spawning              |
| **Streamlit Dashboard**     | AI reasoning engine, planning engine, ethical guardrails, manual override                     |
| **Webcam AI**               | Live emergency classification using CNN + YOLO ensemble (OR rule)                             |
| **Ethical Guardrails**      | Confidence ≥85%, max distance 500m, green light limit 45s, human override                     |
| **Real‑time performance**   | End‑to‑end inference ≈ **32 ms** (0.032 sec) per image                                       |

---

## 📊 Performance Summary (Latest Evaluation)

| Component             | Accuracy | Inference time (avg) |
|-----------------------|----------|----------------------|
| CNN (ResNet18)        | 84.13%   | 25.4 ms              |
| YOLO (global)         | 88.89%   | 38.6 ms              |
| Random Forest         | 83.33%   | 10.6 ms              |
| **Ensemble OR**       | 84.13%   | 32.0 ms (end‑to‑end) |

📈 **Average accuracy (4 components)** = **85.1%**  
⚡ **Total system response time** ≈ 0.032 seconds per image (excluding UI rendering).

---

## 🛠️ Requirements

- Python 3.10 or higher  
- Webcam (optional, for live demo)  
- Internet connection (for first‑time model download)

---

## 📦 Installation

```bash
git clone https://github.com/your-username/Smart-Emergency-Traffic-System.git
cd Smart-Emergency-Traffic-System
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
# source venv/bin/activate
pip install -r requirements.txt