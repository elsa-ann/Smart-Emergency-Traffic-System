# Smart Emergency Traffic System (SETS)

AI-powered system for emergency vehicle preemption using YOLO object detection and Random Forest decision making.

## Authors

- **Alifah Fai'zah Rufaidah** (140810240006)
- **Elsa Rizki Utami** (140810240040)
- **Adella Safitri Akmaliyah** (140810240094)

Project submitted for AI course – Smart Emergency Traffic System (SETS).

## Features
- **YOLOv11** (99.4% mAP50) – detects ambulances & firetrucks from CCTV feeds
- **Random Forest** (95% accuracy) – decides when to open an emergency corridor
- **Interactive Prototype** – HTML canvas simulation with traffic lights, CCTV, and vehicle movement
- **Streamlit Dashboard** – real‑time AI reasoning, planning, ethical guardrails
- **Webcam AI** – live classification using a lightweight Random Forest model (trained on your own dataset)
- **Ethical Guardrails** – confidence threshold (85%), max distance (500m), green light limit (45s), human override

## Requirements
- Python 3.10+
- Webcam (optional)

## Installation

```bash
git clone https://github.com/your-username/Smart-Emergency-Traffic-System.git
cd Smart-Emergency-Traffic-System
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate (Linux/Mac)
pip install -r requirements.txt