# API Documentation — SECS

## Decision Engine

### `DecisionEngine` class
Located: `src/models/decision_engine.py`

```python
from src.models import DecisionEngine

engine = DecisionEngine()
decision, confidence = engine.predict(
    vehicle_type='ambulance',
    traffic_density='medium',
    time_of_day='afternoon',
    weather='clear',
    distance_m=300,
    confidence_pct=92.5,
    vehicle_count=5
)

# Output:
# decision = 'OPEN_CORRIDOR' | 'CAUTION' | 'NO_ACTION'
# confidence = float (0-100)
```

### Input Parameters
- `vehicle_type`: str — 'ambulance', 'firetruck', 'car', 'bus', 'truck', 'motorcycle'
- `traffic_density`: str — 'clear', 'low', 'medium', 'high', 'jam'
- `time_of_day`: str — 'morning', 'afternoon', 'evening', 'night'
- `weather`: str — 'clear', 'rain', 'fog'
- `distance_m`: float — meters to intersection
- `confidence_pct`: float — detection confidence (60-99)
- `vehicle_count`: int — vehicles at intersection

### Output
- `decision`: str — 'OPEN_CORRIDOR', 'CAUTION', or 'NO_ACTION'
- `confidence`: float — probability (0-100)

## YOLO Detector

### `YOLODetector` class
Located: `src/models/yolo_detector.py`

```python
from src.models import YOLODetector

detector = YOLODetector(model_path='Trained_Model/best.pt')
detections = detector.detect(frame)  # frame = numpy array

# detections = [
#   {'class': 'ambulance', 'confidence': 0.95, 'bbox': [x1, y1, x2, y2]},
#   ...
# ]
```

## Data Loaders

### `load_traffic_decisions()`
```python
from src.data import load_traffic_decisions

df = load_traffic_decisions('Dataset/traffic_decisions.csv')
print(df.head())
# Columns: vehicle_type, traffic_density, time_of_day, weather, 
#          distance_m, confidence_pct, vehicle_count, decision
```

### `load_emergency_vehicles()`
```python
from src.data import load_emergency_vehicles

df = load_emergency_vehicles('Dataset/emergency-vehicles/train.csv')
# Columns: image_names, emergency_or_not
```

## Webcam Classifier

### `predict_vehicle_type()`
Located: `notebooks/01_webcam_classifier_training.ipynb`

```python
import joblib
from pathlib import Path

def predict_vehicle_type(image_path):
    model = joblib.load('Trained_Model/webcam_classifier.pkl')
    features = extract_features(image_path)
    pred_class = model.predict([features])[0]
    confidence = model.predict_proba([features])[0].max()
    return ['Regular Vehicle', 'Emergency Vehicle'][pred_class], confidence
```

### Usage in Streamlit
```python
import streamlit as st
from PIL import Image

camera_image = st.camera_input("Capture vehicle")
if camera_image is not None:
    image = Image.open(camera_image)
    pred_class, conf = predict_vehicle_type(image)
    st.write(f"Predicted: {pred_class} ({conf:.2%})")
```
