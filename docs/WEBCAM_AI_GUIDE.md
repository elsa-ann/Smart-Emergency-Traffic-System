# Webcam AI — Implementation Guide

## Overview
`Webcam AI` adalah view di Streamlit yang memungkinkan capture dari webcam dan melakukan live vehicle detection.

**Status**: Currently demo (placeholder webcam capture). Penuh fungsionalitas membutuhkan:
1. ✅ Webcam classifier model (ada di `notebooks/01_webcam_classifier_training.ipynb`)
2. ✅ Feature extractor (ada di notebook)
3. ⏳ Integration ke Streamlit (ada di `app.py`, tapi placeholder)

## Current Implementation (Demo)

Di `Web_Prototype/app.py`, view `Webcam AI` saat ini:

```python
elif view == "Webcam AI":
    st.title("Webcam AI — Emergency Vehicle Detector")
    st.markdown("Use your webcam to capture an approaching vehicle and simulate emergency detection.")
    camera_image = st.camera_input("Point the camera at the vehicle")
    if camera_image is not None:
        st.image(camera_image, caption="Captured image", use_column_width=True)
        st.markdown("---")
        st.markdown("**Prediction**: This feature is currently in demo mode. Train a webcam classifier and connect it with live inference.")
    else:
        st.info("Please enable your webcam and allow access to preview frames.")
    st.stop()
```

## What's Missing?

1. **Load trained model**
   ```python
   import joblib
   model = joblib.load('Trained_Model/webcam_classifier.pkl')
   ```

2. **Feature extractor function**
   ```python
   def extract_features(image):
       # Convert PIL image to numpy
       img_np = np.array(image)
       # Extract color histogram, edges, etc.
       # Return feature vector
   ```

3. **Prediction logic**
   ```python
   features = extract_features(camera_image)
   pred_class = model.predict([features])[0]
   confidence = model.predict_proba([features])[0].max()
   ```

## How to Enable Full Webcam AI

### Step 1: Train Webcam Classifier (if not already done)

```bash
# Buka Jupyter
jupyter notebook notebooks/01_webcam_classifier_training.ipynb

# Run semua cells untuk train model
# Output: Trained_Model/webcam_classifier.pkl
```

### Step 2: Update `Web_Prototype/app.py`

Replace placeholder di view `Webcam AI`:

```python
elif view == "Webcam AI":
    st.title("🎥 Webcam AI — Emergency Vehicle Detector")
    st.markdown("Capture a vehicle image from your webcam dan AI akan mengklasifikasi apakah emergency atau regular vehicle.")
    
    import joblib
    from PIL import Image
    import numpy as np
    
    # Load model & feature extractor
    @st.cache_resource
    def load_webcam_model():
        model = joblib.load('Trained_Model/webcam_classifier.pkl')
        return model
    
    def extract_features(image, img_size=(128, 128)):
        import cv2
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, img_size)
        
        # Color histogram
        hist_b = cv2.calcHist([img_cv], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([img_cv], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([img_cv], [2], None, [32], [0, 256])
        hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        
        # Edge detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_feature = np.array([edges.mean(), edges.std(), edges.max()])
        
        return np.concatenate([hist, edge_feature])
    
    # Capture image
    camera_image = st.camera_input("📷 Arahkan kamera ke kendaraan")
    
    if camera_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(camera_image, caption="Captured Image", use_column_width=True)
        
        with col2:
            try:
                model = load_webcam_model()
                features = extract_features(camera_image)
                
                pred_class_idx = model.predict([features])[0]
                confidence = model.predict_proba([features])[0].max()
                
                class_names = ['🚗 Regular Vehicle', '🚑 Emergency Vehicle']
                pred_class = class_names[pred_class_idx]
                
                st.markdown("### Prediction Result")
                
                if pred_class_idx == 1:  # Emergency
                    st.success(f"**{pred_class}**")
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.warning("⚠️ EMERGENCY DETECTED - Open corridor?")
                else:  # Regular
                    st.info(f"**{pred_class}**")
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.success("✓ No action needed")
            
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.info("📹 Please enable your webcam and capture a vehicle image")
    
    st.stop()
```

### Step 3: Update requirements.txt

```
opencv-python-headless
# or
opencv-python
```

## Advanced: Real-time Detection

Untuk real-time detection dari webcam (bukan single capture):

```python
import cv2
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Install: pip install streamlit-webrtc

rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
webrtc_ctx = webrtc_streamer(key="streamlit-webrtc-livenesscheck", rtc_configuration=rtc_configuration, media_stream_constraints={"video": True})

if webrtc_ctx.state.playing:
    frame = webrtc_ctx.state.video_processor.frame
    if frame is not None:
        features = extract_features(frame)
        pred_class = model.predict([features])[0]
        # Display result
```

## Testing

```python
# Test dengan image file
from PIL import Image

test_image = Image.open('Dataset/emergency-vehicles/train/ambulance_001.jpg')
features = extract_features(test_image)
pred = model.predict([features])[0]
conf = model.predict_proba([features])[0].max()

print(f"Prediction: {['Regular', 'Emergency'][pred]}")
print(f"Confidence: {conf:.2%}")
```

## Summary

| Komponen | Status | File |
|----------|--------|------|
| Webcam capture | ✅ Done | `app.py` (Streamlit) |
| Model training | ✅ Done | `notebooks/01_webcam_classifier_training.ipynb` |
| Feature extractor | ✅ Done | Notebook |
| Integration | ⏳ TODO | Update `app.py` with code above |
| Real-time streaming | ⏳ Optional | Use `streamlit-webrtc` |

Jadi untuk full Webcam AI, tinggal:
1. Train model di notebook (sudah ada)
2. Copy code di atas ke `app.py` (replace placeholder)
3. Test dengan image dari webcam
