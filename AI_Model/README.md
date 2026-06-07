# Webcam Classifier Training

This folder includes a training script for a webcam-based image classifier capable of distinguishing emergency vehicles (ambulance/firetruck) from ordinary vehicles.

## Dataset structure

Prepare your images in this format:

```
Dataset/webcam/
  train/
    ambulance/
      img1.jpg
      img2.jpg
      ...
    non_ambulance/
      img1.jpg
      img2.jpg
      ...
  val/
    ambulance/
      img1.jpg
      ...
    non_ambulance/
      img1.jpg
      ...
```

## How to train

From the repository root:

```bash
python AI_Model/train_webcam_classifier.py
```

This will create:

- `Trained_Model/webcam_classifier.pth`
- `Trained_Model/webcam_classes.json`

## How to use

Open the Streamlit app and choose the `Webcam AI` view in `Web_Prototype/app.py`. Capture an image with your laptop camera or upload one, then the model will classify it.

## Notes

- Use at least 30–50 images per class for a simple prototype.
- More variation in angle, lighting, and background improves performance.
- If training takes too long, reduce `EPOCHS` or use a smaller batch size.
