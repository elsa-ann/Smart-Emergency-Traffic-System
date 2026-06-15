# test_yolo.py
from ultralytics import YOLO
import cv2

# Ganti 'path/to/your/image.jpg' dengan lokasi gambar ambulans yang gagal tadi
IMAGE_PATH = 'path/to/your/image.jpg'

# Load model YOLO kamu
model = YOLO('Trained_Model/best.pt')

# Baca gambar dengan OpenCV (format BGR)
img = cv2.imread(IMAGE_PATH)

# Lakukan prediksi
results = model(img, conf=0.25)  # Turunkan threshold dulu jadi 0.25

# Lihat hasilnya
if results[0].boxes is not None:
    print("🎉 YOLO MENDETEKSI SESUATU! 🎉")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        print(f"- {class_name} (Confidence: {conf:.2f})")
else:
    print("❌ YOLO TIDAK MENDETEKSI APA-APA.")

# (Opsional) Tampilkan gambar hasil deteksi
annotated_img = results[0].plot()
cv2.imshow("YOLO Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()