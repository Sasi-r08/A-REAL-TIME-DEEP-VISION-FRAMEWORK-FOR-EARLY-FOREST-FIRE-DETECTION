from ultralytics import YOLO
import os

# Check dataset config
if not os.path.exists("data.yaml"):
    print("❌ data.yaml not found!")
    exit()

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Start training
model.train(
    data="data.yaml",
    epochs=60,
    imgsz=640,
    batch=8,
    device="cpu",     # Your system CPU only
    patience=20,
    optimizer="Adam",
    lr0=0.001
)

print("\n✅ Snow model training completed!")
print("📁 Model saved at: runs/detect/train/weights/best.pt")