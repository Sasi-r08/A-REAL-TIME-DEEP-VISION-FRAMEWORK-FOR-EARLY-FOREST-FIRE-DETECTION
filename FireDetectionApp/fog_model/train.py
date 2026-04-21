from ultralytics import YOLO
import os

# Check data.yaml exists
if not os.path.exists("data.yaml"):
    print("❌ data.yaml not found!")
    exit()

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train Fog Model
model.train(
    data="data.yaml",
    epochs=70,        # little higher training
    imgsz=640,
    batch=8,
    device="cpu",
    patience=25,      # early stopping
    optimizer="Adam",
    lr0=0.001
)

print("\n✅ Fog model training completed!")
print("📁 Model saved at: runs/detect/train/weights/best.pt")