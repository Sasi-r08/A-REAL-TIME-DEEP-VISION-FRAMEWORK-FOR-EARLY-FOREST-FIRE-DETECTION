from ultralytics import YOLO
import cv2
import os

# ==========================
# LOAD TRAINED SNOW MODEL
# ==========================

model_path = "runs/detect/train/weights/best.pt"

if not os.path.exists(model_path):
    print("❌ Snow model not found. Train first.")
    exit()

model = YOLO(model_path)

# ==========================
# INPUT IMAGE
# ==========================

image_path = input("📂 Enter image path: ").strip().strip('"')

if not os.path.exists(image_path):
    print("❌ Image not found!")
    exit()

# ==========================
# RUN DETECTION
# ==========================

results = model.predict(
    source=image_path,
    conf=0.25,   # confidence threshold
    save=False,
    verbose=False
)

result = results[0]

img = cv2.imread(image_path)

# ==========================
# DRAW BOXES + CONFIDENCE
# ==========================

if len(result.boxes) == 0:
    print("⚠ No Snow Detected")
else:
    print("❄ Snow Detected!")

for box in result.boxes:
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    label = f"SNOW {conf:.2f}"

    # Blue color for snow
    color = (255, 0, 0)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        img,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

# ==========================
# SHOW RESULT
# ==========================

cv2.imshow("Snow Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()