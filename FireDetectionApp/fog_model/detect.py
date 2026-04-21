from ultralytics import YOLO
import cv2
import os

# ==========================
# LOAD TRAINED FOG MODEL
# ==========================

model_path = "runs/detect/train/weights/best.pt"

if not os.path.exists(model_path):
    print("❌ Fog model not found. Train first.")
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
    conf=0.25,
    save=False,
    verbose=False
)

result = results[0]
img = cv2.imread(image_path)

# ==========================
# DRAW BOX + CONFIDENCE
# ==========================

if len(result.boxes) == 0:
    print("⚠ No Fog Detected")
else:
    print("🌫 Fog Detected!")

for box in result.boxes:
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    label = f"FOG {conf:.2f}"
    color = (0, 255, 0)

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # Text position adjustment
    text_y = y1 - 10 if y1 - 10 > 30 else y1 + 40

    # Background box for text
    (text_w, text_h), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        2
    )

    cv2.rectangle(
        img,
        (x1, text_y - text_h - 10),
        (x1 + text_w, text_y + 5),
        color,
        -1
    )

    # Put text
    cv2.putText(
        img,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),  # black text
        2
    )

# ==========================
# SHOW RESULT
# ==========================

cv2.imshow("Fog Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()