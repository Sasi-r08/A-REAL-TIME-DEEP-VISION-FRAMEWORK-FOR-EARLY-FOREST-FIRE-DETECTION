import cv2
import time
import threading
from queue import Queue
from flask import Flask, Response, render_template, send_from_directory
from ultralytics import YOLO
from sendtelegram import (
    send_telegram_message,
    send_telegram_photo,
    send_whatsapp_alert,
)
import os

# ── CPU ONLY ─────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# ── MODEL PATHS ──────────────────────────
FIRE_MODEL_PATH = "weights/best.pt"
SNOW_MODEL_PATH = "snow_model/runs/detect/train/weights/best.pt"
FOG_MODEL_PATH  = "fog_model/runs/detect/train/weights/best.pt"

CONF_THRESHOLD = 0.70
ALERT_INTERVAL = 30  # seconds

# ── LOAD MODELS ──────────────────────────
fire_model = YOLO(FIRE_MODEL_PATH)
snow_model = YOLO(SNOW_MODEL_PATH)
fog_model  = YOLO(FOG_MODEL_PATH)

fire_model.to("cpu")
snow_model.to("cpu")
fog_model.to("cpu")

# ── FLASK APP ────────────────────────────
app = Flask(__name__)

CLASS_MAP = {0: "Smoke", 1: "Fire"}
COLORS = {"Fire": (0, 0, 255), "Smoke": (0, 255, 255)}

frame_queue = Queue(maxsize=1)
last_alert_time = 0

# ── SERVE ALERT IMAGES (VERY IMPORTANT FOR WHATSAPP) ──
@app.route('/alerts/<filename>')
def serve_alert(filename):
    return send_from_directory('alerts', filename)

# ── WEBCAM DETECTION THREAD ──────────────
def webcam_detection():
    global last_alert_time
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not accessible")
        return

    print("✅ Webcam Started")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fire_detected = False

        # 1️⃣ FIRE / SMOKE
        fire_results = fire_model(frame)[0]

        for det in fire_results.boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            cls_name = CLASS_MAP.get(cls_id, "Unknown")
            x1, y1, x2, y2 = map(int, det.xyxy[0])

            color = COLORS.get(cls_name, (255, 255, 255))

            if (
                cls_name in ["Fire", "Smoke"]
                and conf >= CONF_THRESHOLD
                and time.time() - last_alert_time > ALERT_INTERVAL
            ):
                fire_detected = True

                os.makedirs("alerts", exist_ok=True)
                filename = f"{cls_name}_{int(time.time())}.jpg"
                image_path = f"alerts/{filename}"
                cv2.imwrite(image_path, frame)

                text_message = (
                    f"🚨 FOREST ALERT 🚨\n"
                    f"Type: {cls_name}\n"
                    f"Confidence: {conf:.2f}\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )

                # TELEGRAM
                send_telegram_message(text_message)
                send_telegram_photo(image_path, f"{cls_name} detected!")

                # WHATSAPP
                send_whatsapp_alert(text_message, image_path)

                last_alert_time = time.time()

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        # 2️⃣ SNOW + FOG (ONLY IF NO FIRE)
        if not fire_detected:

            snow_results = snow_model(frame, conf=0.4)[0]
            for box in snow_results.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"Snow {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

            fog_results = fog_model(frame, conf=0.4)[0]
            for box in fog_results.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Fog {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

    cap.release()

# ── STREAM TO BROWSER ─────────────────────
def gen_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
        else:
            time.sleep(0.01)

# ── ROUTES ───────────────────────────────
@app.route("/")
def index():
    return render_template("webcam.html")

@app.route("/webcam_feed")
def webcam_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

# ── MAIN ─────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=webcam_detection, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)