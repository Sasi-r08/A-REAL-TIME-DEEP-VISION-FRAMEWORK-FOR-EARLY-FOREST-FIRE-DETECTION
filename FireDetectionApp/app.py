import os
import cv2
import time
import threading
import numpy as np
from io import BytesIO
from queue import Queue
from flask import Flask, render_template, request, send_file, Response, send_from_directory, jsonify
from ultralytics import YOLO
from sendtelegram import (
    send_telegram_message,
    send_telegram_photo,
    send_whatsapp_alert
)
from playsound import playsound

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ==============================
# MODEL PATHS
# ==============================
FIRE_MODEL_PATH = "weights/best.pt"
SNOW_MODEL_PATH = "snow_model/runs/detect/train/weights/best.pt"
FOG_MODEL_PATH  = "fog_model/runs/detect/train/weights/best.pt"

CONF_THRESHOLD = 0.70
ALERT_INTERVAL = 30
last_alert_time = 0

SIREN_PATH = os.path.join(os.getcwd(), "siren.mp3")

# ==============================
# LOAD MODELS
# ==============================
fire_model = YOLO(FIRE_MODEL_PATH)
snow_model = YOLO(SNOW_MODEL_PATH)
fog_model  = YOLO(FOG_MODEL_PATH)

fire_model.to("cpu")
snow_model.to("cpu")
fog_model.to("cpu")

app = Flask(__name__)

CLASS_MAP = {0: "Smoke", 1: "Fire"}

frame_queue = Queue(maxsize=1)

# ==============================
# SERVE ALERT IMAGES
# ==============================
@app.route('/alerts/<filename>')
def serve_alert(filename):
    return send_from_directory('alerts', filename)

# ==============================
# ALERT FUNCTION (Fire & Smoke Only)
# ==============================
def trigger_alert(frame, cls_name, conf):
    global last_alert_time

    if time.time() - last_alert_time < ALERT_INTERVAL:
        return

    try:
        os.makedirs("alerts", exist_ok=True)

        timestamp = int(time.time())
        filename = f"{cls_name}_{timestamp}.jpg"
        image_path = os.path.join("alerts", filename)

        cv2.imwrite(image_path, frame)

        message = (
            f"🚨 FOREST ALERT 🚨\n\n"
            f"Type: {cls_name}\n"
            f"Confidence: {conf:.2f}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Telegram
        send_telegram_message(message)
        send_telegram_photo(image_path, message)

        # WhatsApp
        send_whatsapp_alert(message, image_path)


        # Siren only for Fire
        if cls_name == "Fire" and os.path.exists(SIREN_PATH):
            threading.Thread(
                target=playsound,
                args=(SIREN_PATH,),
                daemon=True
            ).start()

        print("✅ Alert Sent Successfully")
        last_alert_time = time.time()

    except Exception as e:
        print("❌ Alert Error:", e)

# ==============================
# FRAME PROCESSOR
# ==============================
def process_frame(frame):

    # 🔥 FIRE + 💨 SMOKE (WITH ALERT)
    fire_results = fire_model(frame)[0]

    for det in fire_results.boxes:
        cls_id = int(det.cls[0])
        conf = float(det.conf[0])
        cls_name = CLASS_MAP.get(cls_id)

        if not cls_name:
            continue

        x1, y1, x2, y2 = map(int, det.xyxy[0])
        color = (0, 0, 255) if cls_name == "Fire" else (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame,
                    f"{cls_name} | {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

        if conf >= CONF_THRESHOLD:
            trigger_alert(frame, cls_name, conf)

    # ❄ SNOW (NO ALERT)
    snow_results = snow_model(frame)[0]

    for det in snow_results.boxes:
        conf = float(det.conf[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0])

        color = (0, 255, 0)  # Green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame,
                    f"Snow | {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

    # 🌫 FOG (NO ALERT)
    fog_results = fog_model(frame)[0]

    for det in fog_results.boxes:
        conf = float(det.conf[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0])

        color = (255, 0, 0)  # Blue (BGR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame,
                    f"Fog | {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

    return frame

# ==============================
# IMAGE DETECTION
# ==============================
@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files.get("file")
    if not file:
        return "No file", 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = process_frame(img)

    _, buf = cv2.imencode(".png", img)
    return send_file(BytesIO(buf.tobytes()), mimetype="image/png")

# ==============================
# WEBCAM DETECTION
# ==============================
def webcam_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not accessible")
        return

    print("✅ Webcam Started")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        processed = process_frame(frame)

        if frame_queue.full():
            frame_queue.get()

        frame_queue.put(processed)

# ==============================
# WEBCAM STREAM
# ==============================
@app.route("/webcam_feed")
def webcam_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def gen_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )
        else:
            time.sleep(0.01)

@app.route("/")
def index():
    return render_template("index.html")

# ==============================
if __name__ == "__main__":
    threading.Thread(target=webcam_detection, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)