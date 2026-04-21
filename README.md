
<p align="center"> # 🔥 Forest Fire Detection System</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-2.0%2B-green?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-red?logo=yolo" alt="YOLOv8">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform">
</p>

> A real-time AI-powered **Forest Fire & Environmental Hazard Detection System** using **YOLOv8** deep learning models. Detects **Fire**, **Smoke**, **Snow**, and **Fog** from live webcam feeds, uploaded images, and video streams — and instantly sends alerts via **Telegram** and **WhatsApp**.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [How to Run](#-how-to-run)
- [Alert System](#-alert-system)
- [API Routes](#-api-routes)
- [References](#-references)

---

## 📖 About the Project

This system leverages the power of **YOLOv8 (You Only Look Once v8)** — a state-of-the-art real-time object detection model — to identify forest fire hazards in three different input modes:

- 🌄 **Uploaded Images** — Static image analysis
- 🎥 **Video Streams** — Frame-by-frame video analysis
- 🎦 **Live Webcam Feed** — Real-time webcam-based monitoring

When fire or smoke is detected above a configurable confidence threshold, the system automatically:
- Sends a **Telegram Bot** message and photo alert
- Sends a **WhatsApp** alert via Twilio
- Plays an audible **siren alarm** for fire detections
- Logs all alerts to `alerts_log.csv`

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔥 Fire Detection | YOLOv8 model trained on DFireDataset |
| 💨 Smoke Detection | Paired alongside fire detection |
| ❄️ Snow Detection | Secondary YOLOv8 model (informational, no alert) |
| 🌫️ Fog Detection | Third YOLOv8 model (informational, no alert) |
| 📸 Image Upload | Upload an image for instant analysis |
| 🎦 Live Webcam | Real-time detection from local webcam |
| 📩 Telegram Alerts | Bot sends text message + detected frame photo |
| 📲 WhatsApp Alerts | Twilio-powered WhatsApp message + image |
| 🔊 Siren Alarm | Plays `siren.mp3` on fire detection |
| ⏱️ Alert Throttling | Configurable cooldown (default: 30 seconds) |
| 🌐 Flask Web UI | Browser-based frontend for all features |

---

## 🏗️ System Architecture

```
User (Browser)
      │
      ▼
 Flask Web App (app.py)
      │
      ├─── /upload_image ──► process_frame()
      │                           │
      ├─── /webcam_feed   ◄── webcam_detection() [Thread]
      │
      └─── process_frame()
                │
                ├── fire_model  (YOLOv8) ─► Fire/Smoke ─► trigger_alert()
                │                                               │
                ├── snow_model  (YOLOv8) ─► Snow (visual)      ├─► Telegram (message + photo)
                │                                               ├─► WhatsApp (Twilio)
                └── fog_model   (YOLOv8) ─► Fog  (visual)      └─► Siren (mp3 playback)
```

---

## 📁 Project Structure

```
FOREST_FIRE/
├── ABSTRACT.pdf                    # Project abstract document
├── content.jpeg                    # Sample content image
└── Forest-fire-detection/
    ├── README.md                   # Sub-project readme
    ├── render.yaml                 # Render deployment config
    ├── Trained-Models/
    │   ├── best.pt                 # Best trained fire model weights
    │   └── last.pt                 # Last trained model checkpoint
    ├── Training NoteBook/          # Jupyter notebooks for training
    ├── Final results/              # Output images/results from training
    ├── trainValBatch-Img/          # Training/validation batch images
    └── FireDetectionApp/           # 🔥 Main Application
        ├── app.py                  # Flask application entry point
        ├── sendtelegram.py         # Telegram & WhatsApp alert module
        ├── webcam_stream.py        # Standalone webcam stream server
        ├── test_model.py           # Model testing utility
        ├── requirements.txt        # Python dependencies
        ├── siren.mp3               # Siren alarm audio
        ├── alerts_log.csv          # Log of all triggered alerts
        ├── .env                    # Environment variables (not committed)
        ├── weights/
        │   └── best.pt             # Fire detection model weights
        ├── snow_model/             # Snow detection YOLOv8 model
        ├── fog_model/              # Fog detection YOLOv8 model
        ├── templates/
        │   ├── index.html          # Main web page template
        │   └── webcam.html         # Webcam stream page
        ├── static/                 # Static files (CSS, JS, images)
        ├── alerts/                 # Saved alert screenshot images
        └── uploads/                # Temporarily uploaded files
```

---

## 📊 Model Performance

The fire/smoke detection model was trained on the [DFireDataset](https://github.com/gaiasd/DFireDataset):

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|---------|--------------|
| **All** | 1722 | 2275 | 0.907 | 0.854 | **0.923** | 0.650 |
| Fire | 887 | 1002 | 0.943 | 0.914 | 0.963 | 0.732 |
| Smoke | 498 | 1273 | 0.871 | 0.794 | 0.883 | 0.567 |

> ✅ **mAP@0.5 = 0.923** — Excellent detection accuracy for real-world fire scenarios.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **YOLOv8 (Ultralytics)** | Object detection for fire, smoke, snow, fog |
| **Flask** | Lightweight web framework / API |
| **OpenCV (cv2)** | Frame capture, image processing, annotation |
| **Telegram Bot API** | Fire/smoke alert messaging |
| **Twilio (WhatsApp API)** | WhatsApp alert delivery |
| **Playsound** | Siren audio playback on fire detection |
| **NumPy** | Array/image data processing |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8 or higher
- A webcam (for live detection)
- Telegram Bot token & Chat ID
- Twilio account (for WhatsApp alerts)

### 1. Clone the Repository

```bash
git clone https://github.com/Sasi-r08/Forest-fire-detection.git
cd Forest-fire-detection/FireDetectionApp
```

### 2. Create a Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

### Telegram & WhatsApp (in `sendtelegram.py`)

Update the following credentials with your own:

```python
# Telegram
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID   = "YOUR_TELEGRAM_CHAT_ID"

# Twilio WhatsApp
account_sid    = "YOUR_TWILIO_ACCOUNT_SID"
auth_token     = "YOUR_TWILIO_AUTH_TOKEN"
twilio_number  = "whatsapp:+14155238886"
your_whatsapp  = "whatsapp:+91XXXXXXXXXX"

# ngrok public URL (required for WhatsApp image delivery)
BASE_URL = "https://your-ngrok-url.ngrok-free.dev"
```

### Detection Thresholds (in `app.py` / `webcam_stream.py`)

```python
CONF_THRESHOLD = 0.70   # Alert triggered when confidence >= 70%
ALERT_INTERVAL = 30     # Minimum seconds between alerts (cooldown)
```

### Model Paths

```python
FIRE_MODEL_PATH = "weights/best.pt"
SNOW_MODEL_PATH = "snow_model/runs/detect/train/weights/best.pt"
FOG_MODEL_PATH  = "fog_model/runs/detect/train/weights/best.pt"
```

---

## ▶️ How to Run

### Run the Main Flask App

```bash
cd FireDetectionApp
python app.py
```

Then open your browser and navigate to:

```
http://localhost:5000
```

### Run the Standalone Webcam Stream Server

```bash
python webcam_stream.py
```

### (Optional) Expose with ngrok for WhatsApp Image Alerts

```bash
ngrok http 5000
```

Copy the provided HTTPS URL and update `BASE_URL` in `sendtelegram.py`.

---

## 📩 Alert System

When **Fire** or **Smoke** is detected with confidence ≥ **0.70**:

1. 📸 A screenshot of the detected frame is saved in `alerts/`
2. 📩 A **Telegram message** with detection details is sent
3. 🖼️ A **Telegram photo** (the alert frame) is sent
4. 📲 A **WhatsApp text + image** is sent via Twilio
5. 🔊 A **siren alarm** plays (for Fire only)

> ⏱️ Alerts are throttled to once every **30 seconds** to avoid spam.

**Alert message format:**
```
🚨 FOREST ALERT 🚨

Type: Fire
Confidence: 0.87
Time: 2025-05-24 19:18:03
```

---

## 🌐 API Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Main web interface |
| `/upload_image` | POST | Upload an image for fire/smoke/snow/fog detection |
| `/webcam_feed` | GET | Live MJPEG webcam stream with detections |
| `/alerts/<filename>` | GET | Serve saved alert images (used by WhatsApp) |

---

## 📚 References

- 📂 **Dataset:** [DFireDataset](https://github.com/gaiasd/DFireDataset) — A large-scale dataset for fire and smoke detection
- 📖 **Article:** [Stay Ahead of the Flames – Wildfire Detection with YOLOv8](https://medium.com/institute-of-smart-systems-and-ai/stay-ahead-of-the-flames-a-comprehensive-guide-to-wildfire-prevention-with-yolov8-3eb8edd1121a)
- 🤖 **YOLOv8:** [Ultralytics Documentation](https://docs.ultralytics.com/)
- 🤖 **Telegram Bot API:** [Official Docs](https://core.telegram.org/bots/api)
- 📱 **Twilio WhatsApp API:** [Official Docs](https://www.twilio.com/docs/whatsapp)

---

## 📄 License

This project is licensed under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).

---

<p align="center">
  Made with ❤️ for wildfire prevention and forest safety<br/>
  <strong>🔥 Detect Early. Alert Fast. Save Lives. 🔥</strong>
</p>

