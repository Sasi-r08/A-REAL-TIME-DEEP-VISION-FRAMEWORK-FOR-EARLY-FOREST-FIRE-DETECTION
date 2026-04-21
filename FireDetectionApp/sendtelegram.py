import requests
import os
from twilio.rest import Client

# ======================================
# 🔴 TELEGRAM DETAILS (UNCHANGED)
# ======================================
BOT_TOKEN = "*****************************************"
CHAT_ID = "************************"

# ======================================
# 🔴 TWILIO WHATSAPP DETAILS
# ======================================
import os

account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

twilio_number = "whatsapp:+14**********)"
your_whatsapp = "whatsapp:+91**********"

# Create client
client = Client(account_sid, auth_token)

# ======================================
# NGROK BASE URL (UPDATE DAILY)
# ======================================
BASE_URL = "https://inge-scrupulous-jeana.ngrok-free.dev"

# ======================================
# TELEGRAM MESSAGE
# ======================================
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

        response = requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": message
        })

        print("Telegram Message Status:", response.status_code)

    except Exception as e:
        print("❌ Telegram Message Error:", e)


# ======================================
# TELEGRAM PHOTO
# ======================================
def send_telegram_photo(image_path, caption=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

        with open(image_path, "rb") as photo:
            response = requests.post(
                url,
                files={"photo": photo},
                data={
                    "chat_id": CHAT_ID,
                    "caption": caption
                }
            )

        print("Telegram Photo Status:", response.status_code)

    except Exception as e:
        print("❌ Telegram Photo Error:", e)


# ======================================
# WHATSAPP ALERT
# ======================================
def send_whatsapp_alert(message, image_path):
    try:
        filename = os.path.basename(image_path)
        image_url = f"{BASE_URL}/alerts/{filename}"

        print("WhatsApp Image URL:", image_url)

        # Send Text
        text_msg = client.messages.create(
            from_=twilio_number,   # ✅ FIX HERE
            body=message,
            to=your_whatsapp
        )
        print("WhatsApp Text SID:", text_msg.sid)

        # Send Image
        image_msg = client.messages.create(
            from_=twilio_number,   # ✅ FIX HERE
            body="🔥 Fire / Smoke Detected Screenshot",
            media_url=[image_url],
            to=your_whatsapp
        )
        print("WhatsApp Image SID:", image_msg.sid)

        print("✅ WhatsApp Message + Image Sent Successfully")

    except Exception as e:
        print("❌ WhatsApp Error:", e)