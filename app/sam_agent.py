import os
import cv2
import requests

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_SAM_MODEL")

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def frame_to_bytes(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()

def detect_objects(frame):
    image_bytes = frame_to_bytes(frame)

    files = {
        "image": ("image.jpg", image_bytes, "image/jpeg")
    }

    response = requests.post(API_URL, headers=HEADERS, files=files)
    response.raise_for_status()

    return response.json()
