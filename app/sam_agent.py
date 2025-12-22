import os
import cv2
import requests
import base64
from pathlib import Path
from dotenv import load_dotenv
from camera import capture_image

current_dir = Path(__file__).resolve().parent
env_path = current_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_SAM_MODEL")

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

print("HF_API_KEY loaded:", HF_API_KEY is not None)
print("HF_MODEL:", HF_MODEL)

def frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

def detect_objects(frame):
    image_base64 = frame_to_base64(frame)

    payload = {
        "inputs": image_base64
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 503:
        raise RuntimeError("Model is loading, retry in 20 seconds")

    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    frame = capture_image()
    output = detect_objects(frame)
    print("SAM output keys:", output.keys())
