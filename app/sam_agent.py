import os, cv2, requests
from pathlib import Path
from dotenv import load_dotenv
from camera import capture_image

current_dir = Path(__file__).resolve().parent
env_path = current_dir.parent / '.env.example'
load_result = load_dotenv(dotenv_path=".env.example")

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_SAM_MODEL")

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

print("HF_API_KEY loaded:", HF_API_KEY is not None)
print("HF_MODEL:", HF_MODEL)

def frame_to_bytes(frame):
    _, buffer = cv2.imencode(".jpg",frame)
    return buffer.tobytes()

def detect_objects(frame):
    image_bytes = frame_to_bytes(frame)
    files = {
        "image":("image.jpg",image_bytes,"image/jpeg")
    }
    response = requests.post(API_URL,headers=HEADERS,files=files)
    response.raise_for_status()

    return response.json()

if __name__ == "__main__":
    frame = capture_image()
    output = detect_objects(frame)
    print("sam output key :",output.keys())