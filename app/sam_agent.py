import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from camera import capture_image

MODEL_PATH = Path("/Users/mdsaibhossain/code/python/Eye-of-LLM/sam_model_test/yolo11n.pt")
model = YOLO(MODEL_PATH)

def detect_objects(frame):
    results = model(frame)
    objects = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

    for bbox, conf, cls in zip(boxes, confs, classes):
        objects.append({
            "bbox": bbox.tolist(),
            "confidence": float(conf),
            "class_id": int(cls),
            "class_name": model.names[int(cls)]  # YOLO provides class names
        })

    return objects

# if __name__ == "__main__":
#     frame = capture_image()
#     output = detect_objects(frame)
#     print("Detected objects:")
#     for obj in output:
#         print(obj)