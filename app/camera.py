import cv2

def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot access Camera")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture image")

    return frame

# if __name__ =="__main__":
#     img = capture_image()
#     cv2.imshow("Camera Test", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()