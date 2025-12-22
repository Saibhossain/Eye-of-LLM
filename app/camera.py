import cv2

class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam. Check macOS permissions for Terminal/PyCharm.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def get_center_coordinates(self, frame):
        """Returns the (x, y) tuple of the center of the frame."""
        h, w, _ = frame.shape
        return w // 2, h // 2

    def release(self):
        self.cap.release()