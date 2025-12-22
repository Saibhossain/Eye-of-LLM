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

#
# # --- DEBUG/TEST BLOCK ---
# if __name__ == "__main__":
#     print("Testing Camera...")
#     try:
#         cam = Camera()
#         print("Camera initialized. Press 'q' to quit.")
#         while True:
#             frame = cam.get_frame()
#             if frame is None:
#                 print("Failed to get frame!")
#                 break
#
#             # Draw a crosshair to prove we can edit the frame
#             cx, cy = cam.get_center_coordinates(frame)
#             cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
#             cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
#
#             cv2.imshow("Camera Test", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         cam.release()
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Camera Test Failed: {e}")