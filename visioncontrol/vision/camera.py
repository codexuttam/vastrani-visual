import cv2
import time

class Camera:
    def __init__(self, camera_index=0, target_fps=30):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.cap = None
        self.prev_time = 0

    def start(self):
        """Initializes and starts the webcam."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

    def get_frame(self):
        """Reads a frame from the webcam, calculates FPS, and returns it."""
        if not self.cap:
            return False, None, 0.0

        success, frame = self.cap.read()
        if not success:
            return False, None, 0.0

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time) if self.prev_time > 0 else 0.0
        self.prev_time = current_time

        # Flip horizontally for selfie-view
        frame = cv2.flip(frame, 1)

        return True, frame, fps

    def stop(self):
        """Releases the webcam resources."""
        if self.cap:
            self.cap.release()
