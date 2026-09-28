import cv2
from config import Config
from vision import Camera

class VisionControlApp:
    def __init__(self):
        self.config = Config()
        self.camera = Camera(camera_index=self.config.CAMERA_INDEX, target_fps=self.config.TARGET_FPS)
        self.running = False

    def start(self):
        print("Starting VisionControl...")
        try:
            self.camera.start()
            self.running = True
            self.run_loop()
        except Exception as e:
            print(f"Error starting application: {e}")
        finally:
            self.cleanup()

    def run_loop(self):
        print("VisionControl running. Press 'q' or 'ESC' to quit.")
        while self.running:
            success, frame, fps = self.camera.get_frame()
            if not success:
                print("Failed to grab frame. Exiting...")
                break

            # Add FPS to the frame (Basic presentation for Phase 1)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(self.config.WINDOW_NAME, frame)

            # Wait for 1 ms, and check if user pressed 'q' or 'ESC'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.running = False

    def cleanup(self):
        print("Shutting down VisionControl...")
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    app = VisionControlApp()
    app.start()
