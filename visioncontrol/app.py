import cv2
import sys
from config import Config
from vision import Camera
from ui.hud import HUD

class VisionControlApp:
    def __init__(self):
        self.config = Config()
        self.camera = Camera(
            camera_index=self.config.CAMERA_INDEX, 
            target_fps=self.config.TARGET_FPS,
            width=self.config.FRAME_WIDTH,
            height=self.config.FRAME_HEIGHT
        )
        self.hud = HUD()
        self.running = False

    def start(self):
        print("Initializing VisionControl...")
        try:
            self.camera.start()
            self.running = True
            self.run_loop()
        except RuntimeError as e:
            print(f"\n[FATAL ERROR] {e}")
            print("Please ensure your camera is connected and permitted in OS settings.\n")
            sys.exit(1)
        except Exception as e:
            print(f"\n[UNEXPECTED ERROR] {e}\n")
            sys.exit(1)
        finally:
            self.cleanup()

    def run_loop(self):
        print(f"VisionControl active on Camera {self.config.CAMERA_INDEX}. Press 'q' or 'ESC' to quit.")
        while self.running:
            success, frame, fps = self.camera.get_frame()
            if not success:
                print("\n[WARNING] Failed to grab frame. Attempting recovery or exiting...\n")
                break

            # Render UI
            frame = self.hud.render(frame, fps)

            # Display frame
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
