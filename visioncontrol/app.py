"""
app.py — VisionControl Application Coordinator

Responsibilities:
    - Initialise and own application lifecycle
    - Coordinate Camera → HandTracker → FeatureExtractor → HUD pipeline
    - Handle clean startup and shutdown
    - NOT contain business logic
"""

import cv2
import sys

from config import Config
from vision import Camera, HandTracker
from gestures import FeatureExtractor
from ui.hud import HUD


class VisionControlApp:
    def __init__(self):
        self.config = Config()
        self.camera = Camera(
            camera_index=self.config.CAMERA_INDEX,
            target_fps=self.config.TARGET_FPS,
            width=self.config.FRAME_WIDTH,
            height=self.config.FRAME_HEIGHT,
        )
        self.hand_tracker = HandTracker(
            max_num_hands=self.config.MAX_NUM_HANDS,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE,
        )
        self.feature_extractor = FeatureExtractor(
            alpha=self.config.SMOOTHING_ALPHA,
            finger_extension_threshold=self.config.FINGER_EXTENSION_THRESHOLD,
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
        print(
            f"VisionControl active on Camera {self.config.CAMERA_INDEX}. "
            f"Press 'q' or 'ESC' to quit."
        )
        while self.running:
            # ── 1. Capture frame ─────────────────────────────────────────────
            success, frame, fps = self.camera.get_frame()
            if not success:
                print("\n[WARNING] Failed to grab frame. Attempting recovery...\n")
                break

            # ── 2. Detect hand landmarks (Phase 2) ───────────────────────────
            frame, landmarks_list = self.hand_tracker.process(frame)

            # ── 3. Extract features (Phase 3) ────────────────────────────────
            landmarks = landmarks_list[0] if landmarks_list else None
            features = self.feature_extractor.extract(landmarks)

            # ── 4. Render HUD ────────────────────────────────────────────────
            frame = self.hud.render(
                frame, fps,
                features=features,
                debug=self.config.DEBUG_FEATURES,
            )

            # ── 5. Display ───────────────────────────────────────────────────
            cv2.imshow(self.config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                self.running = False

    def cleanup(self):
        print("Shutting down VisionControl...")
        self.camera.stop()
        self.hand_tracker.close()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


if __name__ == "__main__":
    app = VisionControlApp()
    app.start()
