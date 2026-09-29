"""
app.py — VisionControl Application Coordinator

Pipeline per frame:
    Camera → HandTracker → FeatureExtractor → GestureStateMachine
           → GestureMapper → DeviceController → HUD → Display

Phase 5 adds:
    GestureMapper → abstract action → DeviceController → registry + virtual devices
"""

import cv2
import sys
from typing import Optional

from config import Config
from vision import Camera, HandTracker
from gestures import (
    FeatureExtractor, GestureStateMachine, GestureMapper,
    GestureResult, GestureEvent, ControlMode, GestureType,
)
from devices import DeviceController, CommandBus, SerialTransport
from ui.hud import HUD


class VisionControlApp:
    def __init__(self):
        cfg = Config()
        self.config = cfg

        self.camera = Camera(
            camera_index=cfg.CAMERA_INDEX,
            target_fps=cfg.TARGET_FPS,
            width=cfg.FRAME_WIDTH,
            height=cfg.FRAME_HEIGHT,
        )
        self.hand_tracker = HandTracker(
            max_num_hands=cfg.MAX_NUM_HANDS,
            min_detection_confidence=cfg.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MIN_TRACKING_CONFIDENCE,
        )
        self.feature_extractor = FeatureExtractor(
            alpha=cfg.SMOOTHING_ALPHA,
            finger_extension_threshold=cfg.FINGER_EXTENSION_THRESHOLD,
        )
        self.gesture_sm = GestureStateMachine(
            confirm_frames=cfg.GESTURE_CONFIRM_FRAMES,
            cooldown_ms=cfg.GESTURE_COOLDOWN_MS,
            pinch_on_threshold=cfg.PINCH_ON_THRESHOLD,
            pinch_off_threshold=cfg.PINCH_OFF_THRESHOLD,
            swipe_history_size=cfg.SWIPE_HISTORY_SIZE,
            swipe_min_distance=cfg.SWIPE_MIN_DISTANCE,
            swipe_min_speed=cfg.SWIPE_MIN_SPEED,
            swipe_max_vertical_ratio=cfg.SWIPE_MAX_VERTICAL_RATIO,
            swipe_max_duration=cfg.SWIPE_MAX_DURATION,
            swipe_horizontal_dominance=cfg.SWIPE_HORIZONTAL_DOMINANCE,
        )
        self.gesture_mapper = GestureMapper()

        # Phase 6 — Serial Transport & Command Bus
        self.serial_transport = SerialTransport(
            port=cfg.SERIAL_PORT,
            baud_rate=cfg.SERIAL_BAUD_RATE,
            timeout=cfg.SERIAL_TIMEOUT,
            auto_detect=cfg.SERIAL_AUTO_DETECT,
            reconnect_interval=cfg.SERIAL_RECONNECT_INTERVAL,
            debug_serial=cfg.DEBUG_SERIAL,
        )
        self.command_bus = CommandBus(
            mode=cfg.DEVICE_MODE,
            transport=self.serial_transport,
        )

        # Phase 5 & 6 — device controller with command bus
        self.device_controller = DeviceController(
            command_bus=self.command_bus,
            mode=cfg.DEVICE_MODE,
        )

        self.hud = HUD()

        self.running = False
        # Persist last event for a short display window
        self._last_event: Optional[GestureEvent] = None
        self._last_event_clear_counter: int = 0
        self._EVENT_DISPLAY_FRAMES = 30

    def start(self):
        print("Initializing VisionControl...")
        try:
            if self.config.DEVICE_MODE == "HARDWARE" and self.config.SERIAL_ENABLED:
                print(f"[HARDWARE MODE] Connecting to Arduino...")
                self.serial_transport.connect()
            else:
                print("[VIRTUAL MODE] Hardware transport disabled.")

            self.camera.start()
            self.running = True
            self.run_loop()
        except RuntimeError as e:
            print(f"\n[FATAL ERROR] {e}")
            print("Please ensure your camera is connected and permitted in OS settings.\n")
            sys.exit(1)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"\n[UNEXPECTED ERROR] {e}\n")
            sys.exit(1)
        finally:
            self.cleanup()

    def run_loop(self):
        print(
            f"VisionControl active on Camera {self.config.CAMERA_INDEX}. "
            "Press 'q' or 'ESC' to quit."
        )
        while self.running:
            # 1. Capture ──────────────────────────────────────────────────────
            success, frame, fps = self.camera.get_frame()
            if not success:
                print("\n[WARNING] Failed to grab frame.\n")
                break

            # 2. Hand landmarks (Phase 2) ──────────────────────────────────────
            frame, landmarks_list = self.hand_tracker.process(frame)
            landmarks = landmarks_list[0] if landmarks_list else None

            # 3. Feature extraction (Phase 3) ──────────────────────────────────
            features = self.feature_extractor.extract(landmarks)

            # 4. Gesture state machine (Phase 4) ───────────────────────────────
            result, event = self.gesture_sm.update(features)

            # 5. Map gesture → action, update mode ────────────────────────────
            if event is not None:
                action = self.gesture_mapper.map_event(event)
                self._last_event = event
                self._last_event_clear_counter = self._EVENT_DISPLAY_FRAMES

                # 5b. Phase 5 & 6 — feed action into device controller ───────────
                cmd_result = self.device_controller.handle_action(action)
                # Sync control mode between gesture mapper and device controller
                self.device_controller.mode = self.gesture_mapper.mode

            # Count down event display window
            if self._last_event_clear_counter > 0:
                self._last_event_clear_counter -= 1
            else:
                self._last_event = None

            # 6. Render HUD ────────────────────────────────────────────────────
            frame = self.hud.render(
                frame,
                fps,
                features=features,
                result=result,
                event=self._last_event,
                mode=self.gesture_mapper.mode,
                debug_features=self.config.DEBUG_FEATURES,
                debug_gestures=self.config.DEBUG_GESTURES,
                # Phase 5 & 6
                registry=self.device_controller.registry,
                pending_action=self.device_controller.pending_action,
                debug_devices=self.config.DEBUG_DEVICES,
                command_history=self.device_controller.command_history(),
                device_mode=self.command_bus.mode,
                arduino_connected=self.serial_transport.is_connected(),
                arduino_unresponsive=self.serial_transport.is_unresponsive,
            )

            # 7. Display ───────────────────────────────────────────────────────
            cv2.imshow(self.config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                self.running = False

    def cleanup(self):
        print("Shutting down VisionControl...")
        self.camera.stop()
        self.hand_tracker.close()
        self.serial_transport.disconnect()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


if __name__ == "__main__":
    app = VisionControlApp()
    app.start()
