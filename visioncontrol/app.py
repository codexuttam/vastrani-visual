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
from vision import Camera, HandTracker, FaceTracker, FaceFeatureExtractor
from gestures import (
    FeatureExtractor, GestureStateMachine, GestureMapper,
    GestureResult, GestureEvent, ControlMode, GestureType,
)
from devices import DeviceController, CommandBus, SerialTransport
from ar import ARController, ARRenderer
from ai import AIRouter
from intent import IntentEngine, IntentService
from intent.api import IntentAPIServer
from ui.hud import HUD
from ui.contracts import HUDMode, NotificationCategory
from health import StartupDiagnostics, HealthMonitor
from version import __version__

import queue
import threading

INTENT_DEMO_COMMANDS = [
    "turn on the bedroom fan",
    "could you please make the lights brighter",
    "turn on the fan and lower the brightness",
    "turn it off",
    "turn off everything",
    "do that",
    "open spotify",
]


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

        # Phase 7 — Face Tracker & Face Feature Extractor
        self.face_enabled = cfg.FACE_ENABLED
        if self.face_enabled:
            self.face_tracker = FaceTracker(
                max_num_faces=cfg.FACE_MAX_DETECTIONS,
                min_detection_confidence=cfg.MIN_FACE_DETECTION_CONFIDENCE,
                min_tracking_confidence=cfg.MIN_FACE_TRACKING_CONFIDENCE,
            )
            self.face_feature_extractor = FaceFeatureExtractor(
                smoothing_alpha=cfg.FACE_SMOOTHING_ALPHA,
                lost_timeout_ms=cfg.FACE_LOST_TIMEOUT_MS,
            )
        else:
            self.face_tracker = None
            self.face_feature_extractor = None

        # Phase 8 — AR Controller & AR Renderer
        self.ar_enabled = cfg.AR_ENABLED
        self.ar_controller = ARController()
        self.ar_renderer = ARRenderer(
            offset_x=cfg.EMOJI_OFFSET_X,
            offset_y=cfg.EMOJI_OFFSET_Y,
            scale_multiplier=cfg.EMOJI_SCALE_MULTIPLIER,
            min_scale=cfg.EMOJI_MIN_SCALE,
            max_scale=cfg.EMOJI_MAX_SCALE,
            smoothing_alpha=cfg.AR_SMOOTHING_ALPHA,
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

        # Phase 9 — OpenAI Event Router
        self.ai_router = AIRouter(
            ai_mode=cfg.AI_MODE,
            min_confidence=cfg.AI_MIN_CONFIDENCE,
        )

        hud_mode_val = HUDMode.DEVELOPER if cfg.HUD_MODE == "DEVELOPER" else HUDMode.STANDARD
        self.hud = HUD(mode=hud_mode_val, reduced_motion=cfg.REDUCED_MOTION)

        # Phase 10 — Natural-Language Intent Engine (routes through the
        # existing DeviceController / ARController; never executes directly)
        self.intent_engine: Optional[IntentEngine] = None
        self.intent_service: Optional[IntentService] = None
        self.intent_api: Optional[IntentAPIServer] = None
        self._intent_queue: "queue.Queue[str]" = queue.Queue()
        self._intent_demo_idx = 0
        if cfg.INTENT_ENGINE_ENABLED:
            self.intent_engine = IntentEngine.from_config(
                cfg, device_controller=self.device_controller, ar_controller=self.ar_controller
            )
            self.intent_service = IntentService(self.intent_engine)

        self.running = False
        # Persist last event for a short display window
        self._last_event: Optional[GestureEvent] = None
        self._last_event_clear_counter: int = 0
        self._EVENT_DISPLAY_FRAMES = 30
        self._test_prompt_idx = 0

    def start(self):
        StartupDiagnostics.run_checks(self.config)
        try:
            if self.config.DEVICE_MODE == "HARDWARE" and self.config.SERIAL_ENABLED:
                print(f"[HARDWARE MODE] Connecting to Arduino...")
                self.serial_transport.connect()
            else:
                print("[VIRTUAL MODE] Hardware transport disabled.")

            self.camera.start()
            self.running = True
            self._start_intent_services()
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

    # ─── Phase 10: intent engine plumbing ───────────────────────────────────────

    def _start_intent_services(self):
        """Background worker (keeps the camera loop non-blocking), console, optional API."""
        if self.intent_engine is None:
            return
        threading.Thread(target=self._intent_worker, daemon=True).start()
        if self.config.INTENT_CONSOLE_ENABLED and sys.stdin and sys.stdin.isatty():
            threading.Thread(target=self._intent_console, daemon=True).start()
            print("[INTENT] Type a command in this terminal (e.g. 'turn on the fan'); 'yes'/'no' to confirm.")
        if self.config.INTENT_API_ENABLED:
            try:
                self.intent_api = IntentAPIServer(
                    self.intent_service, self.config.INTENT_API_HOST, self.config.INTENT_API_PORT
                ).start()
                print(f"[INTENT] API listening on http://{self.config.INTENT_API_HOST}:"
                      f"{self.config.INTENT_API_PORT}/api/intent/parse")
            except OSError as e:
                print(f"[INTENT] API disabled: {e}")

    def _intent_console(self):
        while self.running:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                return
            if line.strip():
                self._intent_queue.put(line)

    def _intent_worker(self):
        while self.running:
            try:
                text = self._intent_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if text == "__confirm__":
                result = self.intent_engine.confirm()
            elif text == "__cancel__":
                result = self.intent_engine.cancel()
            else:
                print(f"\n[INTENT] > {text}")
                result = self.intent_engine.handle(text)
            print(f"[INTENT] {result.status}: {result.message}")

    def run_loop(self):
        print(
            f"VisionControl active on Camera {self.config.CAMERA_INDEX}. "
            "Press 'q' or 'ESC' to quit. Press 't' to test AI semantic command. "
            "Press 'i' for an intent-engine demo command, 'y'/'n' to confirm/cancel."
        )
        while self.running:
            # 1. Capture ──────────────────────────────────────────────────────
            success, frame, fps = self.camera.get_frame()
            if not success:
                print("\n[WARNING] Failed to grab frame.\n")
                break

            # Poll async completed AI results
            self.ai_router.check_completed_results()

            # 2. Hand landmarks (Phase 2) ──────────────────────────────────────
            frame, landmarks_list = self.hand_tracker.process(frame)
            landmarks = landmarks_list[0] if landmarks_list else None

            # 2b. Face landmarks (Phase 7) ─────────────────────────────────────
            face_state = None
            face_infer_ms = 0.0
            if self.face_enabled and self.face_tracker is not None:
                frame, primary_face_lms, _ = self.face_tracker.process(
                    frame, debug_face=self.config.DEBUG_FACE
                )
                face_state = self.face_feature_extractor.extract(primary_face_lms)
                face_infer_ms = self.face_tracker.inference_ms

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

                # 5c. Phase 8 — feed action into AR controller ───────────────────
                if self.ar_enabled:
                    self.ar_controller.handle_action(action)

            # Count down event display window
            if self._last_event_clear_counter > 0:
                self._last_event_clear_counter -= 1
            else:
                self._last_event = None

            # 6. Render Phase 8 AR Overlay ─────────────────────────────────────
            if self.ar_enabled:
                anchor = face_state.anchor if (face_state and face_state.detected) else None
                frame = self.ar_renderer.render(frame, anchor, self.ar_controller.state)

            # 7. Render HUD ────────────────────────────────────────────────────
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
                # Phase 7
                face_state=face_state,
                debug_face=self.config.DEBUG_FACE,
                face_inference_ms=face_infer_ms,
                # Phase 8
                ar_state=self.ar_controller.state,
                debug_ar=self.config.DEBUG_AR,
                # Phase 9
                ai_status=self.ai_router.status,
                ai_mode=self.ai_router.ai_mode,
                last_ai_record=self.ai_router.last_record,
                debug_ai=self.config.DEBUG_AI,
                # Phase 10
                intent_result=self.intent_engine.last_result if self.intent_engine else None,
                intent_pending=self.intent_engine.has_pending if self.intent_engine else False,
                intent_provider=self.intent_engine.provider_name if self.intent_engine else "off",
                debug_intent=self.config.DEBUG_INTENT and self.intent_engine is not None,
            )

            # 8. Display ───────────────────────────────────────────────────────
            cv2.imshow(self.config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.running = False
            elif key in (13, ord("y"), ord("Y")) and self.intent_engine is not None and self.intent_engine.has_pending:
                self._intent_queue.put("__confirm__")
            elif key in (27, ord("n"), ord("N")) and self.intent_engine is not None and self.intent_engine.has_pending:
                self._intent_queue.put("__cancel__")
            elif key == 27:
                self.running = False
            elif key == ord("t"):
                test_prompts = [
                    "Turn the living room light on",
                    "Set the living room light to 50 percent",
                    "Set the ceiling fan to speed 2",
                    "Turn on the bedroom AC",
                    "Make rocket fly",
                ]
                prompt = test_prompts[self._test_prompt_idx % len(test_prompts)]
                self._test_prompt_idx += 1
                print(f"\n[AI TEST INPUT] Natural language command: '{prompt}'")
                self.ai_router.process_text_command_async(
                    prompt, self.device_controller.registry, self.device_controller
                )
            elif key == ord("i") and self.intent_engine is not None:
                cmd = INTENT_DEMO_COMMANDS[self._intent_demo_idx % len(INTENT_DEMO_COMMANDS)]
                self._intent_demo_idx += 1
                self._intent_queue.put(cmd)
            else:
                self.hud.handle_key_event(key)

    def cleanup(self):
        print("Shutting down VisionControl...")
        self.running = False
        if self.intent_api is not None:
            self.intent_api.stop()
        if self.intent_engine is not None:
            self.intent_engine.shutdown()
        self.ai_router.stop()
        self.camera.stop()
        self.hand_tracker.close()
        if self.face_tracker:
            self.face_tracker.close()
        self.serial_transport.disconnect()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


if __name__ == "__main__":
    app = VisionControlApp()
    app.start()

