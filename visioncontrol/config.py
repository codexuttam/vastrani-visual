import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # ── Camera ────────────────────────────────────────────────────────────────
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
    FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH",  1280))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 720))
    TARGET_FPS   = int(os.getenv("TARGET_FPS",   30))
    WINDOW_NAME  = os.getenv("WINDOW_NAME", "VisionControl")

    # ── Hand tracker ──────────────────────────────────────────────────────────
    MAX_NUM_HANDS            = int(os.getenv("MAX_NUM_HANDS", 1))
    MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", 0.7))
    MIN_TRACKING_CONFIDENCE  = float(os.getenv("MIN_TRACKING_CONFIDENCE",  0.6))

    # ── Feature extraction ────────────────────────────────────────────────────
    SMOOTHING_ALPHA              = float(os.getenv("SMOOTHING_ALPHA",              0.35))
    FINGER_EXTENSION_THRESHOLD   = float(os.getenv("FINGER_EXTENSION_THRESHOLD",   0.5))

    # ── Gesture state machine ─────────────────────────────────────────────────
    GESTURE_CONFIRM_FRAMES       = int(os.getenv("GESTURE_CONFIRM_FRAMES",         5))
    GESTURE_COOLDOWN_MS          = float(os.getenv("GESTURE_COOLDOWN_MS",          500.0))

    # Pinch hysteresis
    PINCH_ON_THRESHOLD           = float(os.getenv("PINCH_ON_THRESHOLD",           0.22))
    PINCH_OFF_THRESHOLD          = float(os.getenv("PINCH_OFF_THRESHOLD",          0.28))

    # Swipe detection
    SWIPE_HISTORY_SIZE           = int(os.getenv("SWIPE_HISTORY_SIZE",             15))
    SWIPE_MIN_DISTANCE           = float(os.getenv("SWIPE_MIN_DISTANCE",           0.18))
    SWIPE_MIN_SPEED              = float(os.getenv("SWIPE_MIN_SPEED",              0.6))
    SWIPE_MAX_VERTICAL_RATIO     = float(os.getenv("SWIPE_MAX_VERTICAL_RATIO",     0.6))
    SWIPE_MAX_DURATION           = float(os.getenv("SWIPE_MAX_DURATION",           0.8))
    SWIPE_HORIZONTAL_DOMINANCE   = float(os.getenv("SWIPE_HORIZONTAL_DOMINANCE",   1.4))

    # ── Debug flags ───────────────────────────────────────────────────────────
    DEBUG_FEATURES  = os.getenv("DEBUG_FEATURES",  "true").lower()  == "true"
    DEBUG_GESTURES  = os.getenv("DEBUG_GESTURES",  "true").lower()  == "true"

    # ── Phase 5: Device control ───────────────────────────────────────────────
    COMMAND_HISTORY_SIZE = int(os.getenv("COMMAND_HISTORY_SIZE", 50))
    DEBUG_DEVICES        = os.getenv("DEBUG_DEVICES", "true").lower() == "true"

    # ── Phase 6: Serial Hardware Integration ──────────────────────────────────
    SERIAL_ENABLED            = os.getenv("SERIAL_ENABLED", "true").lower() == "true"
    SERIAL_PORT               = os.getenv("SERIAL_PORT", "")
    SERIAL_BAUD_RATE          = int(os.getenv("SERIAL_BAUD_RATE", 115200))
    SERIAL_TIMEOUT            = float(os.getenv("SERIAL_TIMEOUT", 1.0))
    SERIAL_RECONNECT_INTERVAL = float(os.getenv("SERIAL_RECONNECT_INTERVAL", 3.0))
    SERIAL_AUTO_DETECT        = os.getenv("SERIAL_AUTO_DETECT", "true").lower() == "true"
    DEVICE_MODE               = os.getenv("DEVICE_MODE", "VIRTUAL").upper()
    DEBUG_SERIAL              = os.getenv("DEBUG_SERIAL", "true").lower() == "true"

    # ── Phase 7: Face Tracking ────────────────────────────────────────────────
    FACE_ENABLED                  = os.getenv("FACE_ENABLED", "true").lower() == "true"
    DEBUG_FACE                    = os.getenv("DEBUG_FACE", "true").lower() == "true"
    FACE_SMOOTHING_ALPHA          = float(os.getenv("FACE_SMOOTHING_ALPHA", 0.35))
    FACE_LOST_TIMEOUT_MS          = float(os.getenv("FACE_LOST_TIMEOUT_MS", 500.0))
    FACE_MAX_DETECTIONS           = int(os.getenv("FACE_MAX_DETECTIONS", 2))
    MIN_FACE_DETECTION_CONFIDENCE = float(os.getenv("MIN_FACE_DETECTION_CONFIDENCE", 0.5))
    MIN_FACE_TRACKING_CONFIDENCE  = float(os.getenv("MIN_FACE_TRACKING_CONFIDENCE", 0.5))

    # ── Phase 8: AR Emoji & Face Effect Engine ────────────────────────────────
    AR_ENABLED              = os.getenv("AR_ENABLED", "true").lower() == "true"
    DEBUG_AR                = os.getenv("DEBUG_AR", "true").lower() == "true"
    AR_SMOOTHING_ALPHA      = float(os.getenv("AR_SMOOTHING_ALPHA", 0.4))
    EMOJI_SCALE_MULTIPLIER  = float(os.getenv("EMOJI_SCALE_MULTIPLIER", 1.5))
    EMOJI_MIN_SCALE         = float(os.getenv("EMOJI_MIN_SCALE", 0.1))
    EMOJI_MAX_SCALE         = float(os.getenv("EMOJI_MAX_SCALE", 2.0))
    EMOJI_OFFSET_X          = float(os.getenv("EMOJI_OFFSET_X", 0.0))
    EMOJI_OFFSET_Y          = float(os.getenv("EMOJI_OFFSET_Y", -0.05))

    # ── Phase 9: OpenAI Intelligence Layer ───────────────────────────────────
    OPENAI_API_KEY          = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL            = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_TIMEOUT          = float(os.getenv("OPENAI_TIMEOUT", 10.0))
    OPENAI_ENABLED          = os.getenv("OPENAI_ENABLED", "true").lower() == "true"
    AI_MODE                 = os.getenv("AI_MODE", "EVENT").upper()
    AI_MIN_CONFIDENCE       = float(os.getenv("AI_MIN_CONFIDENCE", 0.75))
    AI_MIN_REQUEST_INTERVAL = float(os.getenv("AI_MIN_REQUEST_INTERVAL", 1.0))
    DEBUG_AI                = os.getenv("DEBUG_AI", "true").lower() == "true"
