import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Camera settings
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 1280))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 720))
    TARGET_FPS = int(os.getenv("TARGET_FPS", 30))

    # UI settings
    WINDOW_NAME = os.getenv("WINDOW_NAME", "VisionControl")

    # Hand tracker settings
    MAX_NUM_HANDS = int(os.getenv("MAX_NUM_HANDS", 1))
    MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", 0.7))
    MIN_TRACKING_CONFIDENCE = float(os.getenv("MIN_TRACKING_CONFIDENCE", 0.6))

    # Feature extraction settings
    SMOOTHING_ALPHA = float(os.getenv("SMOOTHING_ALPHA", 0.35))
    FINGER_EXTENSION_THRESHOLD = float(os.getenv("FINGER_EXTENSION_THRESHOLD", 0.5))
    DEBUG_FEATURES = os.getenv("DEBUG_FEATURES", "true").lower() == "true"

    # OpenAI Settings (for later phases)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
