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
    
    # OpenAI Settings (for later phases)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
