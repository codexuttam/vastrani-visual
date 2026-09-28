import os
from config import Config

def test_config_defaults():
    config = Config()
    
    # Assert sensible defaults
    assert config.CAMERA_INDEX == int(os.getenv("CAMERA_INDEX", 0))
    assert config.TARGET_FPS == 30
    assert config.FRAME_WIDTH == 1280
    assert config.FRAME_HEIGHT == 720
    assert config.WINDOW_NAME == "VisionControl"
