from vision.camera import Camera

def test_camera_initialization():
    camera = Camera(camera_index=1, target_fps=60, width=1920, height=1080)
    
    assert camera.camera_index == 1
    assert camera.target_fps == 60
    assert camera.width == 1920
    assert camera.height == 1080
    assert camera.cap is None
