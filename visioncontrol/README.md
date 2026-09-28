# VisionControl

AI-powered touchless human-computer interface.

## Phase 1
Phase 1 implements the foundational structure of the project and a working webcam application with a basic HUD. It does not include hand tracking, gesture recognition, AI integration, or hardware control.

### Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (optional):
   Copy `.env.example` to `.env` and modify as needed (e.g. changing `CAMERA_INDEX`).

### How to Run

1. Ensure your virtual environment is active.
2. Run the application:
   ```bash
   python app.py
   ```

### Controls
- **Q** or **ESC**: Quit the application safely.

### Expected Result
- A webcam window opens with a dark/cyan HUD displaying "VISIONCONTROL", a "CAMERA LIVE" indicator, your current FPS, and "Gesture interface initializing..." in the center.
- The image should be mirrored (selfie view).
- If the camera fails to open (e.g. due to macOS permissions), a clear error message is printed to the terminal without crashing.

### Future Phases
Subsequent phases will introduce hand landmark detection, gesture mapping, a command state machine, Arduino serial communication, AR emoji rendering, and OpenAI integration.
