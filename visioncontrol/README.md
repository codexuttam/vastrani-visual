# VisionControl

AI-powered touchless human-computer interface.

---

## Phase 1 — Project Bootstrap & Webcam

* Project structure created
* Centralized configuration (`config.py`)
* Webcam capture with mirrored selfie view
* FPS display
* VisionControl HUD (dark/cyan style)
* Clean shutdown on Q/ESC

---

## Phase 2 — Hand Landmark Detection

* MediaPipe Hands integration (`vision/hand_tracker.py`)
* 21-point hand landmark detection
* Landmark skeleton drawn on webcam feed
* Modular `HandTracker` class — no UI logic inside

---

## Phase 3 — Hand Feature Extraction & Smoothing

### What was implemented

* **Normalized coordinates** — all x/y/z in [0.0, 1.0] frame space
* **Palm center** — averaged from wrist + 4 MCP landmarks
* **Finger extension detection** — geometric tip/pip/angle heuristic for each finger
* **Finger joint angles** — calculated at PIP joints using `calculate_angle(a, b, c)`
* **Pinch distance** — thumb tip ↔ index tip, normalized by palm size
* **Hand orientation** — angle in degrees from wrist→middle MCP
* **Hand movement** — dx, dy, speed, direction (LEFT/RIGHT/UP/DOWN/STATIONARY)
* **EMA smoothing** — configurable alpha (`SMOOTHING_ALPHA`), applied to all features
* **Debug diagnostics panel** — rendered in top-right corner when `DEBUG_FEATURES=true`

### Feature Data Structure

```python
@dataclass
class HandFeatureSet:
    valid: bool              # False when no hand detected

    palm_x: float            # normalized [0,1]
    palm_y: float
    palm_z: float

    thumb_extended: bool
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool

    thumb_angle: float       # degrees
    index_angle: float
    middle_angle: float
    ring_angle: float
    pinky_angle: float

    thumb_index_distance: float   # normalized by palm size
    palm_size: float

    hand_angle: float        # wrist→middle MCP direction in degrees

    dx: float                # smoothed movement delta
    dy: float
    speed: float
    direction: str           # LEFT / RIGHT / UP / DOWN / STATIONARY
```

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python app.py
```

Controls: **Q** or **ESC** to quit.

## Running Tests

```bash
pytest tests/ -v
```

## Configuration (`.env`)

Copy `.env.example` to `.env` to override defaults:

| Variable                    | Default       | Description                          |
|-----------------------------|---------------|--------------------------------------|
| `CAMERA_INDEX`              | `0`           | Webcam index                         |
| `FRAME_WIDTH`               | `1280`        | Capture width                        |
| `FRAME_HEIGHT`              | `720`         | Capture height                       |
| `TARGET_FPS`                | `30`          | Target render FPS                    |
| `SMOOTHING_ALPHA`           | `0.35`        | EMA alpha (0→smooth, 1→raw)          |
| `FINGER_EXTENSION_THRESHOLD`| `0.5`         | Angle fraction for extension detect  |
| `DEBUG_FEATURES`            | `true`        | Show feature diagnostics panel       |

---

## Future Phases

* **Phase 4** — Gesture state machine (OPEN_PALM, FIST, SWIPE, etc.)
* **Phase 5** — Virtual device control
* **Phase 6** — Arduino serial communication
* **Phase 7** — Face tracking
* **Phase 8** — AR emoji renderer
* **Phase 9** — OpenAI integration
* **Phase 10** — Natural-language intent
* **Phase 11** — Polished UI/HUD
* **Phase 12** — Testing, reliability, documentation
