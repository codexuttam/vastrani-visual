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

## Phase 4 — Gesture Recognition State Machine

### Implemented

- **OPEN_PALM** — all 5 fingers extended + angle validation
- **FIST** — all fingers folded + curl angle validation
- **TWO_FINGERS** — index + middle extended, ring + pinky folded
- **PINCH** — thumb↔index normalized distance with hysteresis
- **SWIPE_LEFT / SWIPE_RIGHT** — palm movement history: distance, velocity, horizontal dominance, duration
- **Temporal confirmation** — gesture must hold for `GESTURE_CONFIRM_FRAMES` consecutive frames before activating
- **Debouncing** — gesture event fires exactly once per activation, not every frame
- **Pinch hysteresis** — `PINCH_ON_THRESHOLD` / `PINCH_OFF_THRESHOLD` prevent oscillation
- **Swipe anti-diagonal** — horizontal displacement must dominate vertical (`SWIPE_HORIZONTAL_DOMINANCE`)
- **Geometric confidence** — calculated from finger angles, state consistency; normalized to [0,1]
- **State machine** — NO_GESTURE → CANDIDATE → ACTIVE → COOLDOWN → NO_GESTURE
- **GestureMapper** — Gesture → Abstract Action (no device or Arduino coupling)
- **ControlMode** — IDLE / CONTROL tracked by mapper
- **EMERGENCY_STOP** — FIST produces this action and returns mode to IDLE
- **Gesture engine HUD panel** — Current, Event, Action, Confidence, Mode, State displayed live

### Static vs Event gestures

| Category | Gestures |
|---|---|
| **Static** (continuous) | OPEN_PALM, FIST, TWO_FINGERS, PINCH |
| **Event** (one-shot) | SWIPE_LEFT, SWIPE_RIGHT |

Static gestures represent *current hand state*. Event gestures represent *an action that happened* and fire once via cooldown.

### Gesture → Action mapping

| Gesture | Action |
|---|---|
| OPEN_PALM | ENTER_CONTROL |
| FIST | EMERGENCY_STOP |
| TWO_FINGERS | SELECT |
| PINCH | CONFIRM |
| SWIPE_LEFT | PREVIOUS |
| SWIPE_RIGHT | NEXT |

### Configurable thresholds

| Variable | Default | Description |
|---|---|---|
| `GESTURE_CONFIRM_FRAMES` | `5` | Frames needed to confirm a static gesture |
| `GESTURE_COOLDOWN_MS` | `500` | Cooldown (ms) after gesture fires |
| `PINCH_ON_THRESHOLD` | `0.22` | Normalized distance to activate pinch |
| `PINCH_OFF_THRESHOLD` | `0.28` | Normalized distance to release pinch |
| `SWIPE_HISTORY_SIZE` | `15` | Palm positions kept in rolling history |
| `SWIPE_MIN_DISTANCE` | `0.18` | Minimum horizontal displacement |
| `SWIPE_MIN_SPEED` | `0.6` | Minimum speed (units/s) |
| `SWIPE_MAX_VERTICAL_RATIO` | `0.6` | Max vertical/horizontal ratio |
| `SWIPE_MAX_DURATION` | `0.8` | Max swipe duration (seconds) |
| `SWIPE_HORIZONTAL_DOMINANCE` | `1.4` | dx must be N× greater than dy |
| `DEBUG_GESTURES` | `true` | Show gesture engine panel in HUD |



* **Phase 4** — Gesture state machine (OPEN_PALM, FIST, SWIPE, etc.)
* **Phase 5** — Virtual device control
* **Phase 6** — Arduino serial communication
* **Phase 7** — Face tracking
* **Phase 8** — AR emoji renderer
* **Phase 9** — OpenAI integration
* **Phase 10** — Natural-language intent
* **Phase 11** — Polished UI/HUD
* **Phase 12** — Testing, reliability, documentation
