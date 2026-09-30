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
* **Phase 5** — Virtual device control (DONE)
* **Phase 6** — Arduino serial hardware integration (DONE)
* **Phase 7** — Face tracking (DONE)
* **Phase 8** — AR emoji renderer (DONE)
* **Phase 9** — OpenAI integration
* **Phase 10** — Natural-language intent
* **Phase 11** — Polished UI/HUD
* **Phase 12** — Testing, reliability, documentation

---

## Phase 8 — AR Emoji & Face Effects

### Implemented:

- AR renderer (`ar/renderer.py`)
- face-anchor compositing (`ar/compositor.py`)
- emoji asset manager (`ar/assets.py`)
- alpha blending (`ARCompositor.blend_overlay`)
- scaling (`EMOJI_SCALE_MULTIPLIER`)
- rotation (`rotate_overlay`)
- AR smoothing (`AR_SMOOTHING_ALPHA`)
- face-loss handling (graceful hide on invalid anchor)
- emoji navigation (`SWIPE_LEFT`, `SWIPE_RIGHT`)
- emoji selection (`TWO_FINGERS`)
- AR toggle (`PINCH`)
- gesture-controlled AR effects (`ar/effects.py`)
- AR HUD panel & top Emoji Rail (`ui/hud.py`)

### Available Effects:

- `happy` (😀)
- `laughing` (😂)
- `cool` (😎)
- `angry` (😡)
- `love` (😍)
- `thinking` (🤔)

### Gesture Controls for AR:

| Gesture | Abstract Action | AR Engine Action |
|---|---|---|
| `SWIPE_RIGHT` | `NEXT` | Navigate to next emoji effect |
| `SWIPE_LEFT` | `PREVIOUS` | Navigate to previous emoji effect |
| `TWO_FINGERS` | `SELECT` | Select / Activate highlighted emoji |
| `PINCH` | `CONFIRM` | Toggle AR effect ON / OFF |
| `FIST` | `EMERGENCY_STOP` | Emergency Stop (hides AR effect & safe devices) |

### Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `AR_ENABLED` | `true` | Enable or disable AR emoji overlay rendering |
| `DEBUG_AR` | `true` | Show AR effect engine status panel in HUD |
| `AR_SMOOTHING_ALPHA` | `0.4` | Smoothing alpha for AR emoji motion |
| `EMOJI_SCALE_MULTIPLIER` | `1.5` | Scale multiplier relative to face region |
| `EMOJI_MIN_SCALE` | `0.1` | Minimum allowed emoji scale relative to frame |
| `EMOJI_MAX_SCALE` | `2.0` | Maximum allowed emoji scale relative to frame |

---

## Phase 7 — Face Tracking

### Implemented:

- local face landmark detection (`vision/face_tracker.py`)
- normalized face coordinates (`[0.0, 1.0]`)
- face center (`center_x`, `center_y`)
- face width/height (`width`, `height`)
- face scale (`scale`)
- yaw, pitch, roll head orientation estimation
- face movement tracking (`dx`, `dy`, `speed`)
- temporal smoothing (`ExponentialSmoother`, `PointSmoother`, `AngleSmoother`)
- face loss handling & grace period (`FACE_LOST_TIMEOUT_MS`)
- primary face selection (largest face region)
- face anchor (`FaceAnchor` model for Phase 8 integration)
- face debug visualization & anchor crosshair rendering

### Face Data Contract (`FaceState`)

```python
@dataclass
class FaceState:
    detected: bool
    center_x: float
    center_y: float
    width: float
    height: float
    scale: float
    yaw: float
    pitch: float
    roll: float
    dx: float
    dy: float
    speed: float
    timestamp: float

    @property
    def anchor(self) -> FaceAnchor: ...
```

### Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `FACE_ENABLED` | `true` | Enable or disable real-time face tracking module |
| `DEBUG_FACE` | `true` | Show facial landmarks, face anchor crosshair, and HUD panel |
| `FACE_SMOOTHING_ALPHA` | `0.35` | Exponential smoothing alpha for face tracking parameters |
| `FACE_LOST_TIMEOUT_MS` | `500.0` | Grace period (ms) before marking face lost after detection fails |
| `FACE_MAX_DETECTIONS` | `2` | Maximum faces detected before primary face selection |

---

## Phase 5 — Virtual Device Control

### Implemented

- Device registry (`devices/registry.py`) with circular selection
- Virtual devices (`devices/virtual_device.py`) — deterministic state abstraction
- Device models (`devices/models.py`) — `DeviceState`, `CommandResult`, `CommandRecord`
- Device validator (`devices/validator.py`) — rejects unknown devices/actions/levels
- Device controller (`devices/controller.py`) — consumes abstract `GestureAction`s
- Power control — ON / OFF / TOGGLE
- Level control — `SET_LEVEL` with per-type range enforcement
- Emergency stop — all devices to safe state instantly
- Command history — in-memory ring buffer (configurable, default 50 entries)
- Device HUD panel — current device, power, level, pending action, recent commands
- Device navigation rail — bottom bar showing all devices, selected highlighted
- Full gesture → device integration (OPEN_PALM, SWIPE, TWO_FINGERS, PINCH, FIST)
- Developer test mode (`python -m devices.dev_mode`) — no webcam required

### Virtual Devices

| ID        | Name               | Type  |
|-----------|--------------------|-------|
| LIGHT_01  | Living Room Light  | LIGHT |
| FAN_01    | Ceiling Fan        | FAN   |
| MUSIC_01  | Music Player       | MUSIC |
| SERVO_01  | Servo Motor        | SERVO |

---

## Phase 6 — Arduino Hardware Integration

### Implemented:

- `pyserial` integration
- serial transport (`devices/serial_transport.py`)
- command bus (`devices/command_bus.py`)
- hardware mode (`DEVICE_MODE=HARDWARE`)
- virtual mode (`DEVICE_MODE=VIRTUAL`)
- serial protocol (`COMMAND|DEVICE_ID|VALUE\n`)
- Arduino handshake (`PING|SYSTEM` → `PONG|SYSTEM`)
- command acknowledgements (`OK|...` / `ERR|...`)
- reconnect handling (auto-reconnect without freezing camera loop)
- emergency stop (`ALL_OFF|SYSTEM`)
- Arduino firmware (`arduino/visioncontrol.ino`)
- hardware status HUD (displays mode & connection state)
- hardware command logging (records TX, RX, ACK, and virtual vs hardware status)

### Wiring Section

| Device ID | Arduino Pin | Pin Type | Expected Hardware / Behavior |
|-----------|-------------|----------|------------------------------|
| `LIGHT_01`| Pin 9       | PWM      | LED indicator or PWM dimmer circuit |
| `FAN_01`  | Pin 8       | Digital  | Safe low-voltage motor driver / Relay module |
| `SERVO_01`| Pin 10      | Servo PWM| SG90 or MG996R Servo motor (0° to 180°) |
| `MUSIC_01`| Pin 11      | Digital  | LED status indicator |

> **SAFETY NOTE**: Do NOT connect Arduino GPIO pins directly to mains AC equipment! Always use isolated, safe low-voltage prototype components or relays.

### Configuration (`.env`)

| Variable                   | Default   | Description |
|----------------------------|-----------|-------------|
| `DEVICE_MODE`              | `VIRTUAL` | `VIRTUAL` (simulated only) or `HARDWARE` (send to Arduino) |
| `SERIAL_ENABLED`           | `true`    | Enable or disable serial module |
| `SERIAL_PORT`              | `""`      | Serial port (auto-detects if blank) |
| `SERIAL_BAUD_RATE`         | `115200`  | Serial baud rate |
| `SERIAL_TIMEOUT`           | `1.0`     | Serial read/write timeout (seconds) |
| `SERIAL_RECONNECT_INTERVAL`| `3.0`     | Reconnection retry interval (seconds) |
| `SERIAL_AUTO_DETECT`       | `true`    | Auto-detect USB serial port on macOS, Linux, or Windows |
| `DEBUG_SERIAL`             | `true`    | Print serial TX/RX debug lines to console |

