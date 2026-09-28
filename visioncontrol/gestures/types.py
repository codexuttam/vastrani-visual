"""
gestures/types.py

Shared enums and dataclasses for Phase 4 gesture recognition.
No UI logic. No device control. No OpenAI.
"""

from dataclasses import dataclass, field
from enum import Enum
import time


class GestureType(Enum):
    """All recognized gesture types."""
    NONE        = "NONE"
    OPEN_PALM   = "OPEN_PALM"
    FIST        = "FIST"
    TWO_FINGERS = "TWO_FINGERS"
    PINCH       = "PINCH"
    SWIPE_LEFT  = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"


class GestureAction(Enum):
    """Abstract actions produced by the gesture mapper. Device-agnostic."""
    NONE            = "NONE"
    ENTER_CONTROL   = "ENTER_CONTROL"
    EMERGENCY_STOP  = "EMERGENCY_STOP"
    SELECT          = "SELECT"
    CONFIRM         = "CONFIRM"
    PREVIOUS        = "PREVIOUS"
    NEXT            = "NEXT"


class StateMachineState(Enum):
    """Internal states of the gesture state machine."""
    NO_GESTURE = "NO_GESTURE"
    CANDIDATE  = "CANDIDATE"
    ACTIVE     = "ACTIVE"
    COOLDOWN   = "COOLDOWN"


class ControlMode(Enum):
    """High-level application control mode."""
    IDLE    = "IDLE"
    CONTROL = "CONTROL"


@dataclass
class GestureResult:
    """
    Current gesture state produced by the state machine each frame.
    Represents ongoing state — not a one-shot event.
    """
    gesture:       GestureType       = GestureType.NONE
    confidence:    float             = 0.0
    active:        bool              = False
    state:         StateMachineState = StateMachineState.NO_GESTURE
    timestamp:     float             = field(default_factory=time.time)


@dataclass
class GestureEvent:
    """
    A one-shot event fired exactly once when a gesture transitions to ACTIVE.
    Distinct from GestureResult which is updated every frame.
    """
    gesture:    GestureType   = GestureType.NONE
    action:     GestureAction = GestureAction.NONE
    confidence: float         = 0.0
    timestamp:  float         = field(default_factory=time.time)
