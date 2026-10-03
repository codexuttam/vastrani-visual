"""
intent/schema.py

Phase 10: Single source of truth for the Natural-Language Intent Engine.

Everything the engine is allowed to express lives here:
    - supported intents
    - supported actions per intent
    - required entities per (intent, action)
    - supported entity names
    - canonical device vocabulary
    - actions that are explicitly outside the permission scope
    - the StructuredCommand data contract (+ strict dict parsing)

No other module should hard-code intent or action names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


# ─── Intents ──────────────────────────────────────────────────────────────────

class IntentType(str, Enum):
    DEVICE_CONTROL  = "device_control"
    DISPLAY_CONTROL = "display_control"
    MEDIA_CONTROL   = "media_control"
    NAVIGATION      = "navigation"
    GESTURE_COMMAND = "gesture_command"
    FACE_COMMAND    = "face_command"
    SYSTEM_COMMAND  = "system_command"
    MULTI_ACTION    = "multi_action"
    UNKNOWN         = "unknown"


SUPPORTED_INTENTS: FrozenSet[str] = frozenset(i.value for i in IntentType)


# ─── Actions ──────────────────────────────────────────────────────────────────

class Action(str, Enum):
    # device_control
    TURN_ON         = "turn_on"
    TURN_OFF        = "turn_off"
    TOGGLE          = "toggle"
    SET_LEVEL       = "set_level"
    INCREASE_LEVEL  = "increase_level"
    DECREASE_LEVEL  = "decrease_level"
    # display_control
    INCREASE_BRIGHTNESS = "increase_brightness"
    DECREASE_BRIGHTNESS = "decrease_brightness"
    SET_BRIGHTNESS      = "set_brightness"
    # media_control
    PLAY            = "play"
    PAUSE           = "pause"
    STOP            = "stop"
    NEXT_TRACK      = "next_track"
    PREVIOUS_TRACK  = "previous_track"
    VOLUME_UP       = "volume_up"
    VOLUME_DOWN     = "volume_down"
    SET_VOLUME      = "set_volume"
    # navigation
    NEXT            = "next"
    PREVIOUS        = "previous"
    SELECT          = "select"
    # gesture_command
    ENTER_CONTROL   = "enter_control"
    CONFIRM         = "confirm"
    # face_command
    SHOW_EFFECT     = "show_effect"
    HIDE_EFFECT     = "hide_effect"
    NEXT_EFFECT     = "next_effect"
    PREVIOUS_EFFECT = "previous_effect"
    # system_command
    EMERGENCY_STOP  = "emergency_stop"
    STATUS          = "status"


A = Action
INTENT_ACTIONS: Dict[str, FrozenSet[str]] = {
    IntentType.DEVICE_CONTROL.value: frozenset(a.value for a in (
        A.TURN_ON, A.TURN_OFF, A.TOGGLE, A.SET_LEVEL, A.INCREASE_LEVEL, A.DECREASE_LEVEL)),
    IntentType.DISPLAY_CONTROL.value: frozenset(a.value for a in (
        A.INCREASE_BRIGHTNESS, A.DECREASE_BRIGHTNESS, A.SET_BRIGHTNESS)),
    IntentType.MEDIA_CONTROL.value: frozenset(a.value for a in (
        A.PLAY, A.PAUSE, A.STOP, A.NEXT_TRACK, A.PREVIOUS_TRACK,
        A.VOLUME_UP, A.VOLUME_DOWN, A.SET_VOLUME)),
    IntentType.NAVIGATION.value: frozenset(a.value for a in (A.NEXT, A.PREVIOUS, A.SELECT)),
    IntentType.GESTURE_COMMAND.value: frozenset(a.value for a in (
        A.ENTER_CONTROL, A.SELECT, A.CONFIRM)),
    IntentType.FACE_COMMAND.value: frozenset(a.value for a in (
        A.SHOW_EFFECT, A.HIDE_EFFECT, A.NEXT_EFFECT, A.PREVIOUS_EFFECT)),
    IntentType.SYSTEM_COMMAND.value: frozenset(a.value for a in (A.EMERGENCY_STOP, A.STATUS)),
    IntentType.MULTI_ACTION.value: frozenset(),
    IntentType.UNKNOWN.value: frozenset(),
}


# ─── Required entities / parameters ───────────────────────────────────────────

_DC = IntentType.DEVICE_CONTROL.value
REQUIRED_ENTITIES: Dict[Tuple[str, str], Tuple[str, ...]] = {
    (_DC, A.TURN_ON.value):        ("device",),
    (_DC, A.TURN_OFF.value):       ("device",),
    (_DC, A.TOGGLE.value):         ("device",),
    (_DC, A.SET_LEVEL.value):      ("device", "value"),
    (_DC, A.INCREASE_LEVEL.value): ("device",),
    (_DC, A.DECREASE_LEVEL.value): ("device",),
    (IntentType.DISPLAY_CONTROL.value, A.SET_BRIGHTNESS.value): ("value",),
    (IntentType.MEDIA_CONTROL.value, A.SET_VOLUME.value):       ("value",),
}

SUPPORTED_ENTITIES: FrozenSet[str] = frozenset({
    "device", "location", "target", "value", "amount", "direction",
    "duration", "application", "media", "person", "gesture",
})

NUMERIC_ENTITIES: FrozenSet[str] = frozenset({"value", "amount", "duration"})

# Bounds for generic numeric entities (device-specific bounds are checked
# against the live registry by the validator).
NUMERIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "value":    (0, 180),     # 180 = servo max angle; percentages are ≤100
    "amount":   (0, 100),
    "duration": (0, 24 * 3600),
}


# ─── Vocabulary ───────────────────────────────────────────────────────────────

# Canonical device names → maps onto devices.models.DeviceType values.
DEVICE_ALL = "all"
CANONICAL_DEVICES: Dict[str, str] = {
    "light": "LIGHT",
    "fan":   "FAN",
    "music": "MUSIC",
    "servo": "SERVO",
    DEVICE_ALL: "*",
}

DISPLAY_TARGETS: FrozenSet[str] = frozenset({"screen", "display", "monitor", "brightness"})

PRONOUNS: FrozenSet[str] = frozenset({"it", "that", "this", "them", "those", "these"})

# Actions explicitly outside the application's permission scope. Parsers may
# recognise them (so the user gets a clear refusal) but they are never routable.
BLOCKED_ACTIONS: FrozenSet[str] = frozenset({
    "shutdown_host", "restart_host", "delete_files", "run_shell",
    "install_software", "open_application", "send_message", "make_payment",
})

# Actions on DEVICE_ALL that always require explicit confirmation.
BULK_RISKY_ACTIONS: FrozenSet[str] = frozenset({
    A.TURN_ON.value, A.TURN_OFF.value, A.TOGGLE.value, A.SET_LEVEL.value,
})

MAX_RAW_INPUT_LENGTH = 500
MAX_ENTITY_STRING_LENGTH = 64
MAX_MULTI_ACTIONS = 5


# ─── Structured command ───────────────────────────────────────────────────────

class SchemaError(ValueError):
    """Raised when a candidate command does not match the structural schema."""


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass
class StructuredCommand:
    """
    The only shape that may cross the boundary between the intent engine and
    the action layer.
    """
    intent: str
    action: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    requires_confirmation: bool = False
    raw_input: str = ""
    actions: List["StructuredCommand"] = field(default_factory=list)

    @property
    def is_multi(self) -> bool:
        return self.intent == IntentType.MULTI_ACTION.value

    def children(self) -> List["StructuredCommand"]:
        return list(self.actions) if self.is_multi else [self]

    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        if self.is_multi:
            d: Dict[str, Any] = {
                "intent": self.intent,
                "actions": [c.to_dict(include_raw=False) for c in self.actions],
                "confidence": round(self.confidence, 3),
                "requires_confirmation": self.requires_confirmation,
            }
        else:
            d = {
                "intent": self.intent,
                "action": self.action,
                "entities": dict(self.entities),
                "parameters": dict(self.parameters),
                "confidence": round(self.confidence, 3),
                "requires_confirmation": self.requires_confirmation,
            }
        if include_raw:
            d["raw_input"] = self.raw_input
        return d

    # ── Strict structural parsing (used for every provider output) ───────────

    @classmethod
    def from_dict(cls, data: Any, raw_input: str = "", _depth: int = 0) -> "StructuredCommand":
        if not isinstance(data, dict):
            raise SchemaError("Command must be a JSON object.")

        intent = data.get("intent")
        if not isinstance(intent, str) or not intent.strip():
            raise SchemaError("Field 'intent' must be a non-empty string.")
        intent = intent.strip().lower()

        conf = data.get("confidence", 0.0)
        if not _is_number(conf):
            raise SchemaError("Field 'confidence' must be a number.")
        if not 0.0 <= float(conf) <= 1.0:
            raise SchemaError("Field 'confidence' must be between 0 and 1.")

        rc = data.get("requires_confirmation", False)
        if not isinstance(rc, bool):
            raise SchemaError("Field 'requires_confirmation' must be a boolean.")

        if intent == IntentType.MULTI_ACTION.value:
            if _depth > 0:
                raise SchemaError("Nested multi_action commands are not allowed.")
            raw_actions = data.get("actions")
            if not isinstance(raw_actions, list) or not raw_actions:
                raise SchemaError("multi_action requires a non-empty 'actions' list.")
            if len(raw_actions) > MAX_MULTI_ACTIONS:
                raise SchemaError(f"multi_action supports at most {MAX_MULTI_ACTIONS} actions.")
            children = [
                cls.from_dict({"confidence": conf, **a} if isinstance(a, dict) else a,
                              raw_input=raw_input, _depth=1)
                for a in raw_actions
            ]
            return cls(intent=intent, confidence=float(conf), requires_confirmation=rc,
                       raw_input=raw_input, actions=children)

        action = data.get("action")
        if action is not None and not isinstance(action, str):
            raise SchemaError("Field 'action' must be a string or null.")
        entities = data.get("entities")
        parameters = data.get("parameters")
        entities = {} if entities is None else entities
        parameters = {} if parameters is None else parameters
        if not isinstance(entities, dict):
            raise SchemaError("Field 'entities' must be an object.")
        if not isinstance(parameters, dict):
            raise SchemaError("Field 'parameters' must be an object.")

        return cls(
            intent=intent,
            action=action.strip().lower() if isinstance(action, str) else None,
            entities=dict(entities),
            parameters=dict(parameters),
            confidence=float(conf),
            requires_confirmation=rc,
            raw_input=raw_input,
        )


def schema_summary() -> Dict[str, Any]:
    """Machine-readable description of the schema (used for LLM prompts / docs)."""
    return {
        "intents": {k: sorted(v) for k, v in INTENT_ACTIONS.items()},
        "required_entities": {f"{i}.{a}": list(e) for (i, a), e in REQUIRED_ENTITIES.items()},
        "entities": sorted(SUPPORTED_ENTITIES),
        "devices": sorted(CANONICAL_DEVICES),
    }
