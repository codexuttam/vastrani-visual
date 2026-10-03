"""
intent/messages.py

Phase 10: Human-readable descriptions, confirmation prompts, and
clarification / error messages. No internal details or stack traces.
"""

from typing import Optional

from intent.schema import DEVICE_ALL, CANONICAL_DEVICES, IntentType, StructuredCommand
from intent.validator import ValidationCode, ValidationResult

_PHRASES = {
    "turn_on": "turn on {dev}", "turn_off": "turn off {dev}", "toggle": "toggle {dev}",
    "set_level": "set {dev} to {value}", "increase_level": "increase {dev}",
    "decrease_level": "decrease {dev}",
    "increase_brightness": "increase the brightness", "decrease_brightness": "lower the brightness",
    "set_brightness": "set the brightness to {value}%",
    "play": "play {media}", "pause": "pause {media}", "stop": "stop {media}",
    "next_track": "skip to the next track", "previous_track": "go to the previous track",
    "volume_up": "turn the volume up", "volume_down": "turn the volume down",
    "set_volume": "set the volume to {value}%",
    "next": "select the next device", "previous": "select the previous device",
    "select": "select the current device", "enter_control": "enter control mode",
    "confirm": "confirm the pending action",
    "show_effect": "show the face effect", "hide_effect": "hide the face effect",
    "next_effect": "switch to the next face effect", "previous_effect": "switch to the previous face effect",
    "emergency_stop": "perform an emergency stop on all devices", "status": "report device status",
}

AVAILABLE_DEVICES = ", ".join(d for d in CANONICAL_DEVICES if d != DEVICE_ALL)


def _device_phrase(cmd: StructuredCommand) -> str:
    dev = cmd.entities.get("device")
    if dev == DEVICE_ALL:
        return "all connected devices"
    loc = cmd.entities.get("location")
    return f"the {loc + ' ' if loc else ''}{dev or 'device'}"


def describe(cmd: StructuredCommand) -> str:
    if cmd.is_multi:
        parts = [describe(c) for c in cmd.actions]
        return ", then ".join(parts[:-1]) + (" and then " if len(parts) > 1 else "") + parts[-1]
    template = _PHRASES.get(cmd.action or "", "do '{action}'")
    return template.format(
        dev=_device_phrase(cmd),
        value=cmd.entities.get("value", "?"),
        media=cmd.entities.get("media", "music") if cmd.entities.get("media") else "the music",
        action=cmd.action,
    )


def confirmation_prompt(cmd: StructuredCommand) -> str:
    return f"I understood that you want to {describe(cmd)}. Should I continue?"


def clarification_message(cmd: Optional[StructuredCommand], v: Optional[ValidationResult]) -> str:
    if v is not None and v.code == ValidationCode.AMBIGUOUS:
        return "I'm not sure what you're referring to. Which device or item do you mean?"
    if v is not None and v.code == ValidationCode.MISSING_ENTITY and cmd is not None:
        child = cmd.actions[v.child_index] if cmd.is_multi and v.child_index is not None else cmd
        if "device" in v.missing:
            verb = (child.action or "control").replace("_", " ")
            return f"Which device should I {verb}? Available devices: {AVAILABLE_DEVICES}."
        return f"I need a {', '.join(v.missing)} to do that. Could you be more specific?"
    return "Sorry, I didn't understand that. Could you rephrase? For example: 'turn on the fan'."


def rejection_message(v: ValidationResult) -> str:
    if v.code == ValidationCode.PERMISSION_DENIED:
        return "Sorry, that action is outside what Vatsrani Vision is allowed to do."
    if v.code == ValidationCode.UNSUPPORTED_DEVICE:
        return f"I can't control that device. Available devices: {AVAILABLE_DEVICES}."
    if v.code == ValidationCode.OUT_OF_RANGE:
        return f"That value is out of range. {v.errors[0] if v.errors else ''}".strip()
    return "Sorry, I can't do that safely. Please try a different command."


LOW_CONFIDENCE_MESSAGE = "I'm not confident I understood that. Could you rephrase it?"
EMPTY_INPUT_MESSAGE = "Please say or type a command."
INTERNAL_ERROR_MESSAGE = "Something went wrong while processing your command. Please try again."
NOTHING_PENDING_MESSAGE = "There is nothing waiting for confirmation."
CANCELLED_MESSAGE = "Okay, I cancelled that."
EXPIRED_MESSAGE = "The confirmation request expired. Please repeat your command."
