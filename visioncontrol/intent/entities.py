"""
intent/entities.py

Phase 10: Extensible entity extraction.

Each extractor is a small function  (text) -> dict  registered on the
EntityExtractor. New entity types (Phase 11/12) are added via
`EntityExtractor.register(name, fn)` without touching the parser.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from intent.schema import DEVICE_ALL, DISPLAY_TARGETS, PRONOUNS

Extractor = Callable[[str], Optional[Any]]

# Alias → canonical device name. Longest aliases are matched first.
DEVICE_ALIASES: Dict[str, str] = {
    "ceiling fan": "fan", "fans": "fan", "fan": "fan",
    "lights": "light", "light": "light", "lamps": "light", "lamp": "light",
    "bulbs": "light", "bulb": "light", "lighting": "light",
    "music player": "music", "speakers": "music", "speaker": "music",
    "stereo": "music", "music": "music",
    "servo motor": "servo", "servo": "servo", "motor": "servo",
    "all devices": DEVICE_ALL, "every device": DEVICE_ALL,
    "everything": DEVICE_ALL, "all": DEVICE_ALL,
}

LOCATIONS = [
    "living room", "bedroom", "kitchen", "bathroom", "office", "hall",
    "hallway", "garage", "dining room", "study", "balcony", "lounge",
]

MEDIA_WORDS = ["music", "song", "track", "playlist", "video", "podcast", "radio", "audio"]

GESTURES = {
    "two fingers": "TWO_FINGERS", "open palm": "OPEN_PALM", "swipe left": "SWIPE_LEFT",
    "swipe right": "SWIPE_RIGHT", "thumbs up": "THUMBS_UP", "pinch": "PINCH", "fist": "FIST",
}

_DURATION_UNITS = {"s": 1, "sec": 1, "second": 1, "m": 60, "min": 60, "minute": 60, "h": 3600, "hr": 3600, "hour": 3600}


def _first_phrase(text: str, phrases) -> Optional[str]:
    for p in sorted(phrases, key=len, reverse=True):
        if re.search(r"\b" + re.escape(p) + r"\b", text):
            return p
    return None


def _num(s: str):
    v = float(s)
    return int(v) if v.is_integer() else v


def extract_device(text: str) -> Optional[str]:
    alias = _first_phrase(text, DEVICE_ALIASES.keys())
    if alias is None:
        return None
    # "all lights" → light, not all
    if DEVICE_ALIASES[alias] == DEVICE_ALL and alias == "all":
        specific = _first_phrase(text, [a for a, c in DEVICE_ALIASES.items() if c != DEVICE_ALL])
        if specific:
            return DEVICE_ALIASES[specific]
    return DEVICE_ALIASES[alias]


def extract_location(text: str) -> Optional[str]:
    return _first_phrase(text, LOCATIONS)


def extract_target(text: str) -> Optional[str]:
    t = _first_phrase(text, [d for d in DISPLAY_TARGETS if d != "brightness"])
    return t


def extract_value(text: str):
    m = re.search(r"\b(?:to|at|speed|level|volume)\s+(\d+(?:\.\d+)?)%?", text)
    if m:
        return _num(m.group(1))
    m = re.search(r"\b(\d+(?:\.\d+)?)%", text)
    if m and not re.search(r"\bby\s+" + re.escape(m.group(1)), text):
        return _num(m.group(1))
    return None


def extract_amount(text: str):
    m = re.search(r"\bby\s+(\d+(?:\.\d+)?)%?", text)
    return _num(m.group(1)) if m else None


def extract_direction(text: str) -> Optional[str]:
    m = re.search(r"\b(up|down|left|right|forward|backward)\b", text)
    return m.group(1) if m else None


def extract_duration(text: str) -> Optional[int]:
    m = re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s*(s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?)\b", text)
    if not m:
        return None
    unit = m.group(2).rstrip("s") or "s"
    return int(float(m.group(1)) * _DURATION_UNITS.get(unit, 1))


def extract_application(text: str) -> Optional[str]:
    m = re.search(r"\b(?:open|launch|run)\s+([a-z0-9]+)", text)
    return m.group(1) if m else None


def extract_media(text: str) -> Optional[str]:
    return _first_phrase(text, MEDIA_WORDS)


def extract_person(text: str) -> Optional[str]:
    m = re.search(r"\b(?:call|message|text|email)\s+([a-z]+)", text)
    return m.group(1) if m else None


def extract_gesture(text: str) -> Optional[str]:
    g = _first_phrase(text, GESTURES.keys())
    return GESTURES[g] if g else None


def find_pronoun(text: str) -> Optional[str]:
    for tok in text.split():
        if tok.strip(",") in PRONOUNS:
            return tok.strip(",")
    return None


class EntityExtractor:
    """Runs every registered extractor and returns the non-empty results."""

    def __init__(self):
        self._extractors: List[Tuple[str, Extractor]] = []
        for name, fn in (
            ("device", extract_device), ("location", extract_location),
            ("target", extract_target), ("value", extract_value),
            ("amount", extract_amount), ("direction", extract_direction),
            ("duration", extract_duration), ("application", extract_application),
            ("media", extract_media), ("person", extract_person),
            ("gesture", extract_gesture),
        ):
            self.register(name, fn)

    def register(self, name: str, fn: Extractor) -> None:
        self._extractors = [(n, f) for n, f in self._extractors if n != name]
        self._extractors.append((name, fn))

    def extract(self, text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, fn in self._extractors:
            try:
                v = fn(text)
            except Exception:
                v = None
            if v is not None:
                out[name] = v
        return out
