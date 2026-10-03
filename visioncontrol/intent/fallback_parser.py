"""
intent/fallback_parser.py

Phase 10: Deterministic rule-based intent parser.

Used when:
    - no LLM provider is configured,
    - the LLM provider times out / errors,
    - the LLM returns malformed output.

Produces StructuredCommand objects only — never executes anything.
"""

import re
from typing import Dict, List, Optional, Tuple

from intent.entities import EntityExtractor, find_pronoun
from intent.normalizer import NormalizedInput, normalize
from intent.schema import (
    Action as A, DEVICE_ALL, IntentType as I, MAX_MULTI_ACTIONS, StructuredCommand,
)

# Confidence levels assigned by the deterministic parser.
CONF_EXPLICIT  = 0.95   # explicit verb + object ("turn on the fan")
CONF_SHORTHAND = 0.90   # terse form ("fan on")
CONF_FUZZY     = 0.72   # colloquial / weak verb ("kill the fan")
CONF_PRONOUN   = 0.50   # unresolved reference ("turn it on")
CONF_UNKNOWN   = 0.0

_SPLIT_RE = re.compile(r"\s*,\s*(?:and\s+|then\s+)?|\s+(?:and then|and also|then|and|also)\s+")


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


class FallbackParser:
    """Rule-based parser for common commands. Pure and deterministic."""

    def __init__(self, extractor: Optional[EntityExtractor] = None):
        self.extractor = extractor or EntityExtractor()

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, raw_input: str, normalized: Optional[NormalizedInput] = None) -> StructuredCommand:
        norm = normalized or normalize(raw_input)
        if norm.is_empty:
            return StructuredCommand(intent=I.UNKNOWN.value, confidence=CONF_UNKNOWN, raw_input=raw_input)

        segments = [s for s in _SPLIT_RE.split(norm.text) if s and s.strip()]
        if len(segments) > 1:
            children: List[StructuredCommand] = []
            for seg in segments[:MAX_MULTI_ACTIONS]:
                child = self.parse_segment(seg, raw_input)
                if child.intent == I.UNKNOWN.value and children:
                    inherited = self._inherit(seg, children[-1], raw_input)
                    if inherited is not None:
                        child = inherited
                children.append(child)
            if any(c.intent != I.UNKNOWN.value for c in children):
                return StructuredCommand(
                    intent=I.MULTI_ACTION.value,
                    confidence=min(c.confidence for c in children),
                    raw_input=raw_input,
                    actions=children,
                )
        return self.parse_segment(norm.text, raw_input)

    # ── Segment parsing ───────────────────────────────────────────────────────

    def _inherit(self, seg: str, prev: StructuredCommand, raw: str) -> Optional[StructuredCommand]:
        """'turn on the fan and the light' → second clause inherits turn_on."""
        ents = self.extractor.extract(seg)
        if prev.intent == I.DEVICE_CONTROL.value and "device" in ents:
            e = {"device": ents["device"]}
            if "location" in ents:
                e["location"] = ents["location"]
            return self._cmd(I.DEVICE_CONTROL, prev.action, e, min(prev.confidence, CONF_SHORTHAND), raw)
        return None

    @staticmethod
    def _cmd(intent: I, action, entities: Dict, conf: float, raw: str, **params) -> StructuredCommand:
        action_val = action.value if hasattr(action, "value") else action
        return StructuredCommand(intent=intent.value, action=action_val, entities=entities,
                                 parameters=params, confidence=conf, raw_input=raw)

    def parse_segment(self, t: str, raw: str) -> StructuredCommand:
        ents = self.extractor.extract(t)
        pronoun = find_pronoun(t)
        device = ents.get("device")

        def keep(*names) -> Dict:
            return {k: ents[k] for k in names if k in ents}

        # 1. Out-of-scope requests (recognised so we can refuse clearly) ─────
        blocked = self._blocked_action(t)
        if blocked:
            return self._cmd(I.SYSTEM_COMMAND, blocked, keep("application", "person"), CONF_EXPLICIT, raw)

        # 2. Emergency stop / status ─────────────────────────────────────────
        if _has(t, r"\b(emergency|panic)\b|\bstop (everything|all)\b|\bkill switch\b"):
            return self._cmd(I.SYSTEM_COMMAND, A.EMERGENCY_STOP, {}, 0.97, raw)
        if _has(t, r"\b(status|which devices are on|what is on|device report)\b"):
            return self._cmd(I.SYSTEM_COMMAND, A.STATUS, {}, CONF_EXPLICIT, raw)

        # 3. Face / AR effects ───────────────────────────────────────────────
        if _has(t, r"\b(emoji|emojis|effect|effects|filter|face mask|ar)\b"):
            if _has(t, r"\b(next|change|switch|cycle)\b"):
                return self._cmd(I.FACE_COMMAND, A.NEXT_EFFECT, {}, CONF_EXPLICIT, raw)
            if _has(t, r"\b(previous|prev|back|last)\b"):
                return self._cmd(I.FACE_COMMAND, A.PREVIOUS_EFFECT, {}, CONF_EXPLICIT, raw)
            if _has(t, r"\b(hide|off|disable|remove|stop)\b"):
                return self._cmd(I.FACE_COMMAND, A.HIDE_EFFECT, {}, CONF_EXPLICIT, raw)
            if _has(t, r"\b(show|on|enable|display|start|add)\b"):
                return self._cmd(I.FACE_COMMAND, A.SHOW_EFFECT, {}, CONF_EXPLICIT, raw)

        # 4. Gesture-equivalent commands ─────────────────────────────────────
        if _has(t, r"\benter control( mode)?\b|\bcontrol mode\b|\bstart control\b"):
            return self._cmd(I.GESTURE_COMMAND, A.ENTER_CONTROL, {}, CONF_EXPLICIT, raw)
        if _has(t, r"^confirm\b|\bconfirm (selection|action|it)\b"):
            return self._cmd(I.GESTURE_COMMAND, A.CONFIRM, {}, CONF_EXPLICIT, raw)
        if "gesture" in ents and _has(t, r"\b(simulate|do|perform|gesture)\b"):
            g = ents["gesture"]
            action = {"PINCH": A.CONFIRM, "TWO_FINGERS": A.SELECT, "OPEN_PALM": A.ENTER_CONTROL}.get(g)
            if action:
                return self._cmd(I.GESTURE_COMMAND, action, {"gesture": g}, CONF_SHORTHAND, raw)

        # 5. Navigation ──────────────────────────────────────────────────────
        media_word = "media" in ents and ents["media"] != "music" or _has(t, r"\b(song|track)\b")
        if not media_word:
            if _has(t, r"\b(next|go right)\b( device)?") and not device:
                return self._cmd(I.NAVIGATION, A.NEXT, {}, CONF_EXPLICIT if "device" in t else CONF_SHORTHAND, raw)
            if _has(t, r"\b(previous|prev|go back|go left)\b") and not device:
                return self._cmd(I.NAVIGATION, A.PREVIOUS, {}, CONF_EXPLICIT if "device" in t else CONF_SHORTHAND, raw)
            if _has(t, r"^select\b|\bselect (current )?device\b|\bselect\b.*\b(device|this one)\b"):
                return self._cmd(I.NAVIGATION, A.SELECT, {}, CONF_EXPLICIT, raw)

        # 6. Media ───────────────────────────────────────────────────────────
        m = self._media(t, ents, raw)
        if m is not None:
            return m

        # 7. Brightness ──────────────────────────────────────────────────────
        b = self._brightness(t, ents, device, pronoun, raw)
        if b is not None:
            return b

        # 8. Device power / level ────────────────────────────────────────────
        d = self._device(t, ents, device, pronoun, raw)
        if d is not None:
            return d

        # 9. Unknown ─────────────────────────────────────────────────────────
        e = {"target": pronoun} if pronoun else {}
        return self._cmd(I.UNKNOWN, None, e, CONF_UNKNOWN, raw)

    # ── Rule groups ───────────────────────────────────────────────────────────

    @staticmethod
    def _blocked_action(t: str) -> Optional[str]:
        rules: List[Tuple[str, str]] = [
            (r"\b(shut ?down|power off|turn off)\s+(computer|pc|laptop|system|machine|host)\b", "shutdown_host"),
            (r"\b(restart|reboot)\s+(computer|pc|laptop|system|machine|host)\b|^(restart|reboot)$", "restart_host"),
            (r"\b(delete|erase|wipe|format|rm)\b.*\b(file|files|folder|disk|drive|everything)\b|\brm\s+rf\b", "delete_files"),
            (r"\b(run|execute)\s+(command|script|shell|terminal|code)\b|\bsudo\b", "run_shell"),
            (r"\b(install|download)\b", "install_software"),
            (r"\b(open|launch)\s+(?!palm\b)[a-z0-9]+", "open_application"),
            (r"\b(send|text|message|email)\s+[a-z]+", "send_message"),
            (r"\b(pay|buy|purchase|transfer money)\b", "make_payment"),
        ]
        for pat, action in rules:
            if re.search(pat, t):
                return action
        return None

    def _media(self, t: str, ents: Dict, raw: str) -> Optional[StructuredCommand]:
        media = {"media": ents["media"]} if "media" in ents else {}
        if _has(t, r"\b(play|resume|unpause)\b"):
            return self._cmd(I.MEDIA_CONTROL, A.PLAY, media, CONF_EXPLICIT, raw)
        if _has(t, r"\bpause\b"):
            return self._cmd(I.MEDIA_CONTROL, A.PAUSE, media, CONF_EXPLICIT, raw)
        if _has(t, r"\b(skip|next)\b") and _has(t, r"\b(song|track|music)\b") or _has(t, r"^skip\b"):
            return self._cmd(I.MEDIA_CONTROL, A.NEXT_TRACK, media, CONF_EXPLICIT, raw)
        if _has(t, r"\b(previous|last|prev)\b") and _has(t, r"\b(song|track)\b"):
            return self._cmd(I.MEDIA_CONTROL, A.PREVIOUS_TRACK, media, CONF_EXPLICIT, raw)
        if _has(t, r"\bstop\b") and _has(t, r"\b(music|song|track|playback|audio)\b"):
            return self._cmd(I.MEDIA_CONTROL, A.STOP, media, CONF_EXPLICIT, raw)
        if _has(t, r"\bvolume\b|\blouder\b|\bquieter\b|\bsofter\b|\bmute\b"):
            if "value" in ents and _has(t, r"\b(set|to|at)\b"):
                return self._cmd(I.MEDIA_CONTROL, A.SET_VOLUME, {"value": ents["value"]}, CONF_EXPLICIT, raw)
            if _has(t, r"\bmute\b"):
                return self._cmd(I.MEDIA_CONTROL, A.SET_VOLUME, {"value": 0}, CONF_SHORTHAND, raw)
            amount = {"amount": ents["amount"]} if "amount" in ents else {}
            if _has(t, r"\b(up|louder|increase|raise|higher|more)\b"):
                return self._cmd(I.MEDIA_CONTROL, A.VOLUME_UP, amount, CONF_EXPLICIT, raw)
            if _has(t, r"\b(down|quieter|softer|decrease|lower|reduce|less)\b"):
                return self._cmd(I.MEDIA_CONTROL, A.VOLUME_DOWN, amount, CONF_EXPLICIT, raw)
        return None

    def _brightness(self, t, ents, device, pronoun, raw) -> Optional[StructuredCommand]:
        up = _has(t, r"\bbrighter\b|\bbrighten\b|\b(increase|raise|boost|more|turn up)\b.*\bbrightness\b|\bbrightness\b.*\b(up|higher|increase)\b")
        down = _has(t, r"\bdimmer\b|\bdim\b|\bdarker\b|\b(decrease|lower|reduce|less|turn down)\b.*\bbrightness\b|\bbrightness\b.*\b(down|lower|decrease)\b")
        set_ = _has(t, r"\bbrightness\b") and "value" in ents
        if not (up or down or set_):
            return None
        amount = {"amount": ents["amount"]} if "amount" in ents else {}

        if device == "light":
            e = {"device": "light", **({"location": ents["location"]} if "location" in ents else {}), **amount}
            if set_:
                return self._cmd(I.DEVICE_CONTROL, A.SET_LEVEL, {**e, "value": ents["value"]}, CONF_EXPLICIT, raw)
            return self._cmd(I.DEVICE_CONTROL, A.INCREASE_LEVEL if up else A.DECREASE_LEVEL, e, CONF_EXPLICIT, raw)

        e = dict(amount)
        if "target" in ents:
            e["target"] = ents["target"]
        elif pronoun:
            e["target"] = pronoun          # "make it brighter" → ambiguous
            return self._cmd(I.DISPLAY_CONTROL, A.INCREASE_BRIGHTNESS if up else A.DECREASE_BRIGHTNESS,
                             e, CONF_PRONOUN, raw)
        if set_:
            return self._cmd(I.DISPLAY_CONTROL, A.SET_BRIGHTNESS, {**e, "value": ents["value"]}, CONF_EXPLICIT, raw)
        return self._cmd(I.DISPLAY_CONTROL, A.INCREASE_BRIGHTNESS if up else A.DECREASE_BRIGHTNESS,
                         e, CONF_EXPLICIT, raw)

    def _device(self, t, ents, device, pronoun, raw) -> Optional[StructuredCommand]:
        has_on = _has(t, r"\bon\b")
        has_off = _has(t, r"\boff\b")
        verb = _has(t, r"\b(turn|switch|power|put)\b")
        action = None
        conf = CONF_EXPLICIT

        if has_on and has_off:
            action, conf = A.TOGGLE, CONF_PRONOUN
        elif _has(t, r"\b(toggle|flip)\b"):
            action = A.TOGGLE
        elif has_off or _has(t, r"\b(shut down|shut|deactivate|disable)\b"):
            action, conf = A.TURN_OFF, (CONF_EXPLICIT if verb or not has_off or len(t.split()) > 2 else CONF_SHORTHAND)
        elif _has(t, r"\b(kill|cut)\b"):
            action, conf = A.TURN_OFF, CONF_FUZZY
        elif _has(t, r"\bstop\b") and device:
            action, conf = A.TURN_OFF, CONF_SHORTHAND
        elif has_on or _has(t, r"\b(start|activate|enable)\b"):
            action, conf = A.TURN_ON, (CONF_EXPLICIT if verb or not has_on or len(t.split()) > 2 else CONF_SHORTHAND)
        elif _has(t, r"\b(fire up|boot up|wake up)\b"):
            action, conf = A.TURN_ON, CONF_FUZZY
        elif _has(t, r"\bset\b") and "value" in ents or (_has(t, r"\b(speed|level|angle)\s+\d") and device):
            action = A.SET_LEVEL
        elif _has(t, r"\b(increase|raise|turn up|speed up|faster|higher|more)\b"):
            action = A.INCREASE_LEVEL
        elif _has(t, r"\b(decrease|lower|reduce|turn down|slow down|slower|less)\b"):
            action = A.DECREASE_LEVEL

        if action is None:
            return None

        entities: Dict = {}
        if device:
            entities["device"] = device
        elif pronoun:
            entities["device"] = pronoun    # resolved from context or flagged ambiguous
            conf = CONF_PRONOUN
        for k in ("location", "value", "amount", "duration"):
            if k in ents:
                entities[k] = ents[k]
        if action != A.SET_LEVEL:
            entities.pop("value", None)
        return self._cmd(I.DEVICE_CONTROL, action, entities, conf, raw)
