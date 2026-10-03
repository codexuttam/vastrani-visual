"""
intent/validator.py

Phase 10: Intent Validator + Safety / Permission layer.

Sits between ANY parser output (LLM or fallback) and the action layer.
Rejects malformed, unsupported, out-of-scope, or impossible commands and
flags commands that need clarification or confirmation.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from intent.schema import (
    BLOCKED_ACTIONS, BULK_RISKY_ACTIONS, CANONICAL_DEVICES, DEVICE_ALL,
    DISPLAY_TARGETS, INTENT_ACTIONS, IntentType, MAX_ENTITY_STRING_LENGTH,
    NUMERIC_BOUNDS, NUMERIC_ENTITIES, PRONOUNS, REQUIRED_ENTITIES,
    SUPPORTED_ENTITIES, SUPPORTED_INTENTS, StructuredCommand,
)


class ValidationCode:
    OK                 = "OK"
    UNKNOWN_INTENT     = "UNKNOWN_INTENT"
    UNSUPPORTED_INTENT = "UNSUPPORTED_INTENT"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
    MISSING_ENTITY     = "MISSING_ENTITY"
    AMBIGUOUS          = "AMBIGUOUS"
    INVALID_ENTITY     = "INVALID_ENTITY"
    INVALID_PARAMETER  = "INVALID_PARAMETER"
    OUT_OF_RANGE       = "OUT_OF_RANGE"
    UNSUPPORTED_DEVICE = "UNSUPPORTED_DEVICE"
    PERMISSION_DENIED  = "PERMISSION_DENIED"

    # Codes that mean "ask the user for more information" rather than "refuse".
    CLARIFY = frozenset({UNKNOWN_INTENT, MISSING_ENTITY, AMBIGUOUS})


@dataclass
class ValidationResult:
    valid: bool
    code: str = ValidationCode.OK
    errors: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    child_index: Optional[int] = None

    @property
    def needs_clarification(self) -> bool:
        return not self.valid and self.code in ValidationCode.CLARIFY


def _fail(code: str, msg: str, **kw) -> ValidationResult:
    return ValidationResult(valid=False, code=code, errors=[msg], **kw)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class IntentValidator:
    """Schema + safety validation. Stateless; registry is optional."""

    def validate(self, cmd: StructuredCommand, registry=None) -> ValidationResult:
        if not isinstance(cmd, StructuredCommand):
            return _fail(ValidationCode.INVALID_PARAMETER, "Command is not a StructuredCommand.")

        if cmd.is_multi:
            if not cmd.actions:
                return _fail(ValidationCode.MISSING_ENTITY, "multi_action has no child actions.")
            needs_confirm = cmd.requires_confirmation
            for idx, child in enumerate(cmd.actions):
                r = self._validate_single(child, registry)
                if not r.valid:
                    r.child_index = idx
                    return r
                needs_confirm = needs_confirm or r.requires_confirmation
            return ValidationResult(valid=True, requires_confirmation=needs_confirm)
        return self._validate_single(cmd, registry)

    # ── Single command ────────────────────────────────────────────────────────

    def _validate_single(self, cmd: StructuredCommand, registry) -> ValidationResult:
        intent = cmd.intent
        if intent not in SUPPORTED_INTENTS:
            return _fail(ValidationCode.UNSUPPORTED_INTENT, f"Unsupported intent '{intent}'.")
        if intent == IntentType.MULTI_ACTION.value:
            return _fail(ValidationCode.UNSUPPORTED_INTENT, "Nested multi_action is not allowed.")

        pronoun = self._pronoun_entity(cmd)
        if intent == IntentType.UNKNOWN.value:
            if pronoun:
                return _fail(ValidationCode.AMBIGUOUS, f"Unclear reference '{pronoun}'.", missing=["target"])
            return _fail(ValidationCode.UNKNOWN_INTENT, "Could not determine what you want to do.")

        # Permission scope is checked before the generic action check so the
        # user receives an explicit refusal.
        if cmd.action in BLOCKED_ACTIONS:
            return _fail(ValidationCode.PERMISSION_DENIED,
                         f"'{cmd.action}' is outside Vatsrani Vision's permission scope.")
        if not cmd.action or cmd.action not in INTENT_ACTIONS[intent]:
            return _fail(ValidationCode.UNSUPPORTED_ACTION,
                         f"Action '{cmd.action}' is not supported for intent '{intent}'.")

        # Entities: names, types, sizes
        for k, v in cmd.entities.items():
            if k not in SUPPORTED_ENTITIES:
                return _fail(ValidationCode.INVALID_ENTITY, f"Unsupported entity '{k}'.")
            if k in NUMERIC_ENTITIES:
                if not _is_number(v):
                    return _fail(ValidationCode.INVALID_PARAMETER, f"Entity '{k}' must be numeric.")
                lo, hi = NUMERIC_BOUNDS[k]
                if not lo <= v <= hi:
                    return _fail(ValidationCode.OUT_OF_RANGE, f"Entity '{k}'={v} is outside {lo}–{hi}.")
            elif not isinstance(v, str) or not v or len(v) > MAX_ENTITY_STRING_LENGTH:
                return _fail(ValidationCode.INVALID_ENTITY, f"Entity '{k}' must be a short string.")

        for k, v in cmd.parameters.items():
            if not isinstance(k, str) or not (v is None or isinstance(v, (str, int, float, bool))):
                return _fail(ValidationCode.INVALID_PARAMETER, f"Parameter '{k}' has an invalid type.")

        if pronoun:
            return _fail(ValidationCode.AMBIGUOUS, f"Unclear reference '{pronoun}'.",
                         missing=list(REQUIRED_ENTITIES.get((intent, cmd.action), ("target",))))

        missing = [e for e in REQUIRED_ENTITIES.get((intent, cmd.action), ()) if e not in cmd.entities]
        if missing:
            return _fail(ValidationCode.MISSING_ENTITY, f"Missing required entity: {', '.join(missing)}.",
                         missing=missing)

        device = cmd.entities.get("device")
        if device is not None:
            if device not in CANONICAL_DEVICES:
                return _fail(ValidationCode.UNSUPPORTED_DEVICE, f"Unknown device '{device}'.")
            r = self._check_device_bounds(cmd, device, registry)
            if r is not None:
                return r

        target = cmd.entities.get("target")
        if intent == IntentType.DISPLAY_CONTROL.value and target and target not in DISPLAY_TARGETS:
            return _fail(ValidationCode.INVALID_ENTITY, f"'{target}' is not a display target.")
        if intent in (IntentType.DISPLAY_CONTROL.value, IntentType.MEDIA_CONTROL.value):
            v = cmd.entities.get("value")
            if v is not None and not 0 <= v <= 100:
                return _fail(ValidationCode.OUT_OF_RANGE, f"Value {v} must be between 0 and 100.")

        requires_confirmation = cmd.requires_confirmation or (
            device == DEVICE_ALL and cmd.action in BULK_RISKY_ACTIONS
        )
        return ValidationResult(valid=True, requires_confirmation=requires_confirmation)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pronoun_entity(cmd: StructuredCommand) -> Optional[str]:
        for k in ("device", "target"):
            v = cmd.entities.get(k)
            if isinstance(v, str) and v in PRONOUNS:
                return v
        return None

    @staticmethod
    def _check_device_bounds(cmd, device, registry) -> Optional[ValidationResult]:
        value = cmd.entities.get("value")
        if device == DEVICE_ALL:
            return None
        if registry is None:
            if value is not None:
                from devices.models import DeviceState, DeviceType
                st = DeviceState("_", "_", DeviceType(CANONICAL_DEVICES[device]))
                if not st.min_level <= value <= st.max_level:
                    return _fail(ValidationCode.OUT_OF_RANGE,
                                 f"{device} level must be between {st.min_level} and {st.max_level}.")
            return None
        type_name = CANONICAL_DEVICES[device]
        matches = [d for d in registry.list_devices() if d.state.device_type.value == type_name]
        if not matches:
            return _fail(ValidationCode.UNSUPPORTED_DEVICE, f"No '{device}' device is registered.")
        if value is not None:
            st = matches[0].state
            if not st.min_level <= value <= st.max_level:
                return _fail(ValidationCode.OUT_OF_RANGE,
                             f"{device} level must be between {st.min_level} and {st.max_level}.")
        return None
