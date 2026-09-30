"""
ai/safety.py

Phase 9: AI Safety Layer.

Deterministic validation of AIIntent output before passing to device controller.
Independent of OpenAI API calls.
"""

from typing import Optional, Tuple, Set
from ai.schemas import AIIntent, SUPPORTED_INTENTS
from devices.registry import DeviceRegistry


class AISafetyValidator:
    """
    Deterministic AI Safety Validator protecting device execution.
    """

    def __init__(self, min_confidence: float = 0.75):
        self.min_confidence: float = min_confidence

    def validate(
        self,
        intent: AIIntent,
        registry: Optional[DeviceRegistry] = None,
    ) -> Tuple[bool, str]:
        """
        Validates AIIntent against strict safety rules.

        Returns:
            (is_safe: bool, status_message: str)
        """
        # Rule 1: Validate supported intent name
        it = intent.intent.upper()
        if it not in SUPPORTED_INTENTS:
            return False, f"Unsupported intent: '{intent.intent}'"

        # Rule 2: Validate confidence threshold
        if intent.confidence < self.min_confidence:
            return False, f"Low confidence: {intent.confidence:.2f} < {self.min_confidence:.2f}"

        # Rule 3: Validate device ID if intent operates on a device
        device_intents = {"TURN_ON", "TURN_OFF", "TOGGLE", "SET_LEVEL"}
        if it in device_intents:
            if not intent.device_id:
                return False, f"Intent '{it}' requires a device_id."

            if registry is not None:
                if intent.device_id not in registry.device_ids:
                    return False, f"Unknown device ID: '{intent.device_id}'. Registered: {sorted(registry.device_ids)}"

                # Rule 4: Validate SET_LEVEL value range
                if it == "SET_LEVEL":
                    if intent.value is None:
                        return False, "SET_LEVEL intent requires a value."

                    try:
                        val = int(intent.value)
                    except (ValueError, TypeError):
                        return False, f"Invalid level value type: {type(intent.value).__name__}"

                    device = registry.get(intent.device_id)
                    if device is not None:
                        min_v = device.state.min_level
                        max_v = device.state.max_level
                        if val < min_v or val > max_v:
                            return False, (
                                f"Level {val} out of range for {intent.device_id} "
                                f"({device.state.device_type.value}). Allowed: [{min_v}, {max_v}]."
                            )

        return True, "Passed safety validation."
