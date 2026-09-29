"""
devices/validator.py

Phase 5: Command validation layer.

Every device command passes through the validator before execution.
Returns a structured CommandResult — never raises on bad input.
The same validation logic will later protect physical hardware (Phase 6+).

Does NOT:
    - control devices
    - call OpenAI
    - contain UI code
    - access Arduino
"""

from typing import Optional

from devices.models import DeviceState, DeviceType, CommandResult

# ─── Supported action names ───────────────────────────────────────────────────

SUPPORTED_ACTIONS = frozenset({
    "POWER_ON",
    "POWER_OFF",
    "TOGGLE",
    "SET_LEVEL",
    "EMERGENCY_STOP",
})


class DeviceValidator:
    """
    Validates device commands before execution.

    Usage:
        result = validator.validate_power("LIGHT_01", "POWER_ON", known_ids)
        result = validator.validate_level("SERVO_01", 190, state)
    """

    # ─── Device existence ─────────────────────────────────────────────────────

    def validate_device_exists(
        self,
        device_id: str,
        registered_ids: set,
    ) -> CommandResult:
        if device_id not in registered_ids:
            return CommandResult(
                False,
                f"Unknown device: '{device_id}'. "
                f"Registered: {sorted(registered_ids)}",
                device_id=device_id,
            )
        return CommandResult(True, "Device found.", device_id=device_id)

    # ─── Action existence ─────────────────────────────────────────────────────

    def validate_action(self, action: str) -> CommandResult:
        if action not in SUPPORTED_ACTIONS:
            return CommandResult(
                False,
                f"Unknown action: '{action}'. "
                f"Supported: {sorted(SUPPORTED_ACTIONS)}",
                action=action,
            )
        return CommandResult(True, "Action valid.", action=action)

    # ─── Level range ──────────────────────────────────────────────────────────

    def validate_level(
        self,
        device_id: str,
        value: int,
        state: DeviceState,
    ) -> CommandResult:
        if not isinstance(value, int):
            return CommandResult(
                False,
                f"Level must be an integer, got {type(value).__name__}.",
                device_id=device_id,
                action="SET_LEVEL",
                value=value,
            )
        min_v = state.min_level
        max_v = state.max_level
        if value < min_v or value > max_v:
            return CommandResult(
                False,
                (
                    f"Level {value} invalid for {device_id} "
                    f"({state.device_type.value}). "
                    f"Allowed range: [{min_v}, {max_v}]."
                ),
                device_id=device_id,
                action="SET_LEVEL",
                value=value,
            )
        return CommandResult(
            True,
            f"Level {value} valid for {device_id}.",
            device_id=device_id,
            action="SET_LEVEL",
            value=value,
        )

    # ─── Full command validation ──────────────────────────────────────────────

    def validate_command(
        self,
        device_id: str,
        action: str,
        registered_ids: set,
        state: Optional[DeviceState] = None,
        level: Optional[int] = None,
    ) -> CommandResult:
        """
        Full validation pipeline:
          1. Device exists
          2. Action is supported
          3. Level range (if action is SET_LEVEL)
        """
        # Step 1: device
        result = self.validate_device_exists(device_id, registered_ids)
        if not result.success:
            return result

        # Step 2: action
        result = self.validate_action(action)
        if not result.success:
            return result

        # Step 3: level (only for SET_LEVEL)
        if action == "SET_LEVEL":
            if level is None:
                return CommandResult(
                    False,
                    "SET_LEVEL requires a level value.",
                    device_id=device_id,
                    action=action,
                )
            if state is None:
                return CommandResult(
                    False,
                    "Cannot validate level without device state.",
                    device_id=device_id,
                    action=action,
                    value=level,
                )
            result = self.validate_level(device_id, level, state)
            if not result.success:
                return result

        return CommandResult(
            True,
            f"Command {action} on {device_id} is valid.",
            device_id=device_id,
            action=action,
            value=level,
        )
