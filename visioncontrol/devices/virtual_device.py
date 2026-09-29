"""
devices/virtual_device.py

Phase 5: VirtualDevice — stateful abstraction for a single virtual device.

Each VirtualDevice wraps a DeviceState and exposes deterministic methods.
Phase 6 will replace or extend this layer for physical hardware.

Does NOT:
    - control Arduino
    - call OpenAI
    - contain UI code
"""

from devices.models import DeviceState, DeviceType, CommandResult


class VirtualDevice:
    """
    Encapsulates the mutable state of a single virtual device.

    All mutations return a CommandResult — never raise on bad input.
    Validation of range/bounds is performed here before mutation.
    """

    def __init__(self, state: DeviceState):
        self._state = state

    # ─── Read-only access ─────────────────────────────────────────────────────

    @property
    def state(self) -> DeviceState:
        return self._state

    @property
    def device_id(self) -> str:
        return self._state.device_id

    @property
    def device_type(self) -> DeviceType:
        return self._state.device_type

    # ─── Power operations ─────────────────────────────────────────────────────

    def power_on(self) -> CommandResult:
        if not self._state.enabled:
            return CommandResult(False, f"{self.device_id} is disabled.", self.device_id, "POWER_ON")
        if self._state.power:
            return CommandResult(True, f"{self.device_id} already ON.", self.device_id, "POWER_ON")
        self._state.power = True
        return CommandResult(True, f"{self.device_id} powered ON.", self.device_id, "POWER_ON")

    def power_off(self) -> CommandResult:
        if not self._state.enabled:
            return CommandResult(False, f"{self.device_id} is disabled.", self.device_id, "POWER_OFF")
        if not self._state.power:
            return CommandResult(True, f"{self.device_id} already OFF.", self.device_id, "POWER_OFF")
        self._state.power = False
        return CommandResult(True, f"{self.device_id} powered OFF.", self.device_id, "POWER_OFF")

    def toggle(self) -> CommandResult:
        if not self._state.enabled:
            return CommandResult(False, f"{self.device_id} is disabled.", self.device_id, "TOGGLE")
        if self._state.power:
            return self.power_off()
        return self.power_on()

    # ─── Level operations ─────────────────────────────────────────────────────

    def set_level(self, value: int) -> CommandResult:
        if not self._state.enabled:
            return CommandResult(
                False, f"{self.device_id} is disabled.", self.device_id, "SET_LEVEL", value
            )
        min_v = self._state.min_level
        max_v = self._state.max_level
        if not isinstance(value, int):
            return CommandResult(
                False,
                f"Level must be an integer, got {type(value).__name__}.",
                self.device_id, "SET_LEVEL", value,
            )
        if value < min_v or value > max_v:
            return CommandResult(
                False,
                f"Level {value} out of range [{min_v}, {max_v}] for {self.device_id}.",
                self.device_id, "SET_LEVEL", value,
            )
        self._state.level = value
        unit = self._state.level_unit
        return CommandResult(
            True,
            f"{self.device_id} level set to {value}{unit}.",
            self.device_id, "SET_LEVEL", value,
        )

    # ─── Safe state ───────────────────────────────────────────────────────────

    def emergency_stop(self) -> CommandResult:
        """Place device into a safe, powered-off state immediately."""
        self._state.power = False
        # Keep level at 0 for light/music/servo; fan stays at 0
        self._state.level = 0
        return CommandResult(
            True,
            f"{self.device_id} emergency stopped → SAFE STATE.",
            self.device_id, "EMERGENCY_STOP",
        )
