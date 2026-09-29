"""
devices/controller.py

Phase 5: DeviceController — central device-control hub.

Consumes abstract GestureActions from Phase 4 and translates them
into validated virtual device operations.

Architecture:
    GestureAction → DeviceController → Validator → VirtualDevice → DeviceState

Does NOT:
    - process raw hand landmarks
    - call OpenAI
    - access Arduino / serial
    - contain UI code
"""

import time
from collections import deque
from typing import Deque, List, Optional

from gestures.types import GestureAction, ControlMode
from devices.models import CommandRecord, CommandResult, DeviceType
from devices.registry import DeviceRegistry, create_default_registry
from devices.validator import DeviceValidator
from devices.virtual_device import VirtualDevice
from devices.command_bus import CommandBus

# ─── Command history size (configurable) ─────────────────────────────────────

COMMAND_HISTORY_SIZE = 50


# ─── Pending action enum ──────────────────────────────────────────────────────

class PendingAction:
    """Represents an action staged for CONFIRM."""
    TOGGLE = "TOGGLE"
    POWER_ON = "POWER_ON"
    POWER_OFF = "POWER_OFF"


class DeviceController:
    """
    Central device-control hub.

    Usage:
        controller = DeviceController()
        controller.handle_action(GestureAction.NEXT)
        controller.handle_action(GestureAction.EMERGENCY_STOP)

    Direct operations:
        controller.power_on("LIGHT_01")
        controller.power_off("FAN_01")
        controller.toggle("MUSIC_01")
        controller.set_level("SERVO_01", 90)
    """

    def __init__(
        self,
        registry: Optional[DeviceRegistry] = None,
        command_bus: Optional[CommandBus] = None,
        mode: str = "VIRTUAL",
    ):
        self.registry: DeviceRegistry = registry or create_default_registry()
        self.validator = DeviceValidator()
        self.mode: ControlMode = ControlMode.IDLE

        # Phase 6: Command Bus
        self.command_bus: CommandBus = command_bus or CommandBus(mode=mode)

        # Pending action — set by SELECT, consumed by CONFIRM
        self._pending_action: Optional[str] = None

        # Command history ring buffer
        self._history: Deque[CommandRecord] = deque(maxlen=COMMAND_HISTORY_SIZE)

    # ─── Action handler (entry point for gesture actions) ─────────────────────

    def handle_action(self, action: GestureAction) -> CommandResult:
        """
        Translate an abstract GestureAction into a device operation.
        This is the only entry point that accepts GestureActions.
        """
        if action == GestureAction.ENTER_CONTROL:
            return self._enter_control()
        elif action == GestureAction.NEXT:
            return self._next()
        elif action == GestureAction.PREVIOUS:
            return self._previous()
        elif action == GestureAction.SELECT:
            return self._select()
        elif action == GestureAction.CONFIRM:
            return self._confirm()
        elif action == GestureAction.EMERGENCY_STOP:
            return self._emergency_stop()
        elif action == GestureAction.NONE:
            return CommandResult(True, "No action.", action="NONE")
        else:
            return CommandResult(False, f"Unhandled action: {action}.", action=str(action))

    # ─── Private action implementations ──────────────────────────────────────

    def _enter_control(self) -> CommandResult:
        if self.mode == ControlMode.CONTROL:
            return CommandResult(True, "Already in CONTROL mode.", action="ENTER_CONTROL")
        self.mode = ControlMode.CONTROL
        return CommandResult(True, "Entered CONTROL mode.", action="ENTER_CONTROL")

    def _next(self) -> CommandResult:
        device = self.registry.next()
        if device is None:
            return CommandResult(False, "No devices registered.", action="NEXT")
        self._pending_action = None
        msg = f"Selected next device: {device.device_id} ({device.state.name})"
        return CommandResult(True, msg, device_id=device.device_id, action="NEXT")

    def _previous(self) -> CommandResult:
        device = self.registry.previous()
        if device is None:
            return CommandResult(False, "No devices registered.", action="PREVIOUS")
        self._pending_action = None
        msg = f"Selected previous device: {device.device_id} ({device.state.name})"
        return CommandResult(True, msg, device_id=device.device_id, action="PREVIOUS")

    def _select(self) -> CommandResult:
        device = self.registry.current()
        if device is None:
            return CommandResult(False, "No current device.", action="SELECT")
        # Stage a TOGGLE as the pending action for the next CONFIRM
        self._pending_action = PendingAction.TOGGLE
        msg = (
            f"Device selected: {device.device_id} ({device.state.name}) | "
            f"Power: {'ON' if device.state.power else 'OFF'} | "
            f"Pending: TOGGLE"
        )
        return CommandResult(True, msg, device_id=device.device_id, action="SELECT")

    def _confirm(self) -> CommandResult:
        if self._pending_action is None:
            return CommandResult(True, "CONFIRM: no pending action.", action="CONFIRM")
        device = self.registry.current()
        if device is None:
            return CommandResult(False, "CONFIRM: no current device.", action="CONFIRM")

        pending = self._pending_action
        self._pending_action = None  # consume

        if pending == PendingAction.TOGGLE:
            result = self.toggle(device.device_id)
        elif pending == PendingAction.POWER_ON:
            result = self.power_on(device.device_id)
        elif pending == PendingAction.POWER_OFF:
            result = self.power_off(device.device_id)
        else:
            result = CommandResult(False, f"Unknown pending action: {pending}.")

        return result

    def _emergency_stop(self) -> CommandResult:
        """Immediately place every registered device in a safe state."""
        self.mode = ControlMode.IDLE
        self._pending_action = None
        messages = []

        # Hardware emergency stop transmission
        hw_sent, hw_ack, hw_err = self.command_bus.send("SYSTEM", "EMERGENCY_STOP", None)

        for device in self.registry.list_devices():
            result = device.emergency_stop()
            self._record(
                device.device_id, "EMERGENCY_STOP", None, result,
                hw_sent=hw_sent, hw_ack=hw_ack, hw_error=hw_err
            )
            messages.append(result.message)

        return CommandResult(
            True,
            "EMERGENCY STOP: all devices safe. " + " | ".join(messages),
            action="EMERGENCY_STOP",
        )

    # ─── Direct device operations (also usable from tests / dev mode) ─────────

    def power_on(self, device_id: str) -> CommandResult:
        result = self._validate("POWER_ON", device_id)
        if not result.success:
            return result
        device = self.registry.get(device_id)
        result = device.power_on()

        hw_sent, hw_ack, hw_err = self.command_bus.send(device_id, "POWER_ON", None)
        self._record(device_id, "POWER_ON", None, result, hw_sent, hw_ack, hw_err)
        return result

    def power_off(self, device_id: str) -> CommandResult:
        result = self._validate("POWER_OFF", device_id)
        if not result.success:
            return result
        device = self.registry.get(device_id)
        result = device.power_off()

        hw_sent, hw_ack, hw_err = self.command_bus.send(device_id, "POWER_OFF", None)
        self._record(device_id, "POWER_OFF", None, result, hw_sent, hw_ack, hw_err)
        return result

    def toggle(self, device_id: str) -> CommandResult:
        result = self._validate("TOGGLE", device_id)
        if not result.success:
            return result
        device = self.registry.get(device_id)
        result = device.toggle()

        hw_sent, hw_ack, hw_err = self.command_bus.send(device_id, "TOGGLE", None)
        self._record(device_id, "TOGGLE", None, result, hw_sent, hw_ack, hw_err)
        return result

    def set_level(self, device_id: str, value: int) -> CommandResult:
        device = self.registry.get(device_id)
        result = self.validator.validate_command(
            device_id, "SET_LEVEL",
            self.registry.device_ids,
            state=device.state if device else None,
            level=value,
        )
        if not result.success:
            self._record(device_id, "SET_LEVEL", value, result, False, False, result.message)
            return result
        result = device.set_level(value)

        hw_sent, hw_ack, hw_err = self.command_bus.send(device_id, "SET_LEVEL", value)
        self._record(device_id, "SET_LEVEL", value, result, hw_sent, hw_ack, hw_err)
        return result

    # ─── History ──────────────────────────────────────────────────────────────

    def command_history(self) -> List[CommandRecord]:
        return list(self._history)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _validate(self, action: str, device_id: str, level: Optional[int] = None) -> CommandResult:
        return self.validator.validate_command(
            device_id, action, self.registry.device_ids, level=level
        )

    def _record(
        self,
        device_id: str,
        action: str,
        value: Optional[int],
        result: CommandResult,
        hw_sent: bool = False,
        hw_ack: bool = False,
        hw_error: Optional[str] = None,
    ):
        self._history.append(CommandRecord(
            timestamp=time.time(),
            device_id=device_id,
            action=action,
            value=value,
            virtual_success=result.success,
            hardware_sent=hw_sent,
            hardware_ack=hw_ack,
            error=hw_error or (None if result.success else result.message),
            success=result.success,
            message=result.message,
        ))

    # ─── State accessors ──────────────────────────────────────────────────────

    @property
    def current_device(self) -> Optional[VirtualDevice]:
        return self.registry.current()

    @property
    def pending_action(self) -> Optional[str]:
        return self._pending_action

    def set_pending_action(self, action: Optional[str]):
        """Allows external code (e.g., dev mode) to stage a pending action."""
        self._pending_action = action
