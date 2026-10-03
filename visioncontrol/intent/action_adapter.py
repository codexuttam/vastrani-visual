"""
intent/action_adapter.py

Phase 10: Bridge from validated StructuredCommands to the EXISTING action
layer (DeviceController from Phase 5/6, ARController from Phase 8).

The intent engine decides WHAT; this adapter only translates to the
existing public operations, which decide HOW (validation, virtual device
state, CommandBus → Arduino). No device logic is duplicated here.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from devices.models import CommandResult, DeviceType
from gestures.types import GestureAction
from intent.schema import (
    Action as A, CANONICAL_DEVICES, DEVICE_ALL, IntentType as I, StructuredCommand,
)

# Default step sizes for relative adjustments, per device type.
DEFAULT_STEPS: Dict[DeviceType, int] = {
    DeviceType.LIGHT: 20, DeviceType.FAN: 1, DeviceType.MUSIC: 10, DeviceType.SERVO: 30,
}

Handler = Callable[[StructuredCommand], "ExecutionResult"]


@dataclass
class ExecutionResult:
    success: bool
    message: str
    details: List[str] = field(default_factory=list)
    device_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"success": self.success, "message": self.message,
                "details": list(self.details), "device_ids": list(self.device_ids)}


class ActionRouterAdapter:
    """Routes StructuredCommands to DeviceController / ARController operations."""

    def __init__(self, device_controller=None, ar_controller=None):
        self.device_controller = device_controller
        self.ar_controller = ar_controller
        self._custom: Dict[Tuple[str, str], Handler] = {}

    def register_handler(self, intent: str, action: str, handler: Handler) -> None:
        """Extension point for Phase 11/12 (e.g. new device classes)."""
        self._custom[(intent, action)] = handler

    # ── Entry point ───────────────────────────────────────────────────────────

    def execute(self, cmd: StructuredCommand) -> ExecutionResult:
        if cmd.is_multi:
            results = [self.execute(c) for c in cmd.actions]
            ok = all(r.success for r in results)
            return ExecutionResult(
                success=ok,
                message="All actions completed." if ok else "Some actions failed.",
                details=[r.message for r in results],
                device_ids=[d for r in results for d in r.device_ids],
            )
        custom = self._custom.get((cmd.intent, cmd.action))
        if custom is not None:
            return custom(cmd)
        route = {
            I.DEVICE_CONTROL.value: self._device,
            I.DISPLAY_CONTROL.value: self._display,
            I.MEDIA_CONTROL.value: self._media,
            I.NAVIGATION.value: self._gesture_action,
            I.GESTURE_COMMAND.value: self._gesture_action,
            I.FACE_COMMAND.value: self._face,
            I.SYSTEM_COMMAND.value: self._system,
        }.get(cmd.intent)
        if route is None:
            return ExecutionResult(False, f"No action route for intent '{cmd.intent}'.")
        try:
            return route(cmd)
        except Exception as e:  # never leak internals to the user
            return ExecutionResult(False, "The action could not be completed.", details=[type(e).__name__])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_controller(self):
        if self.device_controller is None:
            raise RuntimeError("No device controller attached.")
        return self.device_controller

    def _resolve(self, device: str):
        ctrl = self._require_controller()
        devices = ctrl.registry.list_devices()
        if device == DEVICE_ALL:
            return devices
        type_name = CANONICAL_DEVICES.get(device)
        return [d for d in devices if d.state.device_type.value == type_name]

    @staticmethod
    def _collect(results: List[CommandResult], ids: List[str]) -> ExecutionResult:
        if not results:
            return ExecutionResult(False, "No matching device found.")
        ok = all(r.success for r in results)
        msgs = [r.message for r in results]
        return ExecutionResult(ok, msgs[0] if len(msgs) == 1 else ("Done. " if ok else "Partially failed. ") + " | ".join(msgs),
                               details=msgs, device_ids=ids)

    def _adjust(self, devices, sign: int, amount=None) -> ExecutionResult:
        ctrl = self._require_controller()
        results, ids = [], []
        for d in devices:
            st = d.state
            step = int(amount) if amount is not None else DEFAULT_STEPS.get(st.device_type, 1)
            if st.device_type == DeviceType.FAN and amount is not None and amount > st.max_level:
                step = 1  # "by 20%" on a 0–3 fan → one notch
            target = max(st.min_level, min(st.max_level, st.level + sign * step))
            if not st.power and sign > 0:
                ctrl.power_on(d.device_id)
            results.append(ctrl.set_level(d.device_id, target))
            ids.append(d.device_id)
        return self._collect(results, ids)

    # ── Routes ────────────────────────────────────────────────────────────────

    def _device(self, cmd: StructuredCommand) -> ExecutionResult:
        ctrl = self._require_controller()
        devices = self._resolve(cmd.entities["device"])
        ids = [d.device_id for d in devices]
        a = cmd.action
        if a == A.TURN_ON.value:
            return self._collect([ctrl.power_on(i) for i in ids], ids)
        if a == A.TURN_OFF.value:
            return self._collect([ctrl.power_off(i) for i in ids], ids)
        if a == A.TOGGLE.value:
            return self._collect([ctrl.toggle(i) for i in ids], ids)
        if a == A.SET_LEVEL.value:
            return self._collect([ctrl.set_level(i, int(cmd.entities["value"])) for i in ids], ids)
        if a == A.INCREASE_LEVEL.value:
            return self._adjust(devices, +1, cmd.entities.get("amount"))
        if a == A.DECREASE_LEVEL.value:
            return self._adjust(devices, -1, cmd.entities.get("amount"))
        return ExecutionResult(False, f"Unsupported device action '{a}'.")

    def _display(self, cmd: StructuredCommand) -> ExecutionResult:
        # No dedicated display device exists yet; brightness maps to the
        # brightness-capable LIGHT devices in the registry.
        lights = self._resolve("light")
        if cmd.action == A.SET_BRIGHTNESS.value:
            ctrl = self._require_controller()
            ids = [d.device_id for d in lights]
            return self._collect([ctrl.set_level(i, int(cmd.entities["value"])) for i in ids], ids)
        sign = +1 if cmd.action == A.INCREASE_BRIGHTNESS.value else -1
        return self._adjust(lights, sign, cmd.entities.get("amount"))

    def _media(self, cmd: StructuredCommand) -> ExecutionResult:
        ctrl = self._require_controller()
        players = self._resolve("music")
        ids = [d.device_id for d in players]
        a = cmd.action
        if a == A.PLAY.value:
            return self._collect([ctrl.power_on(i) for i in ids], ids)
        if a in (A.PAUSE.value, A.STOP.value):
            return self._collect([ctrl.power_off(i) for i in ids], ids)
        if a == A.VOLUME_UP.value:
            return self._adjust(players, +1, cmd.entities.get("amount"))
        if a == A.VOLUME_DOWN.value:
            return self._adjust(players, -1, cmd.entities.get("amount"))
        if a == A.SET_VOLUME.value:
            return self._collect([ctrl.set_level(i, int(cmd.entities["value"])) for i in ids], ids)
        return ExecutionResult(False, "Track navigation is not supported by the current music device.")

    def _gesture_action(self, cmd: StructuredCommand) -> ExecutionResult:
        mapping = {
            A.NEXT.value: GestureAction.NEXT, A.PREVIOUS.value: GestureAction.PREVIOUS,
            A.SELECT.value: GestureAction.SELECT, A.CONFIRM.value: GestureAction.CONFIRM,
            A.ENTER_CONTROL.value: GestureAction.ENTER_CONTROL,
        }
        r = self._require_controller().handle_action(mapping[cmd.action])
        return ExecutionResult(r.success, r.message, device_ids=[r.device_id] if r.device_id else [])

    def _face(self, cmd: StructuredCommand) -> ExecutionResult:
        ar = self.ar_controller
        if ar is None:
            return ExecutionResult(False, "Face effects are not available.")
        a = cmd.action
        if a == A.NEXT_EFFECT.value:
            ar.next_effect()
        elif a == A.PREVIOUS_EFFECT.value:
            ar.previous_effect()
        elif a == A.SHOW_EFFECT.value:
            ar.state.visible = True
        elif a == A.HIDE_EFFECT.value:
            ar.state.visible = False
        return ExecutionResult(True, f"Face effect: {a.replace('_', ' ')}.")

    def _system(self, cmd: StructuredCommand) -> ExecutionResult:
        ctrl = self._require_controller()
        if cmd.action == A.EMERGENCY_STOP.value:
            r = ctrl.handle_action(GestureAction.EMERGENCY_STOP)
            return ExecutionResult(r.success, "Emergency stop: all devices are now off.", details=[r.message])
        if cmd.action == A.STATUS.value:
            lines = [d.state.summary() for d in ctrl.registry.list_devices()]
            return ExecutionResult(True, f"{sum(d.state.power for d in ctrl.registry.list_devices())} device(s) on.",
                                   details=lines)
        return ExecutionResult(False, "Unsupported system command.")
