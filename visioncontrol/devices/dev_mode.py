"""
devices/dev_mode.py

Phase 5: Developer-only device system test interface.

Allows simulating gesture actions without a webcam.
Run with:
    python -m devices.dev_mode

NOT a CLI application — minimal, for debugging only.
"""

from gestures.types import GestureAction, ControlMode
from devices.controller import DeviceController

COMMANDS = {
    "n":  ("NEXT",           GestureAction.NEXT),
    "p":  ("PREVIOUS",       GestureAction.PREVIOUS),
    "s":  ("SELECT",         GestureAction.SELECT),
    "c":  ("CONFIRM",        GestureAction.CONFIRM),
    "e":  ("EMERGENCY_STOP", GestureAction.EMERGENCY_STOP),
    "ec": ("ENTER_CONTROL",  GestureAction.ENTER_CONTROL),
    "q":  ("QUIT",           None),
}


def print_state(ctrl: DeviceController):
    device = ctrl.registry.current()
    if device is None:
        print("  [no devices]")
        return
    s = device.state
    print(f"  Device  : {s.device_id} — {s.name} ({s.device_type.value})")
    print(f"  Power   : {'ON' if s.power else 'OFF'}")
    print(f"  Level   : {s.level}{s.level_unit}")
    print(f"  Mode    : {ctrl.mode.value}")
    print(f"  Pending : {ctrl.pending_action or 'none'}")


def print_all_devices(ctrl: DeviceController):
    print("\n  All devices:")
    sel = ctrl.registry.current()
    for dev in ctrl.registry.list_devices():
        s = dev.state
        marker = " ►" if (sel and dev.device_id == sel.device_id) else "  "
        print(f"  {marker} {s.device_id:10s} {s.device_type.value:6s}  "
              f"{'ON' if s.power else 'OFF':3s}  lvl={s.level}{s.level_unit}")


def run_dev_mode():
    print("\n" + "=" * 50)
    print("  VisionControl — Phase 5 Dev Mode")
    print("=" * 50)
    print("  Commands:")
    for key, (name, _) in COMMANDS.items():
        print(f"    {key:4s} → {name}")
    print()

    ctrl = DeviceController()

    while True:
        print()
        print_state(ctrl)
        print_all_devices(ctrl)
        print()

        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  [dev mode exiting]")
            break

        if cmd not in COMMANDS:
            print(f"  Unknown command '{cmd}'. Options: {list(COMMANDS.keys())}")
            continue

        name, action = COMMANDS[cmd]

        if action is None:
            print("  [quit]")
            break

        result = ctrl.handle_action(action)
        print(f"  [{name}] → {'OK' if result.success else 'FAIL'}: {result.message}")


if __name__ == "__main__":
    run_dev_mode()
