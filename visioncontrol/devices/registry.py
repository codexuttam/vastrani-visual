"""
devices/registry.py

Phase 5: DeviceRegistry — manages registration and circular selection of virtual devices.

Does NOT:
    - control devices
    - call OpenAI
    - contain UI code
    - access Arduino
"""

from typing import Dict, List, Optional

from devices.models import DeviceState, DeviceType
from devices.virtual_device import VirtualDevice


class DeviceRegistry:
    """
    Manages registration and selection of virtual devices.

    Responsibilities:
        - register / retrieve devices
        - maintain an ordered device list
        - track selected device index
        - circular next / previous navigation
    """

    def __init__(self):
        self._devices: Dict[str, VirtualDevice] = {}
        self._order: List[str] = []        # insertion-ordered IDs
        self._selected_index: int = 0

    # ─── Registration ─────────────────────────────────────────────────────────

    def register(self, state: DeviceState) -> VirtualDevice:
        """Register a new device. Raises ValueError if already registered."""
        if state.device_id in self._devices:
            raise ValueError(f"Device '{state.device_id}' already registered.")
        device = VirtualDevice(state)
        self._devices[state.device_id] = device
        self._order.append(state.device_id)
        return device

    # ─── Retrieval ────────────────────────────────────────────────────────────

    def get(self, device_id: str) -> Optional[VirtualDevice]:
        """Return the VirtualDevice with the given ID, or None."""
        return self._devices.get(device_id)

    def list_devices(self) -> List[VirtualDevice]:
        """Return all registered devices in insertion order."""
        return [self._devices[did] for did in self._order]

    @property
    def device_ids(self) -> set:
        return set(self._devices.keys())

    @property
    def count(self) -> int:
        return len(self._order)

    # ─── Selection ────────────────────────────────────────────────────────────

    def current(self) -> Optional[VirtualDevice]:
        """Return the currently selected device, or None if registry is empty."""
        if not self._order:
            return None
        return self._devices[self._order[self._selected_index]]

    def select(self, device_id: str) -> Optional[VirtualDevice]:
        """Select a device by ID. Returns the device or None if not found."""
        if device_id not in self._devices:
            return None
        self._selected_index = self._order.index(device_id)
        return self._devices[device_id]

    def next(self) -> Optional[VirtualDevice]:
        """Advance selection to the next device (wraps around)."""
        if not self._order:
            return None
        self._selected_index = (self._selected_index + 1) % len(self._order)
        return self.current()

    def previous(self) -> Optional[VirtualDevice]:
        """Move selection to the previous device (wraps around)."""
        if not self._order:
            return None
        self._selected_index = (self._selected_index - 1) % len(self._order)
        return self.current()

    @property
    def selected_index(self) -> int:
        return self._selected_index


# ─── Factory: default virtual devices ────────────────────────────────────────

def create_default_registry() -> DeviceRegistry:
    """
    Create and return a DeviceRegistry pre-populated with the four
    standard virtual devices defined in Phase 5.
    """
    registry = DeviceRegistry()

    registry.register(DeviceState(
        device_id="LIGHT_01",
        name="Living Room Light",
        device_type=DeviceType.LIGHT,
    ))
    registry.register(DeviceState(
        device_id="FAN_01",
        name="Ceiling Fan",
        device_type=DeviceType.FAN,
    ))
    registry.register(DeviceState(
        device_id="MUSIC_01",
        name="Music Player",
        device_type=DeviceType.MUSIC,
    ))
    registry.register(DeviceState(
        device_id="SERVO_01",
        name="Servo Motor",
        device_type=DeviceType.SERVO,
    ))

    return registry
