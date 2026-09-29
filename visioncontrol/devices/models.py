"""
devices/models.py

Phase 5: Device data models.

Defines strongly-typed device representations.
No UI logic. No gesture logic. No Arduino.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─── Device Type ──────────────────────────────────────────────────────────────

class DeviceType(Enum):
    LIGHT = "LIGHT"
    FAN   = "FAN"
    MUSIC = "MUSIC"
    SERVO = "SERVO"


# ─── Device State ─────────────────────────────────────────────────────────────

@dataclass
class DeviceState:
    """
    Complete state representation for a virtual device.

    level semantics per device type:
        LIGHT  : 0–100 (brightness %)
        FAN    : 0–3   (speed setting)
        MUSIC  : 0–100 (volume %)
        SERVO  : 0–180 (angle °)
    """
    device_id:   str
    name:        str
    device_type: DeviceType
    enabled:     bool = True
    power:       bool = False
    level:       int  = 0

    # ── Level bounds per type ────────────────────────────────────────────────

    @property
    def max_level(self) -> int:
        return {
            DeviceType.LIGHT: 100,
            DeviceType.FAN:   3,
            DeviceType.MUSIC: 100,
            DeviceType.SERVO: 180,
        }[self.device_type]

    @property
    def min_level(self) -> int:
        return 0

    @property
    def level_unit(self) -> str:
        return {
            DeviceType.LIGHT: "%",
            DeviceType.FAN:   "/3",
            DeviceType.MUSIC: "%",
            DeviceType.SERVO: "°",
        }[self.device_type]

    def summary(self) -> str:
        power_str = "ON" if self.power else "OFF"
        return (
            f"{self.name} [{self.device_id}] | "
            f"Type: {self.device_type.value} | "
            f"Power: {power_str} | "
            f"Level: {self.level}{self.level_unit}"
        )


# ─── Command Result ───────────────────────────────────────────────────────────

@dataclass
class CommandResult:
    """Structured result from every device command attempt."""
    success: bool
    message: str
    device_id: Optional[str] = None
    action:    Optional[str] = None
    value:     Optional[int] = None

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return f"CommandResult[{status}] {self.message}"


# ─── Command Record ───────────────────────────────────────────────────────────

@dataclass
class CommandRecord:
    """Immutable audit record of an executed device command."""
    timestamp:       float
    device_id:       str
    action:          str
    value:           Optional[int] = None
    virtual_success: bool = True
    hardware_sent:   bool = False
    hardware_ack:    bool = False
    error:           Optional[str] = None
    success:         bool = True
    message:         str = ""

    def __post_init__(self):
        if self.error and not self.message:
            self.message = self.error

    def formatted_time(self) -> str:
        import time
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))
