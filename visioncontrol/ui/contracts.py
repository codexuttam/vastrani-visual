"""
ui/contracts.py

Data models and contracts for VisionControl UI / HUD system.
"""

from enum import Enum
from dataclasses import dataclass, field
import time
from typing import Optional, List, Dict, Any


class SystemStatus(Enum):
    ONLINE = "ONLINE"
    INITIALIZING = "INITIALIZING"
    CAMERA_OFFLINE = "CAMERA OFFLINE"
    AI_OFFLINE = "AI SERVICE OFFLINE"
    ARDUINO_DISCONNECTED = "ARDUINO DISCONNECTED"
    DEGRADED = "DEGRADED"
    ERROR = "SYSTEM ERROR"


class HUDMode(Enum):
    STANDARD = "STANDARD"
    DEVELOPER = "DEVELOPER"


class NotificationCategory(Enum):
    SUCCESS = "SUCCESS"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class NotificationItem:
    id: str
    title: str
    message: str
    category: NotificationCategory
    timestamp: float = field(default_factory=time.time)
    duration_sec: float = 4.0
    dismissed: bool = False

    @property
    def remaining_time(self) -> float:
        return max(0.0, self.duration_sec - (time.time() - self.timestamp))

    @property
    def is_expired(self) -> bool:
        return self.dismissed or (time.time() - self.timestamp >= self.duration_sec)


@dataclass
class EventItem:
    id: str
    timestamp_str: str
    source: str
    message: str
    level: str = "INFO"  # INFO, WARN, ERROR, SUCCESS
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceMetrics:
    fps: float = 0.0
    vision_ms: float = 0.0
    face_ms: float = 0.0
    intent_ms: float = 0.0
    serial_ms: float = 0.0
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    frame_width: int = 1280
    frame_height: int = 720
