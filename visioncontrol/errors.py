"""
errors.py

Centralized Structured Error Handling & Classification System.
Phase 12 — Hardening & Reliability.
"""

from enum import Enum
from dataclasses import dataclass, field
import time
from typing import Optional, Dict, Any


class ErrorSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    RECOVERABLE_ERROR = "RECOVERABLE_ERROR"
    CRITICAL_ERROR = "CRITICAL_ERROR"


@dataclass
class SystemErrorRecord:
    code: str
    subsystem: str
    severity: ErrorSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "subsystem": self.subsystem,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
            "details": self.details or {},
        }


class VisionControlError(Exception):
    """Base exception for Vatsrani Vision."""
    def __init__(self, message: str, code: str = "GENERIC_ERROR", subsystem: str = "core"):
        super().__init__(message)
        self.code = code
        self.subsystem = subsystem


class ConfigurationError(VisionControlError):
    def __init__(self, message: str, code: str = "CONFIG_INVALID"):
        super().__init__(message, code=code, subsystem="config")


class HardwareError(VisionControlError):
    def __init__(self, message: str, code: str = "HARDWARE_FAILURE"):
        super().__init__(message, code=code, subsystem="arduino")


class AIProviderError(VisionControlError):
    def __init__(self, message: str, code: str = "AI_PROVIDER_FAILURE"):
        super().__init__(message, code=code, subsystem="ai")


class IntentError(VisionControlError):
    def __init__(self, message: str, code: str = "INTENT_PARSE_FAILURE"):
        super().__init__(message, code=code, subsystem="intent")
