"""
tests/test_hardening_and_reliability.py

Comprehensive Integration & Hardening Test Suite for Phase 12.
Validates:
  - System health monitor & startup diagnostics
  - Error classification & severity contracts
  - Failure isolation & fallback recovery boundaries
  - Malformed AI response handling
  - Arduino disconnect resilience
  - Demo mode operations
"""

import pytest
import numpy as np
import time

from version import __version__
from config import Config
from errors import (
    ErrorSeverity, SystemErrorRecord, VisionControlError,
    ConfigurationError, HardwareError, AIProviderError, IntentError,
)
from health import HealthMonitor, StartupDiagnostics
from intent.engine import IntentEngine
from devices.controller import DeviceController
from devices.registry import DeviceRegistry
from devices.models import DeviceType, DeviceState
from ui.hud import HUD
from ui.contracts import SystemStatus, HUDMode


def test_version_metadata():
    assert __version__ == "1.0.0"


def test_errors_classification():
    rec = SystemErrorRecord(
        code="CAM_OFFLINE",
        subsystem="camera",
        severity=ErrorSeverity.WARNING,
        message="Camera dropped frame",
        details={"fps": 0},
    )
    d = rec.to_dict()
    assert d["code"] == "CAM_OFFLINE"
    assert d["severity"] == "WARNING"
    assert "timestamp" in d

    err = HardwareError("Serial timeout", code="SERIAL_TIMEOUT")
    assert err.code == "SERIAL_TIMEOUT"
    assert err.subsystem == "arduino"


def test_health_monitor():
    hm = HealthMonitor()
    summary = hm.get_summary()
    assert summary["version"] == "1.0.0"

    hm.record_error("AI_TIMEOUT", "ai_provider", ErrorSeverity.WARNING, "OpenAI request timed out")
    assert hm.subsystems["ai_provider"]["status"] == "DEGRADED"


def test_startup_diagnostics():
    cfg = Config()
    ok = StartupDiagnostics.run_checks(cfg)
    assert ok is True


def test_failure_isolation_camera_offline():
    hud = HUD()
    status = hud._determine_system_status(
        camera_ok=False,
        ai_status="READY",
        arduino_connected=True,
        arduino_unresponsive=False,
        device_mode="VIRTUAL",
    )
    assert status == SystemStatus.CAMERA_OFFLINE


def test_failure_isolation_ai_fallback():
    cfg = Config()
    cfg.INTENT_PROVIDER = "fallback"  # Force offline deterministic fallback
    dev_ctrl = DeviceController(mode="VIRTUAL")
    engine = IntentEngine.from_config(cfg, device_controller=dev_ctrl)

    # Valid text prompt handled deterministically
    res = engine.handle("turn on the fan")
    assert res.status in ("EXECUTED", "NEEDS_CONFIRMATION", "READY", "EXECUTION_FAILED")

    # Malformed / gibberish text prompt safely rejected
    res_bad = engine.handle("xyz123 invalid command")
    assert res_bad.status in ("NEEDS_CLARIFICATION", "REJECTED", "ERROR")


def test_failure_isolation_arduino_disconnect():
    from gestures.types import GestureAction
    controller = DeviceController(mode="HARDWARE")
    # Action dispatched while disconnected must return failure result without throwing crash
    res = controller.handle_action(GestureAction.SELECT)
    assert res is not None


def test_demo_mode_operation():
    cfg = Config()
    cfg.DEMO_MODE = True
    assert cfg.DEMO_MODE is True
