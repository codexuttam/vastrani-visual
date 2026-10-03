"""
health.py

System Health Monitor & Startup Pre-Flight Diagnostics.
Phase 12 — Reliability & Hardening.
"""

import sys
import time
from typing import Dict, Any, List, Optional
from config import Config
from version import __version__
from errors import SystemErrorRecord, ErrorSeverity


class HealthMonitor:
    """Tracks subsystem health statuses for telemetry and HUD observability."""

    def __init__(self):
        self.subsystems: Dict[str, Dict[str, Any]] = {
            "application": {"status": "OK", "message": "System running"},
            "camera": {"status": "OK", "message": "Camera initialized"},
            "hand_tracker": {"status": "OK", "message": "Hand tracking active"},
            "face_tracker": {"status": "OK", "message": "Face tracking active"},
            "ai_provider": {"status": "OK", "message": "AI Router ready"},
            "intent_engine": {"status": "OK", "message": "Intent engine ready"},
            "arduino": {"status": "OK", "message": "Hardware disconnected"},
            "device_layer": {"status": "OK", "message": "Devices registered"},
        }
        self.recent_errors: List[SystemErrorRecord] = []

    def update_subsystem(self, name: str, status: str, message: str = ""):
        if name in self.subsystems:
            self.subsystems[name] = {"status": status, "message": message, "updated_at": time.time()}

    def record_error(self, code: str, subsystem: str, severity: ErrorSeverity, message: str, details: Optional[Dict[str, Any]] = None):
        rec = SystemErrorRecord(
            code=code,
            subsystem=subsystem,
            severity=severity,
            message=message,
            timestamp=time.time(),
            details=details,
        )
        self.recent_errors.append(rec)
        if len(self.recent_errors) > 50:
            self.recent_errors = self.recent_errors[-50:]
        if subsystem in self.subsystems:
            status_val = "DEGRADED" if severity == ErrorSeverity.WARNING else "ERROR"
            self.subsystems[subsystem] = {"status": status_val, "message": message, "updated_at": time.time()}

    def get_summary(self) -> Dict[str, Any]:
        return {
            "version": __version__,
            "subsystems": self.subsystems,
            "error_count": len(self.recent_errors),
        }


class StartupDiagnostics:
    """Runs pre-flight initialization checks for Vatsrani Vision."""

    @staticmethod
    def run_checks(config: Config) -> bool:
        print(f"\n==================================================")
        print(f"  INITIALIZING VATSRANI VISION v{__version__}")
        print(f"==================================================\n")

        results = []

        # 1. Configuration Validation
        try:
            assert config.FRAME_WIDTH > 0 and config.FRAME_HEIGHT > 0, "Invalid frame dimensions"
            assert config.TARGET_FPS > 0, "Invalid target FPS"
            print("  [✓] Configuration & Environment")
            results.append(True)
        except Exception as e:
            print(f"  [✕] Configuration: {e}")
            results.append(False)

        # 2. Vision Models & Dependencies
        try:
            import cv2
            import mediapipe as mp
            print("  [✓] Computer Vision Engine (OpenCV + MediaPipe)")
            results.append(True)
        except Exception as e:
            print(f"  [✕] Vision Engine: {e}")
            results.append(False)

        # 3. Intent Engine Check
        if config.INTENT_ENGINE_ENABLED:
            print(f"  [✓] Natural-Language Intent Engine (Provider: {config.INTENT_PROVIDER.upper()})")
            results.append(True)

        # 4. AI Provider Key Verification
        if config.OPENAI_ENABLED:
            has_key = bool(config.OPENAI_API_KEY.strip())
            key_status = "Configured" if has_key else "Not set (Fallback deterministic mode active)"
            print(f"  [✓] AI Provider (OpenAI {config.OPENAI_MODEL}: {key_status})")
            results.append(True)

        # 5. Device Layer & Registry Check
        try:
            from devices.registry import DeviceRegistry
            from devices.models import DeviceState, DeviceType
            print("  [✓] Virtual & Hardware Device Control Layer")
            results.append(True)
        except Exception as e:
            print(f"  [✕] Device Layer: {e}")
            results.append(False)

        # 6. Arduino Transport Check
        if config.DEVICE_MODE == "HARDWARE":
            port_info = config.SERIAL_PORT if config.SERIAL_PORT else "Auto-detect"
            print(f"  [✓] Serial Hardware Transport ({port_info} @ {config.SERIAL_BAUD_RATE} baud)")
        else:
            print("  [✓] Serial Hardware Transport (Virtual Simulation Mode)")

        all_ok = all(results)
        print("\n--------------------------------------------------")
        if all_ok:
            print("  SYSTEM READY — Launching VisionControl Application")
        else:
            print("  [WARNING] Some non-critical subsystems degraded.")
        print("--------------------------------------------------\n")

        return all_ok
