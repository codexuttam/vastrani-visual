"""
ui/hud.py

VisionControl Unified Polished HUD & Presentation Engine.
Phase 11 — Polished UI / HUD + System Visualization.

Integrates:
  - System Status Header & Mode Bar
  - Camera Viewport & Failure Recovery Panel
  - Hand Tracking & Finger State Matrix Visualizer
  - Face Tracking State Panel
  - Phase 10 AI Intent Engine Panel
  - High-Priority Confirmation Dialog Modal Card
  - Connected Devices Panel & Bottom Device Navigation Rail
  - Arduino Hardware Connection & Telemetry Panel
  - Recent Command History Panel
  - Real-Time System Event Feed Stream
  - Performance & Pipeline Latency Monitor
  - Toast Notification Manager & Overlay
"""

import cv2
import numpy as np
import time
from typing import Optional, List, Any

from gestures.features import HandFeatureSet
from gestures.types import (
    GestureResult, GestureEvent, GestureType,
    StateMachineState, ControlMode,
)
from devices.models import DeviceState, DeviceType, CommandRecord
from devices.registry import DeviceRegistry

from ui.contracts import (
    SystemStatus, HUDMode, NotificationCategory, NotificationItem, EventItem, PerformanceMetrics,
)
from ui.design_system import (
    CYAN, VIOLET, WHITE, DARK_BG, PANEL_BG, PANEL_BORDER, GREEN_SUCCESS, RED_ERROR, ORANGE_WARN, YELLOW_DEGRADED, FONT_PRIMARY, FONT_MONO,
    draw_glass_panel, draw_status_badge, draw_badge, sanitize_text,
)
from ui.notifications import NotificationManager
from ui.event_stream import EventStreamManager
from ui.performance import PerformanceMonitor

from ui.components.system_panel import SystemPanelComponent
from ui.components.vision_panel import VisionPanelComponent
from ui.components.hand_panel import HandPanelComponent
from ui.components.face_panel import FacePanelComponent
from ui.components.intent_panel import IntentPanelComponent
from ui.components.confirmation_panel import ConfirmationDialogComponent
from ui.components.device_panel import DevicePanelComponent
from ui.components.arduino_panel import ArduinoPanelComponent
from ui.components.history_panel import HistoryPanelComponent
from ui.components.event_stream_panel import EventStreamPanelComponent
from ui.components.performance_panel import PerformancePanelComponent
from ui.components.notification_panel import NotificationPanelComponent


class HUD:
    """Unified Polished VisionControl HUD Coordinator."""

    def __init__(self, mode: HUDMode = HUDMode.STANDARD, reduced_motion: bool = False):
        self.hud_mode = mode
        self.hud_visible = True
        self.reduced_motion = reduced_motion

        # Observability Managers
        self.notification_manager = NotificationManager()
        self.event_stream = EventStreamManager()
        self.performance_monitor = PerformanceMonitor()

        # Modular UI Components
        self.system_panel = SystemPanelComponent()
        self.vision_panel = VisionPanelComponent()
        self.hand_panel = HandPanelComponent()
        self.face_panel = FacePanelComponent()
        self.intent_panel = IntentPanelComponent()
        self.confirmation_dialog = ConfirmationDialogComponent()
        self.device_panel = DevicePanelComponent()
        self.arduino_panel = ArduinoPanelComponent()
        self.history_panel = HistoryPanelComponent()
        self.event_stream_panel = EventStreamPanelComponent()
        self.performance_panel = PerformancePanelComponent()
        self.notification_panel = NotificationPanelComponent()

        # State tracking for transient event banners
        self._last_gesture: Optional[str] = None
        self._last_intent_id: Optional[str] = None
        self._camera_offline = False

    def toggle_mode(self):
        """Toggles between Standard and Developer HUD modes."""
        self.hud_mode = HUDMode.DEVELOPER if self.hud_mode == HUDMode.STANDARD else HUDMode.STANDARD
        self.event_stream.log("HUD", f"Mode switched to {self.hud_mode.value}", "INFO")

    def toggle_visibility(self):
        """Toggles HUD display visibility on/off."""
        self.hud_visible = not self.hud_visible

    def post_notification(self, category: NotificationCategory, title: str, message: str, duration: float = 4.0):
        """Posts a toast notification to the HUD."""
        self.notification_manager.add(category, title, message, duration)

    def log_event(self, source: str, message: str, level: str = "INFO"):
        """Logs a system event into the real-time event feed."""
        self.event_stream.log(source, message, level)

    def handle_key_event(self, key: int) -> Optional[str]:
        """
        Handles HUD keyboard shortcuts:
          - 'd': Toggle Developer Mode
          - 'h': Toggle HUD Visibility
          - 'r': Reset telemetry and clear logs
          - 'c': Toggle camera recovery state
        """
        if key in (ord('d'), ord('D')):
            self.toggle_mode()
            return "TOGGLE_MODE"
        elif key in (ord('h'), ord('H')):
            self.toggle_visibility()
            return "TOGGLE_HUD"
        elif key in (ord('r'), ord('R')):
            self.event_stream.clear()
            self.notification_manager.clear()
            self.event_stream.log("HUD", "Telemetry logs reset", "INFO")
            return "RESET"
        elif key in (ord('c'), ord('C')):
            self._camera_offline = not self._camera_offline
            return "TOGGLE_CAM"
        return None

    def _determine_system_status(
        self,
        camera_ok: bool,
        ai_status: str,
        arduino_connected: bool,
        arduino_unresponsive: bool,
        device_mode: str,
    ) -> SystemStatus:
        """Determines unified system status enum."""
        if not camera_ok or self._camera_offline:
            return SystemStatus.CAMERA_OFFLINE
        if device_mode == "HARDWARE" and (arduino_unresponsive or not arduino_connected):
            return SystemStatus.ARDUINO_DISCONNECTED
        if ai_status in ("ERROR", "TIMEOUT", "REJECTED"):
            return SystemStatus.AI_OFFLINE
        if ai_status == "LOW_CONFIDENCE":
            return SystemStatus.DEGRADED
        return SystemStatus.ONLINE

    def render(
        self,
        frame: np.ndarray,
        fps: float,
        features: Optional[HandFeatureSet] = None,
        result: Optional[GestureResult] = None,
        event: Optional[GestureEvent] = None,
        mode: ControlMode = ControlMode.IDLE,
        debug_features: bool = False,
        debug_gestures: bool = False,
        # ── Phase 5 & 6 ──────────────────────────────────────────────────────
        registry: Optional[DeviceRegistry] = None,
        pending_action: Optional[str] = None,
        debug_devices: bool = True,
        command_history: Optional[List[CommandRecord]] = None,
        device_mode: str = "VIRTUAL",
        arduino_connected: bool = False,
        arduino_unresponsive: bool = False,
        # ── Phase 7 ──────────────────────────────────────────────────────────
        face_state: Optional[object] = None,
        debug_face: bool = False,
        face_inference_ms: float = 0.0,
        # ── Phase 8 ──────────────────────────────────────────────────────────
        ar_state: Optional[object] = None,
        debug_ar: bool = False,
        # ── Phase 9 ──────────────────────────────────────────────────────────
        ai_status: str = "READY",
        ai_mode: str = "EVENT",
        last_ai_record: Optional[object] = None,
        debug_ai: bool = True,
        # ── Phase 10 ─────────────────────────────────────────────────────────
        intent_result: Optional[object] = None,
        intent_pending: bool = False,
        intent_provider: str = "fallback",
        debug_intent: bool = False,
    ) -> np.ndarray:

        h, w, _ = frame.shape

        # 1. Update Performance Telemetry
        metrics = self.performance_monitor.update_frame(
            fps=fps,
            vision_ms=1000.0 / max(1.0, fps),
            face_ms=face_inference_ms,
            intent_ms=240.0 if intent_result else 0.0,
            serial_ms=5.0 if arduino_connected else 0.0,
            frame_width=w,
            frame_height=h,
        )

        # Log gesture change events dynamically
        if result and result.active and result.gesture.value != self._last_gesture:
            self._last_gesture = result.gesture.value
            if self._last_gesture != "IDLE":
                self.log_event("GESTURE", f"Detected {self._last_gesture}", "SUCCESS")
                self.post_notification(NotificationCategory.INFO, "Gesture Detected", f"Recognized {self._last_gesture}")

        # Determine System Status
        sys_status = self._determine_system_status(
            camera_ok=not self._camera_offline,
            ai_status=ai_status,
            arduino_connected=arduino_connected,
            arduino_unresponsive=arduino_unresponsive,
            device_mode=device_mode,
        )

        if not self.hud_visible:
            return frame

        # 2. Camera Offline Overlay check
        if sys_status == SystemStatus.CAMERA_OFFLINE:
            self.vision_panel.render_offline_overlay(frame, "Camera disconnect or failure detected.")

        # 3. Persistent Header & System Status Bar
        self.system_panel.render(
            frame,
            status=sys_status,
            mode=self.hud_mode,
            camera_live=(sys_status != SystemStatus.CAMERA_OFFLINE),
            reduced_motion=self.reduced_motion,
        )

        # 4. Viewport No-Hand Hint
        if (features is None or not features.valid) and sys_status == SystemStatus.ONLINE:
            self.vision_panel.render_no_hand_hint(frame)

        # 5. Hand Tracking Status Panel (Left Column, Top)
        self.hand_panel.render(
            frame,
            x=15,
            y=62,
            w=270,
            h=165,
            features=features,
            result=result,
            mode=mode,
        )

        # 6. Face Tracking Status Panel (Left Column, Middle)
        if debug_face or face_state is not None:
            self.face_panel.render(
                frame,
                x=15,
                y=235,
                w=270,
                h=140,
                face_state=face_state,
                face_infer_ms=face_inference_ms,
            )

        # 7. AI Intent Engine Panel (Left Column, Bottom)
        self.intent_panel.render(
            frame,
            x=15,
            y=383,
            w=270,
            h=165,
            intent_result=intent_result,
            intent_pending=intent_pending,
            provider_name=intent_provider,
        )

        # 8. Right Column: Device Status Panel & History
        if registry is not None:
            self.device_panel.render(
                frame,
                x=w - 295,
                y=62,
                w=280,
                h=165,
                registry=registry,
                pending_action=pending_action,
                device_mode=device_mode,
            )

        # 9. Arduino Telemetry Panel (Right Column, Middle)
        self.arduino_panel.render(
            frame,
            x=w - 295,
            y=235,
            w=280,
            h=140,
            connected=arduino_connected,
            unresponsive=arduino_unresponsive,
            device_mode=device_mode,
        )

        # 10. Command History Panel (Right Column, Bottom)
        self.history_panel.render(
            frame,
            x=w - 295,
            y=383,
            w=280,
            h=165,
            command_history=command_history,
        )

        # 11. Developer Mode Overlay: Real-Time Event Feed & Performance Monitor
        if self.hud_mode == HUDMode.DEVELOPER:
            self.event_stream_panel.render(
                frame,
                x=295,
                y=h - 180,
                w=340,
                h=125,
                events=self.event_stream.get_recent(6),
            )
            self.performance_panel.render(
                frame,
                x=645,
                y=h - 180,
                w=300,
                h=125,
                metrics=metrics,
            )

        # 12. Bottom Navigation Rail
        if registry is not None:
            self.device_panel.render_rail(frame, registry)

        # 13. High-Priority Confirmation Dialog Modal Card
        if intent_pending:
            self.confirmation_dialog.render(
                frame,
                intent_result=intent_result,
                pending_action=pending_action,
            )

        # 14. Active Toast Notifications Overlay (Top-Right)
        toasts = self.notification_manager.get_active()
        self.notification_panel.render(frame, toasts, top_y=65, right_margin=305)

        return frame
