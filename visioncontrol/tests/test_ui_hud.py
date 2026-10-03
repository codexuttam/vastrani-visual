"""
tests/test_ui_hud.py

Unit and Integration Test Suite for Phase 11 — Polished UI / HUD + System Visualization.
"""

import pytest
import numpy as np
import time
from typing import Optional

from ui.contracts import (
    SystemStatus, HUDMode, NotificationCategory, NotificationItem, EventItem, PerformanceMetrics,
)
from ui.design_system import (
    sanitize_text, draw_glass_panel, draw_status_badge, draw_badge, draw_progress_bar, get_pulse_alpha,
)
from ui.notifications import NotificationManager
from ui.event_stream import EventStreamManager
from ui.performance import PerformanceMonitor
from ui.hud import HUD

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

from devices.registry import DeviceRegistry
from devices.virtual_device import VirtualDevice
from devices.models import DeviceType, CommandRecord, DeviceState


# ─── 1. Design System Tests ───────────────────────────────────────────────────

def test_sanitize_text_secrets():
    assert sanitize_text("sk-proj-1234567890") == "sk-***"
    assert sanitize_text("hello world", max_length=8) == "hello..."
    assert sanitize_text("") == ""


def test_design_system_primitives():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Panel
    draw_glass_panel(frame, 10, 10, 200, 100, title="TEST")
    # Badge
    w1 = draw_status_badge(frame, 10, 120, SystemStatus.ONLINE)
    assert w1 > 0
    w2 = draw_badge(frame, 10, 160, "ACTIVE", (0, 200, 100))
    assert w2 > 0
    # Progress Bar
    draw_progress_bar(frame, 10, 200, 100, 10, progress=0.75)

    # Pulse alpha
    alpha = get_pulse_alpha(cycle_sec=1.0, reduced_motion=False)
    assert 0.5 <= alpha <= 1.0
    alpha_reduced = get_pulse_alpha(cycle_sec=1.0, reduced_motion=True)
    assert alpha_reduced == 1.0


# ─── 2. Notification Manager Tests ────────────────────────────────────────────

def test_notification_manager():
    mgr = NotificationManager(max_notifications=3, default_duration=2.0)
    mgr.info("System", "Started successfully")
    mgr.warning("Arduino", "Connection unstable")
    mgr.error("Camera", "Frame drop")

    active = mgr.get_active()
    assert len(active) == 3
    assert active[0].title == "System"

    # Test dismissal
    mgr.dismiss(active[0].id)
    assert len(mgr.get_active()) == 2

    # Test clear
    mgr.clear()
    assert len(mgr.get_active()) == 0


# ─── 3. Event Stream Manager Tests ────────────────────────────────────────────

def test_event_stream_manager():
    stream = EventStreamManager(max_events=5)
    for i in range(10):
        stream.log("GESTURE", f"Gesture {i}", "INFO")

    recent = stream.get_recent(limit=3)
    assert len(recent) == 3
    assert recent[-1].message == "Gesture 9"

    stream.clear()
    assert len(stream.get_recent()) == 0


# ─── 4. Performance Monitor Tests ─────────────────────────────────────────────

def test_performance_monitor():
    pm = PerformanceMonitor(smoothing_alpha=0.5)
    m1 = pm.update_frame(fps=30.0, vision_ms=15.0, face_ms=5.0, intent_ms=100.0)
    assert m1.fps == 30.0
    assert m1.vision_ms == 15.0
    assert m1.face_ms == 5.0

    m2 = pm.update_frame(fps=60.0)
    assert m2.fps == 45.0  # 0.5*60 + 0.5*30


# ─── 5. HUD System Status Calculation Tests ───────────────────────────────────

def test_hud_system_status_determination():
    hud = HUD()
    # Online
    status = hud._determine_system_status(True, "READY", True, False, "VIRTUAL")
    assert status == SystemStatus.ONLINE

    # Camera offline
    status = hud._determine_system_status(False, "READY", True, False, "VIRTUAL")
    assert status == SystemStatus.CAMERA_OFFLINE

    # Arduino disconnected in hardware mode
    status = hud._determine_system_status(True, "READY", False, False, "HARDWARE")
    assert status == SystemStatus.ARDUINO_DISCONNECTED

    # AI Offline
    status = hud._determine_system_status(True, "ERROR", True, False, "VIRTUAL")
    assert status == SystemStatus.AI_OFFLINE


# ─── 6. Keyboard Shortcut Handler Tests ───────────────────────────────────────

def test_hud_keyboard_shortcuts():
    hud = HUD(mode=HUDMode.STANDARD)
    assert hud.hud_mode == HUDMode.STANDARD

    res_d = hud.handle_key_event(ord('d'))
    assert res_d == "TOGGLE_MODE"
    assert hud.hud_mode == HUDMode.DEVELOPER

    res_h = hud.handle_key_event(ord('h'))
    assert res_h == "TOGGLE_HUD"
    assert hud.hud_visible is False

    res_r = hud.handle_key_event(ord('r'))
    assert res_r == "RESET"

    res_c = hud.handle_key_event(ord('c'))
    assert res_c == "TOGGLE_CAM"


# ─── 7. Full HUD Rendering Integration Tests ──────────────────────────────────

def test_hud_rendering_full_pipeline():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    hud = HUD(mode=HUDMode.STANDARD)

    # Setup device registry mock
    registry = DeviceRegistry()
    state = DeviceState(device_id="fan_1", device_type=DeviceType.FAN, name="Bedroom Fan")
    dev = registry.register(state)

    # Command history mock
    rec = CommandRecord(device_id="fan_1", action="TURN_ON", timestamp=time.time(), virtual_success=True)
    history = [rec]

    # Render frame
    rendered = hud.render(
        frame=frame,
        fps=30.0,
        mode=HUDMode.STANDARD,
        registry=registry,
        command_history=history,
        device_mode="VIRTUAL",
        arduino_connected=True,
    )

    assert rendered is not None
    assert rendered.shape == (720, 1280, 3)


def test_hud_rendering_developer_mode():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    hud = HUD(mode=HUDMode.DEVELOPER)

    rendered = hud.render(
        frame=frame,
        fps=60.0,
        ai_status="READY",
        intent_provider="auto",
    )
    assert rendered.shape == (720, 1280, 3)


def test_hud_confirmation_dialog_rendering():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    hud = HUD()

    # Render with pending intent confirmation
    rendered = hud.render(
        frame=frame,
        fps=30.0,
        intent_pending=True,
        pending_action="TURN_ON_LIGHT",
    )
    assert rendered.shape == (720, 1280, 3)


def test_all_components_isolation():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    SystemPanelComponent().render(frame, SystemStatus.ONLINE, HUDMode.STANDARD)
    VisionPanelComponent().render_offline_overlay(frame, "Test message")
    HandPanelComponent().render(frame, 10, 10, 200, 150)
    FacePanelComponent().render(frame, 10, 10, 200, 150)
    IntentPanelComponent().render(frame, 10, 10, 200, 150)
    ConfirmationDialogComponent().render(frame, pending_action="TEST_ACTION")
    DevicePanelComponent().render(frame, 10, 10, 200, 150)
    ArduinoPanelComponent().render(frame, 10, 10, 200, 150)
    HistoryPanelComponent().render(frame, 10, 10, 200, 150)
    EventStreamPanelComponent().render(frame, 10, 10, 200, 150, events=[])
    PerformancePanelComponent().render(frame, 10, 10, 200, 150, metrics=PerformanceMetrics())
    NotificationPanelComponent().render(frame, notifications=[])
