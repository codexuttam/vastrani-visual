"""
ai/router.py

Phase 9: AI Event Router & Asynchronous Processing Engine.

Responsibilities:
    - Decide when OpenAI should be called (Event-driven, AI_MODE=EVENT vs OFF)
    - Run OpenAI requests asynchronously in background thread to preserve 30+ FPS camera loop
    - Enforce dual-layer safety validation before executing device commands
    - Guarantee FIST Emergency Stop executes 100% locally without network latency
    - Pass structured AIResultRecords to main thread for HUD updates
"""

import time
import queue
import threading
from typing import Optional, List, Tuple

from ai.schemas import AIIntent, AIResultRecord
from ai.openai_client import OpenAIClient
from ai.safety import AISafetyValidator
from ai.intent import map_ai_intent_to_gesture_action
from devices.controller import DeviceController
from devices.registry import DeviceRegistry


class AIRouter:
    """
    Asynchronous Event Router for selective OpenAI Intelligence.
    Guarantees camera loop responsiveness.
    """

    def __init__(
        self,
        ai_mode: str = "EVENT",
        client: Optional[OpenAIClient] = None,
        safety_validator: Optional[AISafetyValidator] = None,
        min_confidence: float = 0.75,
    ):
        self.ai_mode: str = ai_mode.upper()
        self.client: OpenAIClient = client or OpenAIClient(enabled=(self.ai_mode != "OFF"))
        self.safety_validator: AISafetyValidator = safety_validator or AISafetyValidator(min_confidence=min_confidence)

        self.status: str = "DISABLED" if self.ai_mode == "OFF" else "READY"
        self.last_record: Optional[AIResultRecord] = None

        # Asynchronous Queue & Thread
        self._request_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self.ai_mode != "OFF":
            self._start_worker()

    def _start_worker(self):
        """Start background worker thread for non-blocking OpenAI processing."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        """Worker thread processing queued AI requests."""
        while not self._stop_event.is_set():
            try:
                task = self._request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            text_input, registry, device_controller = task
            self.status = "PROCESSING"

            record = self.process_text_command_sync(text_input, registry, device_controller)
            self.last_record = record
            self.status = record.status if record.status in ("SUCCESS", "READY", "DISABLED") else record.status

            self._result_queue.put(record)
            self._request_queue.task_done()

            # Set status back to READY after processing
            if self.status not in ("DISABLED", "ERROR", "TIMEOUT"):
                self.status = "READY"

    def process_text_command_async(
        self,
        text_input: str,
        registry: DeviceRegistry,
        device_controller: Optional[DeviceController] = None,
    ) -> bool:
        """
        Non-blocking entry point to queue a natural language command for AI processing.
        """
        if self.ai_mode == "OFF" or not self.client.enabled:
            self.status = "DISABLED"
            return False

        self._request_queue.put((text_input, registry, device_controller))
        self.status = "PROCESSING"
        return True

    def process_text_command_sync(
        self,
        text_input: str,
        registry: DeviceRegistry,
        device_controller: Optional[DeviceController] = None,
    ) -> AIResultRecord:
        """
        Synchronous processing method (used by worker thread and developer unit tests).
        """
        t0 = time.time()
        if self.ai_mode == "OFF":
            return AIResultRecord(
                intent="NONE",
                device_id=None,
                value=None,
                confidence=0.0,
                reason="AI_MODE is OFF.",
                timestamp=t0,
                status="DISABLED",
                input_text=text_input,
            )

        # 1. Call OpenAI Client
        intent, client_status, error_msg = self.client.interpret(text_input, registry)

        if not intent:
            status_code = client_status if client_status in ("TIMEOUT", "DISABLED", "RATE_LIMITED") else "ERROR"
            return AIResultRecord(
                intent="NONE",
                device_id=None,
                value=None,
                confidence=0.0,
                reason=error_msg or "Interpretation failed.",
                timestamp=t0,
                status=status_code,
                error=error_msg,
                input_text=text_input,
            )

        # 2. Safety Layer Validation
        is_safe, safety_msg = self.safety_validator.validate(intent, registry)
        if not is_safe:
            status_code = "LOW_CONFIDENCE" if "Low confidence" in safety_msg else "REJECTED"
            return AIResultRecord(
                intent=intent.intent,
                device_id=intent.device_id,
                value=intent.value,
                confidence=intent.confidence,
                reason=safety_msg,
                timestamp=t0,
                status=status_code,
                error=safety_msg,
                input_text=text_input,
            )

        # 3. Execute Validated Command on DeviceController (if provided)
        if device_controller is not None:
            it = intent.intent.upper()
            if it == "TURN_ON" and intent.device_id:
                device_controller.power_on(intent.device_id)
            elif it == "TURN_OFF" and intent.device_id:
                device_controller.power_off(intent.device_id)
            elif it == "TOGGLE" and intent.device_id:
                device_controller.toggle(intent.device_id)
            elif it == "SET_LEVEL" and intent.device_id and intent.value is not None:
                device_controller.set_level(intent.device_id, int(intent.value))
            elif it == "NEXT":
                device_controller._next()
            elif it == "PREVIOUS":
                device_controller._previous()
            elif it in ("STOP", "EMERGENCY_STOP"):
                device_controller._emergency_stop()

        return AIResultRecord(
            intent=intent.intent,
            device_id=intent.device_id,
            value=intent.value,
            confidence=intent.confidence,
            reason=intent.reason or "Passed validation and executed.",
            timestamp=t0,
            status="SUCCESS",
            input_text=text_input,
        )

    def check_completed_results(self) -> List[AIResultRecord]:
        """
        Polls completed AI result records from the result queue (non-blocking).
        """
        results = []
        while not self._result_queue.empty():
            try:
                res = self._result_queue.get_nowait()
                results.append(res)
            except queue.Empty:
                break
        return results

    def stop(self):
        """Shutdown background worker thread cleanly."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
