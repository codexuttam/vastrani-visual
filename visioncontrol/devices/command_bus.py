"""
devices/command_bus.py

Phase 6: Command Bus & Serial Translation Layer.

Architecture:
    Device Controller
           ↓
      Command Bus
      ├── Virtual Device (updates virtual state)
      └── Serial Transport (transmits to Arduino if HARDWARE mode)

Dual Modes:
    - VIRTUAL  : Commands only modify virtual devices.
    - HARDWARE : Commands modify virtual state AND transmit to Arduino.
                 If Arduino is disconnected, report status without crashing.
"""

from typing import Optional, Tuple
from devices.serial_transport import SerialTransport, SerialResponse


# ─── Command Translation Layer ────────────────────────────────────────────────

def device_command_to_serial(device_id: str, action: str, value: Optional[int] = None) -> str:
    """
    Translates an abstract device action into a deterministic line protocol command for Arduino.

    Examples:
        LIGHT_01 + POWER_ON       -> ON|LIGHT_01
        LIGHT_01 + POWER_OFF      -> OFF|LIGHT_01
        LIGHT_01 + SET_LEVEL 80   -> SET|LIGHT_01|80
        FAN_01 + SET_LEVEL 2      -> SET|FAN_01|2
        SERVO_01 + SET_LEVEL 90   -> SET|SERVO_01|90
        EMERGENCY_STOP            -> ALL_OFF|SYSTEM
        PING                      -> PING|SYSTEM
    """
    if action == "EMERGENCY_STOP":
        return "ALL_OFF|SYSTEM"

    if action == "PING":
        return "PING|SYSTEM"

    if action == "POWER_ON":
        return f"ON|{device_id}"

    if action == "POWER_OFF":
        return f"OFF|{device_id}"

    if action == "TOGGLE":
        return f"TOGGLE|{device_id}"

    if action == "SET_LEVEL":
        val_str = str(value) if value is not None else "0"
        return f"SET|{device_id}|{val_str}"

    return f"{action}|{device_id}"


# ─── Command Bus Class ────────────────────────────────────────────────────────

class CommandBus:
    """
    Command Bus that routes validated commands to virtual devices and/or physical Arduino.
    """

    def __init__(
        self,
        mode: str = "VIRTUAL",
        transport: Optional[SerialTransport] = None,
    ):
        self.mode: str = mode.upper()
        self.transport: SerialTransport = transport or SerialTransport()

    def set_mode(self, mode: str):
        """Switch between VIRTUAL and HARDWARE modes."""
        self.mode = mode.upper()

    @property
    def is_hardware_mode(self) -> bool:
        return self.mode == "HARDWARE"

    @property
    def is_connected(self) -> bool:
        return self.transport.is_connected()

    def send(
        self,
        device_id: str,
        action: str,
        value: Optional[int] = None,
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        Transmits a validated command based on mode.

        Returns:
            (hardware_sent, hardware_ack, error_message)
        """
        # If in VIRTUAL mode, hardware is not used
        if not self.is_hardware_mode:
            return False, False, None

        # Check connection status
        if not self.transport.is_connected():
            # Attempt non-blocking reconnect
            self.transport.connect()

        if not self.transport.is_connected():
            return False, False, "Arduino: DISCONNECTED"

        # Translate command
        serial_cmd = device_command_to_serial(device_id, action, value)

        # Transmit command & read response
        response: SerialResponse = self.transport.send_command_and_get_response(serial_cmd, timeout=0.5)

        if response.success:
            return True, True, None
        else:
            err_msg = response.error or "Arduino ACK failed"
            return True, False, err_msg
