"""
devices/serial_transport.py

Phase 6: Serial transport module for USB Serial communication with Arduino.

Responsibilities:
    - Manage Serial connection (connect, disconnect, reconnect, auto-detect port)
    - Non-blocking sending and receiving over Serial
    - Parse deterministic line-based protocol responses
    - Avoid blocking webcam loop
    - Error handling for unplugged, missing, or malformed serial data

No device-specific logic inside SerialTransport. Pure serial communication layer.
"""

import sys
import time
import logging
from dataclasses import dataclass
from typing import Optional, List

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─── Response Data Model ──────────────────────────────────────────────────────

@dataclass
class SerialResponse:
    """Structured response parsed from Arduino serial communication."""
    success: bool
    device_id: Optional[str] = None
    value: Optional[str] = None
    error: Optional[str] = None
    raw_response: str = ""

    def __repr__(self) -> str:
        if self.success:
            return f"SerialResponse[OK] device={self.device_id} val={self.value}"
        return f"SerialResponse[ERR] error={self.error} raw='{self.raw_response}'"


# ─── Protocol Parser ──────────────────────────────────────────────────────────

def parse_serial_response(raw: str) -> SerialResponse:
    """
    Parses a raw response line from Arduino.

    Examples:
        OK|LIGHT_01|80     -> SerialResponse(success=True, device_id="LIGHT_01", value="80")
        OK|SYSTEM|ALL_OFF  -> SerialResponse(success=True, device_id="SYSTEM", value="ALL_OFF")
        PONG|SYSTEM        -> SerialResponse(success=True, device_id="SYSTEM", value="PONG")
        ERR|INVALID_VALUE  -> SerialResponse(success=False, error="INVALID_VALUE")
    """
    clean = raw.strip()
    if not clean:
        return SerialResponse(success=False, error="EMPTY_RESPONSE", raw_response=raw)

    parts = clean.split("|")
    prefix = parts[0].upper()

    if prefix == "OK":
        device_id = parts[1] if len(parts) > 1 else None
        val = parts[2] if len(parts) > 2 else (parts[1] if len(parts) == 2 else None)
        return SerialResponse(success=True, device_id=device_id, value=val, raw_response=clean)

    if prefix == "PONG":
        device_id = parts[1] if len(parts) > 1 else "SYSTEM"
        return SerialResponse(success=True, device_id=device_id, value="PONG", raw_response=clean)

    if prefix == "ERR":
        err_msg = parts[1] if len(parts) > 1 else "UNKNOWN_ERROR"
        return SerialResponse(success=False, error=err_msg, raw_response=clean)

    # Fallback for unexpected format
    return SerialResponse(success=False, error="MALFORMED_RESPONSE", raw_response=clean)


# ─── Serial Transport Class ───────────────────────────────────────────────────

class SerialTransport:
    """
    Serial Transport handling hardware USB communication.
    Designed to fail gracefully and never crash the main application loop.
    """

    def __init__(
        self,
        port: str = "",
        baud_rate: int = 115200,
        timeout: float = 1.0,
        auto_detect: bool = True,
        reconnect_interval: float = 3.0,
        debug_serial: bool = True,
    ):
        self.port: str = port
        self.baud_rate: int = baud_rate
        self.timeout: float = timeout
        self.auto_detect: bool = auto_detect
        self.reconnect_interval: float = reconnect_interval
        self.debug_serial: bool = debug_serial

        self._ser: Optional[object] = None
        self._connected: bool = False
        self._last_reconnect_attempt: float = 0.0
        self._unresponsive: bool = False

    # ─── Connection Management ────────────────────────────────────────────────

    def auto_detect_port(self) -> Optional[str]:
        """
        Scan available system COM/USB ports for Arduino devices.
        Returns first matching serial port path, or None.
        """
        if not SERIAL_AVAILABLE:
            return None

        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            dev = p.device
            desc = p.description.lower()
            # macOS: /dev/cu.usbmodem*, /dev/cu.usbserial*
            # Linux: /dev/ttyACM*, /dev/ttyUSB*
            # Windows: COM3, COM4...
            if (
                "usbmodem" in dev.lower()
                or "usbserial" in dev.lower()
                or "ttyacm" in dev.lower()
                or "ttyusb" in dev.lower()
                or "arduino" in desc
                or "ch340" in desc
                or "ftdi" in desc
            ):
                return dev
        
        # Fallback to any com port if present
        if ports:
            return ports[0].device
        return None

    def connect(self) -> bool:
        """
        Attempt to establish serial connection with Arduino.
        Does not block for long periods. Returns True if connected.
        """
        if not SERIAL_AVAILABLE:
            self._connected = False
            return False

        if self.is_connected():
            return True

        target_port = self.port
        if not target_port and self.auto_detect:
            target_port = self.auto_detect_port() or ""

        if not target_port:
            self._connected = False
            return False

        try:
            self._ser = serial.Serial(
                port=target_port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            # Give Arduino time to reset on DTR toggle upon connection
            time.sleep(0.1)
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self._connected = True
            self._unresponsive = False
            self.port = target_port
            if self.debug_serial:
                print(f"[SERIAL] Connected to {target_port} @ {self.baud_rate} baud")
            return True
        except Exception as e:
            if self.debug_serial:
                print(f"[SERIAL] Failed to connect to '{target_port}': {e}")
            self._ser = None
            self._connected = False
            return False

    def disconnect(self):
        """Close serial port safely."""
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self._connected = False
        self._unresponsive = False
        if self.debug_serial:
            print("[SERIAL] Disconnected.")

    def reconnect(self) -> bool:
        """Force a disconnect and attempt reconnection."""
        self.disconnect()
        return self.connect()

    def is_connected(self) -> bool:
        """Check if transport has active serial connection."""
        if not SERIAL_AVAILABLE or self._ser is None:
            self._connected = False
            return False
        try:
            if hasattr(self._ser, "is_open"):
                self._connected = self._ser.is_open
            else:
                self._connected = True
        except Exception:
            self._connected = False
        return self._connected

    @property
    def is_unresponsive(self) -> bool:
        return self._unresponsive

    # ─── Communication ────────────────────────────────────────────────────────

    def send(self, command_str: str) -> bool:
        """
        Send a raw line-based protocol command string to Arduino.
        Ensures trailing '\\n' is present. Non-blocking short timeout.
        """
        if not self.is_connected():
            return False

        if not command_str.endswith("\n"):
            command_str += "\n"

        if self.debug_serial:
            print(f"TX → {command_str.strip()}")

        try:
            self._ser.write(command_str.encode("utf-8"))
            self._ser.flush()
            return True
        except Exception as e:
            if self.debug_serial:
                print(f"[SERIAL ERROR] Send failed: {e}")
            self.disconnect()
            return False

    def read_response(self, timeout: Optional[float] = None) -> SerialResponse:
        """
        Read line response from Arduino over serial.
        """
        if not self.is_connected():
            return SerialResponse(success=False, error="DISCONNECTED")

        old_timeout = self._ser.timeout
        if timeout is not None:
            self._ser.timeout = timeout

        try:
            line_bytes = self._ser.readline()
            if timeout is not None:
                self._ser.timeout = old_timeout

            raw_str = line_bytes.decode("utf-8", errors="replace").strip()

            if self.debug_serial and raw_str:
                print(f"RX ← {raw_str}")

            if not raw_str:
                return SerialResponse(success=False, error="TIMEOUT")

            return parse_serial_response(raw_str)
        except Exception as e:
            if timeout is not None and self._ser:
                self._ser.timeout = old_timeout
            if self.debug_serial:
                print(f"[SERIAL ERROR] Read failed: {e}")
            self.disconnect()
            return SerialResponse(success=False, error=str(e))

    def send_command_and_get_response(self, command_str: str, timeout: Optional[float] = None) -> SerialResponse:
        """
        Convenience method to send a command and immediately read response.
        """
        if not self.send(command_str):
            return SerialResponse(success=False, error="SEND_FAILED")
        return self.read_response(timeout=timeout)

    def ping(self) -> bool:
        """
        Send PING|SYSTEM handshake and verify PONG|SYSTEM response.
        """
        resp = self.send_command_and_get_response("PING|SYSTEM\n", timeout=1.0)
        if resp.success and (resp.value == "PONG" or resp.raw_response.startswith("PONG")):
            self._unresponsive = False
            return True
        if self.is_connected():
            self._unresponsive = True
        return False
