"""
devices/__init__.py

Phase 5: Public API for the devices package.
"""

from .models import DeviceState, DeviceType, CommandResult, CommandRecord
from .virtual_device import VirtualDevice
from .registry import DeviceRegistry, create_default_registry
from .validator import DeviceValidator
from .controller import DeviceController
from .serial_transport import SerialTransport, SerialResponse, parse_serial_response
from .command_bus import CommandBus, device_command_to_serial

__all__ = [
    "DeviceState",
    "DeviceType",
    "CommandResult",
    "CommandRecord",
    "VirtualDevice",
    "DeviceRegistry",
    "create_default_registry",
    "DeviceValidator",
    "DeviceController",
    "SerialTransport",
    "SerialResponse",
    "parse_serial_response",
    "CommandBus",
    "device_command_to_serial",
]
