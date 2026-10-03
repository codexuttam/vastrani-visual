"""
ui/event_stream.py

Lightweight real-time event stream feed manager.
Buffers real-time system events with timestamping and severity levels.
"""

import time
import threading
from typing import List
from ui.contracts import EventItem


class EventStreamManager:
    """Buffers real-time events for HUD display and observability."""

    def __init__(self, max_events: int = 50):
        self.max_events = max_events
        self._events: List[EventItem] = []
        self._lock = threading.Lock()
        self._counter = 0

    def log(self, source: str, message: str, level: str = "INFO") -> EventItem:
        """Logs a new event into the feed."""
        with self._lock:
            self._counter += 1
            now = time.time()
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(now))
            item = EventItem(
                id=f"event_{self._counter}",
                timestamp_str=timestamp_str,
                source=source.upper(),
                message=message,
                level=level.upper(),
                timestamp=now,
            )
            self._events.append(item)
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]
            return item

    def get_recent(self, limit: int = 10) -> List[EventItem]:
        """Returns the most recent events."""
        with self._lock:
            return list(self._events[-limit:])

    def clear(self):
        """Clears all buffered events."""
        with self._lock:
            self._events.clear()
