"""
ui/notifications.py

Thread-safe notification and toast manager for VisionControl.
Supports auto-dismissal, manual dismissal, max queue limits,
and category filtering.
"""

import time
import threading
from typing import List, Optional
from ui.contracts import NotificationItem, NotificationCategory


class NotificationManager:
    """Manages active toast notifications with thread-safety."""

    def __init__(self, max_notifications: int = 5, default_duration: float = 4.0):
        self.max_notifications = max_notifications
        self.default_duration = default_duration
        self._items: List[NotificationItem] = []
        self._lock = threading.Lock()
        self._counter = 0

    def add(
        self,
        category: NotificationCategory,
        title: str,
        message: str,
        duration: Optional[float] = None,
    ) -> NotificationItem:
        """Pushes a new toast notification."""
        with self._lock:
            self._counter += 1
            item = NotificationItem(
                id=f"notif_{self._counter}_{int(time.time()*1000)}",
                title=title,
                message=message,
                category=category,
                timestamp=time.time(),
                duration_sec=duration if duration is not None else self.default_duration,
            )

            # Avoid duplicates of exact same message within 1 second
            for existing in self._items:
                if (
                    existing.title == title
                    and existing.message == message
                    and (time.time() - existing.timestamp) < 1.0
                ):
                    existing.timestamp = time.time()
                    return existing

            self._items.append(item)

            # Enforce max stack size by dropping oldest
            if len(self._items) > self.max_notifications:
                self._items = self._items[-self.max_notifications:]

            return item

    def success(self, title: str, message: str, duration: Optional[float] = None) -> NotificationItem:
        return self.add(NotificationCategory.SUCCESS, title, message, duration)

    def info(self, title: str, message: str, duration: Optional[float] = None) -> NotificationItem:
        return self.add(NotificationCategory.INFO, title, message, duration)

    def warning(self, title: str, message: str, duration: Optional[float] = None) -> NotificationItem:
        return self.add(NotificationCategory.WARNING, title, message, duration)

    def error(self, title: str, message: str, duration: Optional[float] = None) -> NotificationItem:
        return self.add(NotificationCategory.ERROR, title, message, duration)

    def get_active(self) -> List[NotificationItem]:
        """Returns non-expired, active notifications."""
        with self._lock:
            now = time.time()
            self._items = [item for item in self._items if not item.dismissed and (now - item.timestamp) < item.duration_sec]
            return list(self._items)

    def dismiss(self, notification_id: str):
        """Manually dismisses a notification by ID."""
        with self._lock:
            for item in self._items:
                if item.id == notification_id:
                    item.dismissed = True
                    break

    def clear(self):
        """Clears all notifications."""
        with self._lock:
            self._items.clear()
