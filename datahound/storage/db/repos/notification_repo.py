"""Notification repository interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from ..models import NotificationRecord


@dataclass(slots=True)
class NotificationFilter:
    task_id: str | None = None
    status: str | None = None


class NotificationRepository(Protocol):
    """Persists delivery receipts for outbound notifications."""

    def record_notification(self, record: NotificationRecord) -> NotificationRecord:
        ...

    def list_notifications(self, notification_filter: NotificationFilter | None = None) -> Iterable[NotificationRecord]:
        ...

    def mark_status(self, notification_id: str, status: str) -> None:
        ...
