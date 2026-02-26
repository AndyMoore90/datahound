"""SQLAlchemy-backed notification repository implementation."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterable
from uuid import UUID

from sqlalchemy import Select, select

from ..engine import SessionFactory, session_scope
from ..models import NotificationModel, NotificationRecord
from .notification_repo import NotificationFilter, NotificationRepository


class SQLNotificationRepository(NotificationRepository):
    """Persist notification delivery receipts in SQL."""

    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    @staticmethod
    def _model_to_record(model: NotificationModel) -> NotificationRecord:
        return NotificationRecord(
            id=model.id,
            task_id=model.task_id,
            channel=model.channel,
            target=model.target,
            message_type=model.message_type,
            payload_json=model.payload_json,
            sent_at=model.sent_at,
            status=model.status,
            provider_message_id=model.provider_message_id,
        )

    def _query(self, notification_filter: NotificationFilter | None = None) -> Select[tuple[NotificationModel]]:
        stmt = select(NotificationModel)
        if notification_filter:
            if notification_filter.task_id:
                stmt = stmt.where(NotificationModel.task_id == notification_filter.task_id)
            if notification_filter.status:
                stmt = stmt.where(NotificationModel.status == notification_filter.status)
        return stmt

    def record_notification(self, record: NotificationRecord) -> NotificationRecord:
        with session_scope(self._session_factory) as session:
            model = NotificationModel(
                task_id=record.task_id,
                channel=record.channel,
                target=record.target,
                message_type=record.message_type,
                payload_json=dict(record.payload_json) if record.payload_json else None,
                sent_at=record.sent_at or datetime.now(UTC),
                status=record.status,
                provider_message_id=record.provider_message_id,
            )
            session.add(model)
            session.flush()
            session.refresh(model)
            return self._model_to_record(model)

    def list_notifications(self, notification_filter: NotificationFilter | None = None) -> Iterable[NotificationRecord]:
        with session_scope(self._session_factory) as session:
            models = session.execute(self._query(notification_filter)).scalars().all()
            return [self._model_to_record(model) for model in models]

    def mark_status(self, notification_id: str, status: str) -> None:
        with session_scope(self._session_factory) as session:
            model = session.execute(select(NotificationModel).where(NotificationModel.id == UUID(notification_id))).scalar_one_or_none()
            if model is None:
                return
            model.status = status
            if status == "sent" and model.sent_at is None:
                model.sent_at = datetime.now(UTC)
            session.add(model)
