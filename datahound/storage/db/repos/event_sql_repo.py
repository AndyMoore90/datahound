"""SQLAlchemy-backed events index repository implementation."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterable, Sequence

from sqlalchemy import Select, and_, select

from ..engine import SessionFactory, session_scope
from ..models import EventIndexModel, EventIndexRecord
from .event_repo import EventRepository


class SQLEventRepository(EventRepository):
    """Persist canonical events index records in SQL."""

    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    @staticmethod
    def _model_to_record(model: EventIndexModel) -> EventIndexRecord:
        return EventIndexRecord(
            id=model.id,
            event_id=model.event_id,
            company=model.company,
            event_type=model.event_type,
            entity_type=model.entity_type,
            entity_id=model.entity_id,
            severity=model.severity,
            status=model.status,
            first_seen_at=model.first_seen_at,
            last_seen_at=model.last_seen_at,
            source_file=model.source_file,
            details_json=model.details_json,
        )

    def _query(self) -> Select[tuple[EventIndexModel]]:
        return select(EventIndexModel)

    def upsert_events(
        self,
        events: Sequence[EventIndexRecord],
        *,
        run_id: str | None = None,
    ) -> int:
        if not events:
            return 0

        now = datetime.now(UTC)
        event_ids = [event.event_id for event in events]
        with session_scope(self._session_factory) as session:
            existing = {
                model.event_id: model
                for model in session.execute(self._query().where(EventIndexModel.event_id.in_(event_ids))).scalars().all()
            }

            affected = 0
            for event in events:
                model = existing.get(event.event_id)
                if model is None:
                    model = EventIndexModel(
                        event_id=event.event_id,
                        company=event.company,
                        event_type=event.event_type,
                        entity_type=event.entity_type,
                        entity_id=event.entity_id,
                        severity=event.severity,
                        status=event.status,
                        first_seen_at=event.first_seen_at,
                        last_seen_at=event.last_seen_at,
                        source_file=event.source_file,
                        details_json=dict(event.details_json) if event.details_json else None,
                    )
                    session.add(model)
                    affected += 1
                    continue

                model.company = event.company
                model.event_type = event.event_type
                model.entity_type = event.entity_type
                model.entity_id = event.entity_id
                model.severity = event.severity
                model.status = event.status
                model.last_seen_at = event.last_seen_at or now
                if not model.first_seen_at:
                    model.first_seen_at = event.first_seen_at or now
                model.source_file = event.source_file
                model.details_json = dict(event.details_json) if event.details_json else model.details_json
                session.add(model)
                affected += 1

            return affected

    def resolve_missing_events(
        self,
        known_event_ids: Iterable[str],
        *,
        event_type: str,
        company: str,
    ) -> Sequence[EventIndexRecord]:
        known = set(known_event_ids)
        with session_scope(self._session_factory) as session:
            stmt = self._query().where(
                and_(
                    EventIndexModel.company == company,
                    EventIndexModel.event_type == event_type,
                    EventIndexModel.status == "active",
                )
            )
            models = session.execute(stmt).scalars().all()
            updated: list[EventIndexRecord] = []
            for model in models:
                if model.event_id in known:
                    continue
                model.status = "resolved"
                model.last_seen_at = datetime.now(UTC)
                session.add(model)
                updated.append(self._model_to_record(model))
            return updated
