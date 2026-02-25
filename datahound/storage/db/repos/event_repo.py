"""Event index repository interface."""
from __future__ import annotations

from typing import Iterable, Protocol, Sequence

from ..models import EventIndexRecord


class EventRepository(Protocol):
    """Manages the canonical events index stored in Postgres."""

    def upsert_events(
        self,
        events: Sequence[EventIndexRecord],
        *,
        run_id: str | None = None,
    ) -> int:
        """Returns number of records affected."""

    def resolve_missing_events(
        self,
        known_event_ids: Iterable[str],
        *,
        event_type: str,
        company: str,
    ) -> Sequence[EventIndexRecord]:
        ...
