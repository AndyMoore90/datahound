"""Review gate repository interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from ..models import ReviewGateRecord


@dataclass(slots=True)
class ReviewGateFilter:
    task_id: str | None = None
    ready: bool | None = None


class ReviewRepository(Protocol):
    """Stores review gate state per task."""

    def upsert_review_gate(self, record: ReviewGateRecord) -> ReviewGateRecord:
        ...

    def mark_ready(self, task_id: str, *, ready: bool) -> None:
        ...

    def list_review_gates(self, review_filter: ReviewGateFilter | None = None) -> Iterable[ReviewGateRecord]:
        ...
