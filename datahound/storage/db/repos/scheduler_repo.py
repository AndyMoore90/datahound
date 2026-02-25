"""Scheduler repository interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol
from uuid import UUID

from ..models import SchedulerRunRecord, SchedulerTaskRecord


@dataclass(slots=True)
class SchedulerTaskFilter:
    """Filtering options for listing tasks."""

    company: str | None = None
    status: str | None = "active"


class SchedulerRepository(Protocol):
    """Persistence boundary for scheduler tasks + runs."""

    def create_task(self, task: SchedulerTaskRecord) -> SchedulerTaskRecord:
        ...

    def update_task(self, task: SchedulerTaskRecord) -> SchedulerTaskRecord:
        ...

    def list_tasks(self, task_filter: SchedulerTaskFilter | None = None) -> Iterable[SchedulerTaskRecord]:
        ...

    def record_run(self, run: SchedulerRunRecord) -> SchedulerRunRecord:
        ...

    def get_task(self, task_id: UUID) -> SchedulerTaskRecord | None:
        ...
