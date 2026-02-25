"""Scheduler repository interface."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Protocol

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

    def delete_task(self, task_key: str) -> bool:
        ...

    def list_tasks(self, task_filter: SchedulerTaskFilter | None = None) -> Iterable[SchedulerTaskRecord]:
        ...

    def record_run(self, run: SchedulerRunRecord) -> SchedulerRunRecord:
        ...

    def list_runs(self, *, task_key: str | None = None, limit: int = 100) -> Iterable[SchedulerRunRecord]:
        ...

    def purge_runs(self, cutoff: datetime) -> int:
        ...

    def get_task(self, task_key: str) -> SchedulerTaskRecord | None:
        ...
