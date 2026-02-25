"""High-level storage DAL interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

from .db.models import (
    DatasetVersionRecord,
    EventIndexRecord,
    NotificationRecord,
    PipelineRunRecord,
    ReviewGateRecord,
    SchedulerRunRecord,
    SchedulerTaskRecord,
)
from .db.repos.event_repo import EventRepository
from .db.repos.notification_repo import NotificationRepository
from .db.repos.review_repo import ReviewRepository
from .db.repos.run_repo import RunRepository
from .db.repos.scheduler_repo import SchedulerRepository, SchedulerTaskFilter
from .manifest import RunManifest


@dataclass(slots=True)
class DALDependencies:
    """Container for the repositories that power a ``StorageDAL`` implementation."""

    scheduler_repo: SchedulerRepository
    run_repo: RunRepository
    event_repo: EventRepository
    notification_repo: NotificationRepository
    review_repo: ReviewRepository


class StorageDAL(Protocol):
    """Facade coordinating scheduler, run, event, and review persistence."""

    dependencies: DALDependencies

    # Scheduler -----------------------------------------------------------------
    def create_task(self, task: SchedulerTaskRecord) -> SchedulerTaskRecord:
        ...

    def update_task(self, task: SchedulerTaskRecord) -> SchedulerTaskRecord:
        ...

    def list_tasks(self, task_filter: SchedulerTaskFilter | None = None) -> Iterable[SchedulerTaskRecord]:
        ...

    def record_task_run(self, run: SchedulerRunRecord) -> SchedulerRunRecord:
        ...

    # Pipeline runs --------------------------------------------------------------
    def start_pipeline_run(self, run: PipelineRunRecord) -> PipelineRunRecord:
        ...

    def finish_pipeline_run(
        self,
        run_id: str,
        *,
        status: str,
        manifest: RunManifest | None,
        error: Mapping[str, object] | None = None,
    ) -> PipelineRunRecord:
        ...

    def register_dataset_version(self, dataset: DatasetVersionRecord) -> DatasetVersionRecord:
        ...

    # Events --------------------------------------------------------------------
    def upsert_events(
        self,
        events: Sequence[EventIndexRecord],
        *,
        run_id: str | None = None,
    ) -> int:
        ...

    def resolve_missing_events(
        self,
        known_event_ids: Iterable[str],
        *,
        event_type: str,
        company: str,
    ) -> Sequence[EventIndexRecord]:
        ...

    # Review + notifications ----------------------------------------------------
    def upsert_review_gate(self, record: ReviewGateRecord) -> ReviewGateRecord:
        ...

    def mark_review_ready(self, task_id: str, *, ready: bool) -> None:
        ...

    def record_notification(self, record: NotificationRecord) -> NotificationRecord:
        ...
