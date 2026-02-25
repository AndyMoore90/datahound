"""Concrete StorageDAL implementation wrapping repository dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from .dal import DALDependencies, StorageDAL
from .db.models import (
    DatasetVersionRecord,
    EventIndexRecord,
    NotificationRecord,
    PipelineRunRecord,
    ReviewGateRecord,
    SchedulerRunRecord,
    SchedulerTaskRecord,
)
from .db.repos.scheduler_repo import SchedulerTaskFilter
from .manifest import RunManifest


@dataclass(slots=True)
class SQLStorageDAL(StorageDAL):
    """Delegates StorageDAL calls to the configured repositories."""

    dependencies: DALDependencies

    # -- helpers --------------------------------------------------------------
    def _scheduler_repo(self):
        repo = self.dependencies.scheduler_repo
        if repo is None:
            raise RuntimeError("Scheduler repository not configured")
        return repo

    def _run_repo(self):
        repo = self.dependencies.run_repo
        if repo is None:
            raise RuntimeError("Run repository not configured")
        return repo

    def _event_repo(self):
        repo = self.dependencies.event_repo
        if repo is None:
            raise RuntimeError("Event repository not configured")
        return repo

    def _notification_repo(self):
        repo = self.dependencies.notification_repo
        if repo is None:
            raise RuntimeError("Notification repository not configured")
        return repo

    def _review_repo(self):
        repo = self.dependencies.review_repo
        if repo is None:
            raise RuntimeError("Review repository not configured")
        return repo

    # -- Scheduler ------------------------------------------------------------
    def create_task(self, task: SchedulerTaskRecord) -> SchedulerTaskRecord:
        return self._scheduler_repo().create_task(task)

    def update_task(self, task: SchedulerTaskRecord) -> SchedulerTaskRecord:
        return self._scheduler_repo().update_task(task)

    def delete_task(self, task_key: str) -> bool:
        return self._scheduler_repo().delete_task(task_key)

    def list_tasks(self, task_filter: SchedulerTaskFilter | None = None) -> Iterable[SchedulerTaskRecord]:
        return self._scheduler_repo().list_tasks(task_filter)

    def get_task(self, task_key: str) -> SchedulerTaskRecord | None:
        return self._scheduler_repo().get_task(task_key)

    def record_task_run(self, run: SchedulerRunRecord) -> SchedulerRunRecord:
        return self._scheduler_repo().record_run(run)

    def list_task_runs(self, *, task_key: str | None = None, limit: int = 100) -> Iterable[SchedulerRunRecord]:
        return self._scheduler_repo().list_runs(task_key=task_key, limit=limit)

    def purge_task_runs(self, *, older_than: datetime) -> int:
        return self._scheduler_repo().purge_runs(older_than)

    # -- Pipeline runs --------------------------------------------------------
    def start_pipeline_run(self, run: PipelineRunRecord) -> PipelineRunRecord:
        return self._run_repo().start_pipeline_run(run)

    def finish_pipeline_run(
        self,
        run_id: str,
        *,
        status: str,
        manifest: RunManifest | None,
        error: Mapping[str, object] | None = None,
    ) -> PipelineRunRecord:
        return self._run_repo().finish_pipeline_run(run_id, status=status, output_manifest=manifest, error=error)

    def register_dataset_version(self, dataset: DatasetVersionRecord) -> DatasetVersionRecord:
        return self._run_repo().register_dataset_version(dataset)

    # -- Events ---------------------------------------------------------------
    def upsert_events(
        self,
        events: Sequence[EventIndexRecord],
        *,
        run_id: str | None = None,
    ) -> int:
        return self._event_repo().upsert_events(events, run_id=run_id)

    def resolve_missing_events(
        self,
        known_event_ids: Iterable[str],
        *,
        event_type: str,
        company: str,
    ) -> Sequence[EventIndexRecord]:
        return self._event_repo().resolve_missing_events(known_event_ids, event_type=event_type, company=company)

    # -- Review + notifications ----------------------------------------------
    def upsert_review_gate(self, record: ReviewGateRecord) -> ReviewGateRecord:
        return self._review_repo().upsert_review_gate(record)

    def mark_review_ready(self, task_id: str, *, ready: bool) -> None:
        self._review_repo().mark_review_ready(task_id, ready=ready)

    def record_notification(self, record: NotificationRecord) -> NotificationRecord:
        return self._notification_repo().record_notification(record)
