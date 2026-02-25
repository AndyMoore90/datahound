"""Tests for the scheduler persistence migration."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from unittest.mock import patch
from uuid import uuid4

from datahound.scheduler.persistence import SchedulerPersistence
from datahound.scheduler.tasks import (
    ScheduleType,
    ScheduledTask,
    TaskConfiguration,
    TaskStatus,
    TaskType,
)
from datahound.storage.dal import DALDependencies, StorageDAL
from datahound.storage.db.models import (
    DatasetVersionRecord,
    EventIndexRecord,
    NotificationRecord,
    PipelineRunRecord,
    ReviewGateRecord,
    SchedulerRunRecord,
    SchedulerTaskRecord,
)
from datahound.storage.db.repos.scheduler_repo import SchedulerTaskFilter
from datahound.storage.manifest import RunManifest


class FakeStorageDAL(StorageDAL):
    """In-memory DAL stub for scheduler-specific tests."""

    def __init__(self):
        self.dependencies = DALDependencies()
        self._tasks: dict[str, SchedulerTaskRecord] = {}
        self._runs: list[SchedulerRunRecord] = []

    # -- Scheduler -----------------------------------------------------
    def create_task(self, task: SchedulerTaskRecord) -> SchedulerTaskRecord:
        stored = replace(task, id=uuid4(), created_at=task.created_at or datetime.now(timezone.utc))
        self._tasks[stored.task_key] = stored
        return stored

    def update_task(self, task: SchedulerTaskRecord) -> SchedulerTaskRecord:
        current = self._tasks.get(task.task_key)
        stored = replace(task, id=current.id if current else uuid4(), updated_at=datetime.now(timezone.utc))
        self._tasks[stored.task_key] = stored
        return stored

    def delete_task(self, task_key: str) -> bool:
        return self._tasks.pop(task_key, None) is not None

    def list_tasks(self, task_filter: SchedulerTaskFilter | None = None) -> Iterable[SchedulerTaskRecord]:
        records = list(self._tasks.values())
        if task_filter:
            if task_filter.company:
                records = [rec for rec in records if rec.company == task_filter.company]
            if task_filter.status:
                records = [rec for rec in records if rec.status == task_filter.status]
        return records

    def get_task(self, task_key: str) -> SchedulerTaskRecord | None:
        return self._tasks.get(task_key)

    def record_task_run(self, run: SchedulerRunRecord) -> SchedulerRunRecord:
        stored = replace(run, id=uuid4())
        self._runs.insert(0, stored)
        return stored

    def list_task_runs(self, *, task_key: str | None = None, limit: int = 100) -> Iterable[SchedulerRunRecord]:
        runs = []
        for record in self._runs:
            metadata = record.metadata_json or {}
            if task_key is None or metadata.get("task_key") == task_key:
                runs.append(record)
        return runs[:limit]

    def purge_task_runs(self, *, older_than: datetime) -> int:
        kept: list[SchedulerRunRecord] = []
        removed = 0
        for run in self._runs:
            if run.started_at < older_than:
                removed += 1
            else:
                kept.append(run)
        self._runs = kept
        return removed

    # -- Remaining protocol hooks (not exercised) ----------------------
    def start_pipeline_run(self, run: PipelineRunRecord) -> PipelineRunRecord:
        raise NotImplementedError

    def finish_pipeline_run(
        self,
        run_id: str,
        *,
        status: str,
        manifest: RunManifest | None,
        error: Mapping[str, object] | None = None,
    ) -> PipelineRunRecord:
        raise NotImplementedError

    def register_dataset_version(self, dataset: DatasetVersionRecord) -> DatasetVersionRecord:
        raise NotImplementedError

    def upsert_events(
        self,
        events: Sequence[EventIndexRecord],
        *,
        run_id: str | None = None,
    ) -> int:
        raise NotImplementedError

    def resolve_missing_events(
        self,
        known_event_ids: Iterable[str],
        *,
        event_type: str,
        company: str,
    ) -> Sequence[EventIndexRecord]:
        raise NotImplementedError

    def upsert_review_gate(self, record: ReviewGateRecord) -> ReviewGateRecord:
        raise NotImplementedError

    def mark_review_ready(self, task_id: str, *, ready: bool) -> None:
        raise NotImplementedError

    def record_notification(self, record: NotificationRecord) -> NotificationRecord:
        raise NotImplementedError


def make_task(task_id: str = "task-1", company: str = "ACME") -> ScheduledTask:
    config = TaskConfiguration(task_type=TaskType.DOWNLOAD, company=company)
    return ScheduledTask(
        task_id=task_id,
        name=f"Test {task_id}",
        description="demo",
        task_config=config,
        schedule_type=ScheduleType.INTERVAL,
        status=TaskStatus.ACTIVE,
        interval_minutes=15,
    )


class SchedulerPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.base_path = Path(self.tmpdir.name)

    def test_file_backend_round_trip(self):
        persistence = SchedulerPersistence(self.base_path)
        task = make_task("file-task")
        self.assertTrue(persistence.add_task(task))
        loaded = persistence.get_task(task.task_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.task_id, task.task_id)
        with open(persistence.tasks_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["task_id"], task.task_id)

    def test_db_backend_records_runs_and_backup(self):
        fake_dal = FakeStorageDAL()
        with patch("datahound.scheduler.persistence.get_storage_dal_from_env", return_value=fake_dal):
            persistence = SchedulerPersistence(self.base_path)
        task = make_task("db-task")
        self.assertTrue(persistence.add_task(task))
        self.assertIn(task.task_id, fake_dal._tasks)
        persistence.log_task_execution(task.task_id, True, "ok", 2.5, task.name, task.task_config.task_type.value)
        history = persistence.get_task_history(task_id=task.task_id)
        self.assertTrue(history)
        self.assertEqual(history[0]["task_id"], task.task_id)
        with open(persistence.tasks_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data[0]["task_id"], task.task_id)

    def test_bootstrap_copies_existing_file_tasks_into_db(self):
        # Seed a file-based snapshot
        with patch("datahound.scheduler.persistence.get_storage_dal_from_env", return_value=None):
            seed_persistence = SchedulerPersistence(self.base_path)
            task = make_task("seed-task")
            seed_persistence.add_task(task)
        fake_dal = FakeStorageDAL()
        with patch("datahound.scheduler.persistence.get_storage_dal_from_env", return_value=fake_dal):
            SchedulerPersistence(self.base_path)
        self.assertIn("seed-task", fake_dal._tasks)


if __name__ == "__main__":
    unittest.main()
