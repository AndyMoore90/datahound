"""Persistence layer for scheduled tasks with Postgres + JSON fallback."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

# fcntl is Unix-only, handle Windows separately
try:  # pragma: no cover - optional import guard
    import fcntl

    HAS_FCNTL = True
except ImportError:  # pragma: no cover - fallback on Windows
    HAS_FCNTL = False

from central_logging.config import scheduler_dir
from datahound.storage.bootstrap import get_storage_dal_from_env
from datahound.storage.dal import StorageDAL
from datahound.storage.db.models import SchedulerRunRecord, SchedulerTaskRecord
from datahound.storage.db.repos.scheduler_repo import SchedulerTaskFilter

from .tasks import ScheduleType, ScheduledTask, TaskStatus, TaskType


class SchedulerPersistence:
    """Handles persistence of scheduled tasks with DB + file compatibility."""

    def __init__(self, data_dir: Path):
        """Initialize persistence with data directory."""

        self.data_dir = Path(data_dir)
        self.scheduler_dir = self.data_dir / "scheduler"
        self.tasks_file = self.scheduler_dir / "scheduled_tasks.json"
        self.history_file = scheduler_dir() / "task_history.jsonl"
        self.lock_file = self.scheduler_dir / ".scheduler.lock"
        self.scheduler_dir.mkdir(parents=True, exist_ok=True)

        self._storage_dal: StorageDAL | None = None
        self._storage_failed = False
        self._tz_label = os.getenv("DATAHOUND_SCHEDULER_TZ", "UTC")

        self._storage_dal = get_storage_dal_from_env()
        if self._storage_dal:
            self._bootstrap_storage()

    # ------------------------------------------------------------------
    # File locking helpers
    @contextmanager
    def _file_lock(self, timeout: int = 10):
        """Context manager for file locking."""

        lock_fd = None
        try:
            if not HAS_FCNTL or os.name == "nt":
                import time

                start_time = time.time()
                while self.lock_file.exists():
                    if time.time() - start_time > timeout:
                        raise TimeoutError("Failed to acquire scheduler lock")
                    time.sleep(0.1)
                self.lock_file.touch()
                yield
            else:  # pragma: no cover - Unix-only branch
                lock_fd = open(self.lock_file, "w")
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield
        finally:
            if not HAS_FCNTL or os.name == "nt":
                if self.lock_file.exists():
                    try:
                        self.lock_file.unlink()
                    except OSError:
                        pass
            elif lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()

    # ------------------------------------------------------------------
    # Public API - tasks
    def load_tasks(self) -> List[ScheduledTask]:
        """Load all scheduled tasks from storage."""

        if self._use_storage():
            return self._list_tasks_from_db(update_backup=True)
        return self._load_tasks_from_file()

    def save_tasks(self, tasks: List[ScheduledTask]) -> bool:
        """Persist a full task snapshot."""

        if self._use_storage():
            try:
                existing = {record.task_key for record in self._storage_dal.list_tasks()}
                seen: set[str] = set()
                for task in tasks:
                    record = self._task_to_record(task)
                    if task.task_id in existing:
                        self._storage_dal.update_task(record)
                    else:
                        self._storage_dal.create_task(record)
                    seen.add(task.task_id)
                for missing in existing - seen:
                    self._storage_dal.delete_task(missing)
                self._save_tasks_to_file(tasks)
                return True
            except Exception as exc:
                self._disable_storage(f"Failed to save tasks via DAL: {exc}")
        return self._save_tasks_to_file(tasks)

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a specific task by ID."""

        if self._use_storage():
            try:
                record = self._storage_dal.get_task(task_id)
                return self._record_to_task(record) if record else None
            except Exception as exc:
                self._disable_storage(f"Failed to read task via DAL: {exc}")
        tasks = self._load_tasks_from_file()
        for task in tasks:
            if task.task_id == task_id:
                return task
        return None

    def add_task(self, task: ScheduledTask) -> bool:
        """Add a new scheduled task."""

        if self._use_storage():
            try:
                self._storage_dal.create_task(self._task_to_record(task))
                self._refresh_file_backup()
                return True
            except Exception as exc:
                self._disable_storage(f"Failed to add task via DAL: {exc}")

        tasks = self._load_tasks_from_file()
        if any(existing.task_id == task.task_id for existing in tasks):
            return False
        tasks.append(task)
        return self._save_tasks_to_file(tasks)

    def update_task(self, task: ScheduledTask) -> bool:
        """Update an existing scheduled task."""

        if self._use_storage():
            try:
                task.updated_at = datetime.now()
                self._storage_dal.update_task(self._task_to_record(task))
                self._refresh_file_backup()
                return True
            except Exception as exc:
                self._disable_storage(f"Failed to update task via DAL: {exc}")

        tasks = self._load_tasks_from_file()
        for i, existing in enumerate(tasks):
            if existing.task_id == task.task_id:
                task.updated_at = datetime.now()
                tasks[i] = task
                return self._save_tasks_to_file(tasks)
        return False

    def delete_task(self, task_id: str) -> bool:
        """Delete a scheduled task."""

        if self._use_storage():
            try:
                deleted = self._storage_dal.delete_task(task_id)
                self._refresh_file_backup()
                return deleted
            except Exception as exc:
                self._disable_storage(f"Failed to delete task via DAL: {exc}")

        tasks = self._load_tasks_from_file()
        new_tasks = [task for task in tasks if task.task_id != task_id]
        if len(new_tasks) == len(tasks):
            return False
        return self._save_tasks_to_file(new_tasks)

    def get_active_tasks(self) -> List[ScheduledTask]:
        """Get all active scheduled tasks."""

        if self._use_storage():
            try:
                records = self._storage_dal.list_tasks(
                    SchedulerTaskFilter(status=TaskStatus.ACTIVE.value)
                )
                return [self._record_to_task(record) for record in records]
            except Exception as exc:
                self._disable_storage(f"Failed to list tasks via DAL: {exc}")
        return [task for task in self._load_tasks_from_file() if task.status == TaskStatus.ACTIVE]

    def get_tasks_by_type(self, task_type: TaskType) -> List[ScheduledTask]:
        """Get all tasks of a specific type."""

        return [task for task in self.load_tasks() if task.task_config.task_type == task_type]

    def get_tasks_by_company(self, company: str) -> List[ScheduledTask]:
        """Get all tasks for a specific company."""

        if self._use_storage():
            try:
                records = self._storage_dal.list_tasks(SchedulerTaskFilter(company=company))
                return [self._record_to_task(record) for record in records]
            except Exception as exc:
                self._disable_storage(f"Failed to filter tasks via DAL: {exc}")
        return [task for task in self._load_tasks_from_file() if task.task_config.company == company]

    # ------------------------------------------------------------------
    # Task execution history
    def log_task_execution(
        self,
        task_id: str,
        success: bool,
        message: str = "",
        duration_seconds: float = 0,
        task_name: str = "",
        task_type: str = "",
    ) -> None:
        """Record a task execution in DB and file history."""

        timestamp = datetime.now(timezone.utc)
        history_entry = {
            "timestamp": timestamp.isoformat(),
            "task_id": task_id,
            "task_name": task_name,
            "task_type": task_type,
            "success": success,
            "message": message,
            "duration_seconds": duration_seconds,
        }

        if self._use_storage():
            try:
                record = self._storage_dal.get_task(task_id)
                if record and record.id:
                    started_at = timestamp - timedelta(seconds=max(duration_seconds, 0))
                    run_record = SchedulerRunRecord(
                        id=None,
                        task_id=record.id,
                        run_id=uuid4().hex,
                        started_at=started_at,
                        finished_at=timestamp,
                        success=success,
                        message=message,
                        duration_ms=int(duration_seconds * 1000),
                        metadata_json={
                            "task_key": task_id,
                            "task_name": task_name,
                            "task_type": task_type,
                        },
                    )
                    self._storage_dal.record_task_run(run_record)
            except Exception as exc:
                self._disable_storage(f"Failed to log task run via DAL: {exc}")
        self._append_history_file(history_entry)

    def get_task_history(self, task_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get task execution history."""

        if self._use_storage():
            try:
                runs = self._storage_dal.list_task_runs(task_key=task_id, limit=limit)
                return [self._run_record_to_history(run) for run in runs]
            except Exception as exc:
                self._disable_storage(f"Failed to read task history via DAL: {exc}")
        return self._read_history_file(task_id=task_id, limit=limit)

    def clear_old_history(self, days_to_keep: int = 30):
        """Clear old task history entries."""

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        if self._use_storage():
            try:
                self._storage_dal.purge_task_runs(older_than=cutoff)
            except Exception as exc:
                self._disable_storage(f"Failed to purge task history via DAL: {exc}")
        self._truncate_history_file(cutoff)

    # ------------------------------------------------------------------
    # Internal helpers - DB bridging
    def _use_storage(self) -> bool:
        return self._storage_dal is not None and not self._storage_failed

    def _disable_storage(self, message: str):
        print(message)
        self._storage_failed = True
        self._storage_dal = None

    def _bootstrap_storage(self):
        try:
            records = list(self._storage_dal.list_tasks())  # type: ignore[union-attr]
            if not records:
                file_tasks = self._load_tasks_from_file()
                for task in file_tasks:
                    self._storage_dal.create_task(self._task_to_record(task))  # type: ignore[union-attr]
                records = list(self._storage_dal.list_tasks())  # type: ignore[union-attr]
            tasks = [self._record_to_task(record) for record in records]
            self._save_tasks_to_file(tasks)
        except Exception as exc:
            self._disable_storage(f"Failed to bootstrap scheduler storage: {exc}")

    def _list_tasks_from_db(
        self,
        task_filter: SchedulerTaskFilter | None = None,
        *,
        update_backup: bool = False,
    ) -> List[ScheduledTask]:
        records = list(self._storage_dal.list_tasks(task_filter))  # type: ignore[union-attr]
        tasks = [self._record_to_task(record) for record in records]
        if update_backup and task_filter is None:
            self._save_tasks_to_file(tasks)
        return tasks

    def _refresh_file_backup(self):
        if self._use_storage():
            self._list_tasks_from_db(update_backup=True)

    def _task_to_record(self, task: ScheduledTask) -> SchedulerTaskRecord:
        payload = task.to_dict()
        payload.setdefault("task_id", task.task_id)
        aware_created = self._ensure_aware(task.created_at)
        aware_updated = self._ensure_aware(task.updated_at)
        return SchedulerTaskRecord(
            id=None,
            task_key=task.task_id,
            task_type=task.task_config.task_type.value,
            company=task.task_config.company,
            config_json=payload,
            schedule_type=task.schedule_type.value,
            schedule_expr=self._encode_schedule_expr(task),
            timezone=self._tz_label,
            status=task.status.value,
            created_at=aware_created,
            updated_at=aware_updated,
        )

    def _record_to_task(self, record: SchedulerTaskRecord) -> ScheduledTask:
        data: Dict[str, Any] = dict(record.config_json or {})
        data.setdefault("task_id", record.task_key)
        if "task_config" not in data:
            data["task_config"] = {
                "task_type": record.task_type,
                "company": record.company,
            }
        return ScheduledTask.from_dict(data)

    @staticmethod
    def _ensure_aware(value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _encode_schedule_expr(task: ScheduledTask) -> str:
        if task.schedule_type == ScheduleType.INTERVAL:
            return f"interval:{task.interval_minutes or 0}"
        if task.schedule_type == ScheduleType.DAILY_TIME:
            return f"daily:{task.daily_time.isoformat() if task.daily_time else '00:00'}"
        if task.schedule_type == ScheduleType.WEEKLY_SCHEDULE and task.weekly_schedule:
            days = ",".join(day.value[:3] for day in task.weekly_schedule.days)
            interval = task.weekly_schedule.interval_minutes or 60
            return f"weekly:{days}@{interval}"
        return task.schedule_type.value

    def _run_record_to_history(self, run: SchedulerRunRecord) -> Dict[str, Any]:
        metadata = dict(run.metadata_json or {})
        timestamp = (run.finished_at or run.started_at).isoformat()
        return {
            "timestamp": timestamp,
            "task_id": metadata.get("task_key"),
            "task_name": metadata.get("task_name"),
            "task_type": metadata.get("task_type"),
            "success": run.success,
            "message": run.message,
            "duration_seconds": (run.duration_ms or 0) / 1000.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers - file persistence
    def _load_tasks_from_file(self) -> List[ScheduledTask]:
        if not self.tasks_file.exists():
            return []
        try:
            with self._file_lock():
                with open(self.tasks_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    return [ScheduledTask.from_dict(task_data) for task_data in data]
        except Exception as exc:
            print(f"Error loading scheduled tasks: {exc}")
            return []

    def _save_tasks_to_file(self, tasks: Iterable[ScheduledTask]) -> bool:
        try:
            with self._file_lock():
                data = [task.to_dict() for task in tasks]
                with open(self.tasks_file, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2)
            return True
        except Exception as exc:
            print(f"Error saving scheduled tasks: {exc}")
            return False

    def _append_history_file(self, entry: Dict[str, Any]):
        try:
            with self._file_lock():
                with open(self.history_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
        except Exception as exc:
            print(f"Error logging task execution: {exc}")

    def _read_history_file(self, task_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            return []
        history: List[Dict[str, Any]] = []
        try:
            with self._file_lock():
                with open(self.history_file, "r", encoding="utf-8") as handle:
                    lines = handle.readlines()
            for line in lines[-limit:]:
                try:
                    entry = json.loads(line.strip())
                    if task_id is None or entry.get("task_id") == task_id:
                        history.append(entry)
                except json.JSONDecodeError:
                    continue
        except Exception as exc:
            print(f"Error reading task history: {exc}")
        return history

    def _truncate_history_file(self, cutoff: datetime):
        if not self.history_file.exists():
            return
        try:
            kept_lines: List[str] = []
            with self._file_lock():
                with open(self.history_file, "r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            entry = json.loads(line.strip())
                            timestamp = datetime.fromisoformat(entry["timestamp"])
                            if timestamp.replace(tzinfo=timestamp.tzinfo or timezone.utc) >= cutoff:
                                kept_lines.append(line)
                        except Exception:
                            continue
                with open(self.history_file, "w", encoding="utf-8") as handle:
                    handle.writelines(kept_lines)
        except Exception as exc:
            print(f"Error clearing old history: {exc}")
