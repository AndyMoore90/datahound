"""SQLAlchemy-backed scheduler repository implementation."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable
from uuid import uuid4

from sqlalchemy import Select, delete, desc, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session

from ..engine import SessionFactory, session_scope
from ..models import SchedulerRunModel, SchedulerRunRecord, SchedulerTaskModel, SchedulerTaskRecord
from .scheduler_repo import SchedulerRepository, SchedulerTaskFilter


class SQLSchedulerRepository(SchedulerRepository):
    """Persist scheduler tasks + runs using SQLAlchemy sessions."""

    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    # -- Task helpers ---------------------------------------------------------
    def _task_model_query(self, *, task_key: str | None = None) -> Select[tuple[SchedulerTaskModel]]:
        stmt = select(SchedulerTaskModel)
        if task_key:
            stmt = stmt.where(SchedulerTaskModel.task_key == task_key)
        return stmt

    @staticmethod
    def _model_to_task_record(model: SchedulerTaskModel) -> SchedulerTaskRecord:
        return SchedulerTaskRecord(
            id=model.id,
            task_key=model.task_key,
            task_type=model.task_type,
            company=model.company,
            config_json=model.config_json,
            schedule_type=model.schedule_type,
            schedule_expr=model.schedule_expr,
            timezone=model.timezone,
            status=model.status,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _get_task_model(self, session: Session, task_key: str) -> SchedulerTaskModel:
        stmt = self._task_model_query(task_key=task_key)
        result = session.execute(stmt).scalar_one_or_none()
        if result is None:
            raise NoResultFound(f"Scheduler task {task_key} not found")
        return result

    def create_task(self, task: SchedulerTaskRecord) -> SchedulerTaskRecord:
        with session_scope(self._session_factory) as session:
            model = SchedulerTaskModel(
                task_key=task.task_key,
                task_type=task.task_type,
                company=task.company,
                config_json=task.config_json,
                schedule_type=task.schedule_type,
                schedule_expr=task.schedule_expr,
                timezone=task.timezone,
                status=task.status,
                created_at=task.created_at or datetime.utcnow(),
                updated_at=task.updated_at or datetime.utcnow(),
            )
            session.add(model)
            session.flush()
            session.refresh(model)
            return self._model_to_task_record(model)

    def update_task(self, task: SchedulerTaskRecord) -> SchedulerTaskRecord:
        with session_scope(self._session_factory) as session:
            model = self._get_task_model(session, task.task_key)
            model.task_type = task.task_type
            model.company = task.company
            model.config_json = task.config_json
            model.schedule_type = task.schedule_type
            model.schedule_expr = task.schedule_expr
            model.timezone = task.timezone
            model.status = task.status
            model.updated_at = task.updated_at or datetime.utcnow()
            session.add(model)
            session.flush()
            session.refresh(model)
            return self._model_to_task_record(model)

    def delete_task(self, task_key: str) -> bool:
        with session_scope(self._session_factory) as session:
            stmt = delete(SchedulerTaskModel).where(SchedulerTaskModel.task_key == task_key)
            result = session.execute(stmt)
            deleted = result.rowcount or 0
            return deleted > 0

    def list_tasks(self, task_filter: SchedulerTaskFilter | None = None) -> Iterable[SchedulerTaskRecord]:
        with session_scope(self._session_factory) as session:
            stmt = self._task_model_query()
            if task_filter:
                if task_filter.company:
                    stmt = stmt.where(SchedulerTaskModel.company == task_filter.company)
                if task_filter.status:
                    stmt = stmt.where(SchedulerTaskModel.status == task_filter.status)
            models = session.execute(stmt).scalars().all()
            return [self._model_to_task_record(model) for model in models]

    def get_task(self, task_key: str) -> SchedulerTaskRecord | None:
        with session_scope(self._session_factory) as session:
            stmt = self._task_model_query(task_key=task_key)
            model = session.execute(stmt).scalar_one_or_none()
            return self._model_to_task_record(model) if model else None

    # -- Run helpers ----------------------------------------------------------
    @staticmethod
    def _model_to_run_record(model: SchedulerRunModel) -> SchedulerRunRecord:
        return SchedulerRunRecord(
            id=model.id,
            task_id=model.task_id,
            run_id=model.run_id,
            started_at=model.started_at,
            finished_at=model.finished_at,
            success=model.success,
            message=model.message,
            duration_ms=model.duration_ms,
            metadata_json=model.metadata_json,
        )

    def record_run(self, run: SchedulerRunRecord) -> SchedulerRunRecord:
        with session_scope(self._session_factory) as session:
            model = SchedulerRunModel(
                task_id=run.task_id,
                run_id=run.run_id or uuid4().hex,
                started_at=run.started_at,
                finished_at=run.finished_at,
                success=run.success,
                message=run.message,
                duration_ms=run.duration_ms,
                metadata_json=run.metadata_json,
            )
            session.add(model)
            session.flush()
            session.refresh(model)
            return self._model_to_run_record(model)

    def list_runs(self, *, task_key: str | None = None, limit: int = 100) -> Iterable[SchedulerRunRecord]:
        with session_scope(self._session_factory) as session:
            stmt = select(SchedulerRunModel).order_by(desc(SchedulerRunModel.started_at)).limit(limit)
            if task_key:
                task_id = (
                    session.execute(
                        select(SchedulerTaskModel.id).where(SchedulerTaskModel.task_key == task_key)
                    ).scalar_one_or_none()
                )
                if task_id is None:
                    return []
                stmt = stmt.where(SchedulerRunModel.task_id == task_id)
            models = session.execute(stmt).scalars().all()
            return [self._model_to_run_record(model) for model in models]

    def purge_runs(self, cutoff: datetime) -> int:
        with session_scope(self._session_factory) as session:
            stmt = delete(SchedulerRunModel).where(SchedulerRunModel.started_at < cutoff)
            result = session.execute(stmt)
            return result.rowcount or 0
