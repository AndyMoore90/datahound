"""SQLAlchemy-backed review gate repository implementation."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterable

from sqlalchemy import Select, select

from ..engine import SessionFactory, session_scope
from ..models import ReviewGateModel, ReviewGateRecord
from .review_repo import ReviewGateFilter, ReviewRepository


class SQLReviewRepository(ReviewRepository):
    """Persist review gate state in SQL."""

    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    @staticmethod
    def _model_to_record(model: ReviewGateModel) -> ReviewGateRecord:
        return ReviewGateRecord(
            id=model.id,
            task_id=model.task_id,
            pr_number=model.pr_number,
            repo=model.repo,
            branch=model.branch,
            ci_passed=model.ci_passed,
            codex_passed=model.codex_passed,
            claude_passed=model.claude_passed,
            gemini_passed=model.gemini_passed,
            branch_up_to_date=model.branch_up_to_date,
            screenshots_included=model.screenshots_included,
            mode=model.mode,
            ready=model.ready,
            updated_at=model.updated_at,
        )

    def _query(self, review_filter: ReviewGateFilter | None = None) -> Select[tuple[ReviewGateModel]]:
        stmt = select(ReviewGateModel)
        if review_filter:
            if review_filter.task_id:
                stmt = stmt.where(ReviewGateModel.task_id == review_filter.task_id)
            if review_filter.ready is not None:
                stmt = stmt.where(ReviewGateModel.ready == review_filter.ready)
        return stmt

    def upsert_review_gate(self, record: ReviewGateRecord) -> ReviewGateRecord:
        with session_scope(self._session_factory) as session:
            model = session.execute(
                select(ReviewGateModel).where(ReviewGateModel.task_id == record.task_id)
            ).scalar_one_or_none()
            if model is None:
                model = ReviewGateModel(
                    task_id=record.task_id,
                    pr_number=record.pr_number,
                    repo=record.repo,
                    branch=record.branch,
                    ci_passed=record.ci_passed,
                    codex_passed=record.codex_passed,
                    claude_passed=record.claude_passed,
                    gemini_passed=record.gemini_passed,
                    branch_up_to_date=record.branch_up_to_date,
                    screenshots_included=record.screenshots_included,
                    mode=record.mode,
                    ready=record.ready,
                    updated_at=record.updated_at or datetime.now(UTC),
                )
                session.add(model)
            else:
                model.pr_number = record.pr_number
                model.repo = record.repo
                model.branch = record.branch
                model.ci_passed = record.ci_passed
                model.codex_passed = record.codex_passed
                model.claude_passed = record.claude_passed
                model.gemini_passed = record.gemini_passed
                model.branch_up_to_date = record.branch_up_to_date
                model.screenshots_included = record.screenshots_included
                model.mode = record.mode
                model.ready = record.ready
                model.updated_at = record.updated_at or datetime.now(UTC)
                session.add(model)

            session.flush()
            session.refresh(model)
            return self._model_to_record(model)

    def mark_ready(self, task_id: str, *, ready: bool) -> None:
        with session_scope(self._session_factory) as session:
            model = session.execute(select(ReviewGateModel).where(ReviewGateModel.task_id == task_id)).scalar_one_or_none()
            if model is None:
                return
            model.ready = ready
            model.updated_at = datetime.now(UTC)
            session.add(model)

    def list_review_gates(self, review_filter: ReviewGateFilter | None = None) -> Iterable[ReviewGateRecord]:
        with session_scope(self._session_factory) as session:
            models = session.execute(self._query(review_filter)).scalars().all()
            return [self._model_to_record(model) for model in models]
