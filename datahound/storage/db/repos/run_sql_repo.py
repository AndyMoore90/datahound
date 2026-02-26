"""SQLAlchemy-backed pipeline run repository implementation."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from ..engine import SessionFactory, session_scope
from ..models import DatasetVersionModel, DatasetVersionRecord, PipelineRunModel, PipelineRunRecord
from .run_repo import PipelineRunFilter, RunRepository


class SQLRunRepository(RunRepository):
    """Persist pipeline runs and dataset versions via SQLAlchemy."""

    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    @staticmethod
    def _model_to_run_record(model: PipelineRunModel) -> PipelineRunRecord:
        return PipelineRunRecord(
            id=model.id,
            run_id=model.run_id,
            company=model.company,
            pipeline_name=model.pipeline_name,
            stage=model.stage,
            status=model.status,
            input_manifest_json=model.input_manifest_json,
            output_manifest_json=model.output_manifest_json,
            error_json=model.error_json,
            started_at=model.started_at,
            finished_at=model.finished_at,
        )

    @staticmethod
    def _model_to_dataset_record(model: DatasetVersionModel) -> DatasetVersionRecord:
        return DatasetVersionRecord(
            id=model.id,
            dataset_name=model.dataset_name,
            company=model.company,
            version_tag=model.version_tag,
            schema_version=model.schema_version,
            file_path=Path(model.file_path),
            row_count=model.row_count,
            checksum=model.checksum,
            produced_by_run_id=model.produced_by_run_id,
            created_at=model.created_at,
        )

    def _run_query(self, run_filter: PipelineRunFilter | None = None) -> Select[tuple[PipelineRunModel]]:
        stmt = select(PipelineRunModel)
        if run_filter:
            if run_filter.company:
                stmt = stmt.where(PipelineRunModel.company == run_filter.company)
            if run_filter.pipeline_name:
                stmt = stmt.where(PipelineRunModel.pipeline_name == run_filter.pipeline_name)
            if run_filter.status:
                stmt = stmt.where(PipelineRunModel.status == run_filter.status)
        return stmt

    def _get_run_model(self, session: Session, run_id: str) -> PipelineRunModel | None:
        return session.execute(self._run_query().where(PipelineRunModel.run_id == run_id)).scalar_one_or_none()

    def start_pipeline_run(self, run: PipelineRunRecord) -> PipelineRunRecord:
        with session_scope(self._session_factory) as session:
            model = PipelineRunModel(
                run_id=run.run_id,
                company=run.company,
                pipeline_name=run.pipeline_name,
                stage=run.stage,
                status=run.status,
                input_manifest_json=run.input_manifest_json,
                output_manifest_json=run.output_manifest_json,
                error_json=run.error_json,
                started_at=run.started_at,
                finished_at=run.finished_at,
            )
            session.add(model)
            session.flush()
            session.refresh(model)
            return self._model_to_run_record(model)

    def finish_pipeline_run(
        self,
        run_id: str,
        *,
        status: str,
        output_manifest: Mapping[str, object] | None,
        error: Mapping[str, object] | None = None,
    ) -> PipelineRunRecord:
        with session_scope(self._session_factory) as session:
            model = self._get_run_model(session, run_id)
            if model is None:
                raise ValueError(f"Pipeline run not found: {run_id}")
            model.status = status
            model.output_manifest_json = dict(output_manifest) if output_manifest else None
            model.error_json = dict(error) if error else None
            model.finished_at = datetime.now(UTC)
            session.add(model)
            session.flush()
            session.refresh(model)
            return self._model_to_run_record(model)

    def register_dataset_version(self, dataset: DatasetVersionRecord) -> DatasetVersionRecord:
        with session_scope(self._session_factory) as session:
            model = DatasetVersionModel(
                dataset_name=dataset.dataset_name,
                company=dataset.company,
                version_tag=dataset.version_tag,
                schema_version=dataset.schema_version,
                file_path=str(dataset.file_path),
                row_count=dataset.row_count,
                checksum=dataset.checksum,
                produced_by_run_id=dataset.produced_by_run_id,
                created_at=dataset.created_at or datetime.now(UTC),
            )
            session.add(model)
            session.flush()
            session.refresh(model)
            return self._model_to_dataset_record(model)

    def list_runs(self, run_filter: PipelineRunFilter | None = None) -> Iterable[PipelineRunRecord]:
        with session_scope(self._session_factory) as session:
            models = session.execute(self._run_query(run_filter)).scalars().all()
            return [self._model_to_run_record(model) for model in models]

    def get_run(self, run_id: str) -> PipelineRunRecord | None:
        with session_scope(self._session_factory) as session:
            model = self._get_run_model(session, run_id)
            return self._model_to_run_record(model) if model else None
