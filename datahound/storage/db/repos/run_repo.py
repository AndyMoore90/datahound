"""Pipeline run repository interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

from ..models import DatasetVersionRecord, PipelineRunRecord


@dataclass(slots=True)
class PipelineRunFilter:
    company: str | None = None
    pipeline_name: str | None = None
    status: str | None = None


class RunRepository(Protocol):
    """Handles pipeline run + manifest persistence."""

    def start_pipeline_run(self, run: PipelineRunRecord) -> PipelineRunRecord:
        ...

    def finish_pipeline_run(
        self,
        run_id: str,
        *,
        status: str,
        output_manifest: Mapping[str, object] | None,
        error: Mapping[str, object] | None = None,
    ) -> PipelineRunRecord:
        ...

    def register_dataset_version(self, dataset: DatasetVersionRecord) -> DatasetVersionRecord:
        ...

    def list_runs(self, run_filter: PipelineRunFilter | None = None) -> Iterable[PipelineRunRecord]:
        ...

    def get_run(self, run_id: str) -> PipelineRunRecord | None:
        ...
