"""Manifest abstraction shared across pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

ManifestStatus = str  # running|success|failed


@dataclass(slots=True)
class ManifestEntry:
    """Represents metadata about a single dataset artifact."""

    dataset_name: str
    version_tag: str
    file_path: Path
    schema_version: str | None = None
    row_count: int | None = None
    checksum: str | None = None
    extras: Mapping[str, Any] | None = None

    def as_dict(self) -> MutableMapping[str, Any]:
        """Return a JSON-serializable representation."""

        payload: MutableMapping[str, Any] = {
            "dataset_name": self.dataset_name,
            "version_tag": self.version_tag,
            "file_path": str(self.file_path),
        }
        if self.schema_version is not None:
            payload["schema_version"] = self.schema_version
        if self.row_count is not None:
            payload["row_count"] = self.row_count
        if self.checksum is not None:
            payload["checksum"] = self.checksum
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload


@dataclass(slots=True)
class DatasetManifest:
    """Groups all entries related to the same dataset."""

    dataset_name: str
    entries: Sequence[ManifestEntry]
    produced_by_run_id: str
    created_at: datetime

    def as_dict(self) -> MutableMapping[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "produced_by_run_id": self.produced_by_run_id,
            "created_at": self.created_at.isoformat(),
            "entries": [entry.as_dict() for entry in self.entries],
        }


@dataclass(slots=True)
class RunManifest:
    """Captures the lifecycle of a single pipeline run."""

    run_id: str
    pipeline_name: str
    company: str
    stage: str
    status: ManifestStatus
    started_at: datetime
    finished_at: datetime | None = None
    input_datasets: Sequence[DatasetManifest] = field(default_factory=tuple)
    output_datasets: Sequence[DatasetManifest] = field(default_factory=tuple)
    error: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "company": self.company,
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "input_datasets": [d.as_dict() for d in self.input_datasets],
            "output_datasets": [d.as_dict() for d in self.output_datasets],
        }
        if self.finished_at:
            payload["finished_at"] = self.finished_at.isoformat()
        if self.error:
            payload["error"] = dict(self.error)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload
