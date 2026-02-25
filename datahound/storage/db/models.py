"""Typed record definitions for the storage control-plane tables.

The classes defined here mirror the conceptual tables in
``docs/STORAGE_REFACTOR_TARGET_ARCHITECTURE.md`` and are intentionally
framework-agnostic. Later phases will map them to SQLAlchemy models.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence
from uuid import UUID

JsonDict = Mapping[str, Any]
MutableJsonDict = MutableMapping[str, Any]

SchedulerStatus = str  # active|paused|deleted
SchedulerRunStatus = str  # running|success|failed
EventStatus = str  # active|resolved|archived
NotificationStatus = str  # sent|failed
ReviewMode = str  # strict|soft
MessageType = str  # ready|failed|info


@dataclass(slots=True)
class SchedulerTaskRecord:
    """Represents one row from ``scheduler_tasks``."""

    id: UUID | None
    task_key: str
    task_type: str
    company: str
    config_json: JsonDict | None
    schedule_type: str
    schedule_expr: str | None
    timezone: str
    status: SchedulerStatus
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class SchedulerRunRecord:
    """Represents one row from ``scheduler_runs``."""

    id: UUID | None
    task_id: UUID
    run_id: str
    started_at: datetime
    finished_at: datetime | None
    success: bool | None
    message: str | None
    duration_ms: int | None
    metadata_json: JsonDict | None


@dataclass(slots=True)
class PipelineRunRecord:
    """Represents one row from ``pipeline_runs``."""

    id: UUID | None
    run_id: str
    company: str
    pipeline_name: str
    stage: str
    status: SchedulerRunStatus
    input_manifest_json: JsonDict | None
    output_manifest_json: JsonDict | None
    error_json: JsonDict | None
    started_at: datetime
    finished_at: datetime | None


@dataclass(slots=True)
class DatasetVersionRecord:
    """Represents one row from ``dataset_versions``."""

    id: UUID | None
    dataset_name: str
    company: str
    version_tag: str
    schema_version: str
    file_path: Path
    row_count: int | None
    checksum: str | None
    produced_by_run_id: str | None
    created_at: datetime | None


@dataclass(slots=True)
class EventIndexRecord:
    """Represents one row from ``events_index``."""

    id: UUID | None
    event_id: str
    company: str
    event_type: str
    entity_type: str
    entity_id: str
    severity: str
    status: EventStatus
    first_seen_at: datetime
    last_seen_at: datetime
    source_file: str | None
    details_json: JsonDict | None


@dataclass(slots=True)
class ReviewGateRecord:
    """Represents one row from ``review_gates``."""

    id: UUID | None
    task_id: str
    pr_number: int | None
    repo: str
    branch: str
    ci_passed: bool | None
    codex_passed: bool | None
    claude_passed: bool | None
    gemini_passed: bool | None
    branch_up_to_date: bool | None
    screenshots_included: bool | None
    mode: ReviewMode
    ready: bool | None
    updated_at: datetime | None


@dataclass(slots=True)
class NotificationRecord:
    """Represents one row from ``notifications``."""

    id: UUID | None
    task_id: str
    channel: str
    target: str
    message_type: MessageType
    payload_json: JsonDict | None
    sent_at: datetime | None
    status: NotificationStatus | None
    provider_message_id: str | None


@dataclass(slots=True)
class IdempotencyKeyRecord:
    """Represents one row from ``idempotency_keys``."""

    key: str
    scope: str
    run_id: str
    created_at: datetime
    expires_at: datetime | None


@dataclass(slots=True)
class DatasetManifestReference:
    """Lightweight reference tying datasets to manifests."""

    dataset_name: str
    version_tag: str
    manifest_path: Path
    produced_by_run_id: str
    schema_version: str | None = None


@dataclass(slots=True)
class RunManifestEnvelope:
    """Serialized manifests persisted to parquet/json."""

    run_id: str
    pipeline_name: str
    stage: str
    status: SchedulerRunStatus
    input_datasets: Sequence[DatasetManifestReference]
    output_datasets: Sequence[DatasetManifestReference]
    metadata: MutableJsonDict
