"""Typed records plus SQLAlchemy models for storage control-plane tables.

The dataclasses reflect the conceptual schema in
``docs/STORAGE_REFACTOR_TARGET_ARCHITECTURE.md`` while the SQLAlchemy models
expose concrete table metadata used by the Postgres engine scaffold.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JsonDict = Mapping[str, Any]
MutableJsonDict = MutableMapping[str, Any]


class StorageBase(DeclarativeBase):
    """Declarative base for storage control-plane tables."""


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


# --- SQLAlchemy declarative tables -------------------------------------------


class SchedulerTaskModel(StorageBase):
    """SQLAlchemy table definition for ``scheduler_tasks``."""

    __tablename__ = "scheduler_tasks"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    company: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    config_json: Mapped[JsonDict | None] = mapped_column(JSONB, nullable=True)
    schedule_type: Mapped[str] = mapped_column(String(32), nullable=False)
    schedule_expr: Mapped[str | None] = mapped_column(String(255), nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[SchedulerStatus] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SchedulerRunModel(StorageBase):
    """SQLAlchemy table definition for ``scheduler_runs``."""

    __tablename__ = "scheduler_runs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("scheduler_tasks.id"), nullable=False, index=True
    )
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    success: Mapped[bool | None] = mapped_column(Boolean)
    message: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[JsonDict | None] = mapped_column(JSONB)


class PipelineRunModel(StorageBase):
    """SQLAlchemy table definition for ``pipeline_runs``."""

    __tablename__ = "pipeline_runs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    company: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    pipeline_name: Mapped[str] = mapped_column(String(255), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[SchedulerRunStatus] = mapped_column(String(32), nullable=False, index=True)
    input_manifest_json: Mapped[JsonDict | None] = mapped_column(JSONB)
    output_manifest_json: Mapped[JsonDict | None] = mapped_column(JSONB)
    error_json: Mapped[JsonDict | None] = mapped_column(JSONB)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DatasetVersionModel(StorageBase):
    """SQLAlchemy table definition for ``dataset_versions``."""

    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_name", "company", "version_tag", name="uq_dataset_versions_dataset_company_version"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    company: Mapped[str] = mapped_column(String(255), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    row_count: Mapped[int | None] = mapped_column(Integer)
    checksum: Mapped[str | None] = mapped_column(String(128))
    produced_by_run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EventIndexModel(StorageBase):
    """SQLAlchemy table definition for ``events_index``."""

    __tablename__ = "events_index"
    __table_args__ = (UniqueConstraint("event_id", name="uq_events_index_event_id"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    company: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[EventStatus] = mapped_column(String(32), nullable=False, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_file: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    details_json: Mapped[JsonDict | None] = mapped_column(JSONB)


class ReviewGateModel(StorageBase):
    """SQLAlchemy table definition for ``review_gates``."""

    __tablename__ = "review_gates"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    pr_number: Mapped[int | None] = mapped_column(Integer)
    repo: Mapped[str] = mapped_column(String(255), nullable=False)
    branch: Mapped[str] = mapped_column(String(255), nullable=False)
    ci_passed: Mapped[bool | None] = mapped_column(Boolean)
    codex_passed: Mapped[bool | None] = mapped_column(Boolean)
    claude_passed: Mapped[bool | None] = mapped_column(Boolean)
    gemini_passed: Mapped[bool | None] = mapped_column(Boolean)
    branch_up_to_date: Mapped[bool | None] = mapped_column(Boolean)
    screenshots_included: Mapped[bool | None] = mapped_column(Boolean)
    mode: Mapped[ReviewMode] = mapped_column(String(16), nullable=False)
    ready: Mapped[bool | None] = mapped_column(Boolean)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class NotificationModel(StorageBase):
    """SQLAlchemy table definition for ``notifications``."""

    __tablename__ = "notifications"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    channel: Mapped[str] = mapped_column(String(64), nullable=False)
    target: Mapped[str] = mapped_column(String(255), nullable=False)
    message_type: Mapped[MessageType] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[JsonDict | None] = mapped_column(JSONB)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[NotificationStatus | None] = mapped_column(String(32))
    provider_message_id: Mapped[str | None] = mapped_column(String(255))


class IdempotencyKeyModel(StorageBase):
    """SQLAlchemy table definition for ``idempotency_keys``."""

    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    scope: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
