import json
from copy import copy
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, Optional

from .config import (
    transcript_pipeline_dir,
    scheduler_dir,
    event_detection_dir,
    event_detection_historical_dir,
    extraction_dir,
    pipeline_dir,
    upsert_changes_dir,
    permits_dir,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    record["ts"] = datetime.now(UTC).isoformat()
    _ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write(
    process: str,
    log_file: str,
    record: Dict[str, Any],
    company: str | None = None,
    page: str | None = None,
    tab: str | None = None,
) -> None:
    base_record: Dict[str, Any] = {
        "process": process,
        **record,
    }
    if company:
        base_record["company"] = company
    if page:
        base_record["page"] = page
    if tab:
        base_record["tab"] = tab
    path = _path_for(process, log_file, company)
    _write_jsonl(path, base_record)


def _path_for(process: str, log_file: str, company: str | None) -> Path:
    if process == "transcript_pipeline":
        return transcript_pipeline_dir() / log_file
    if process == "scheduler":
        return scheduler_dir() / log_file
    if process == "event_detection":
        return event_detection_dir(company) / log_file
    if process == "extraction":
        return extraction_dir(company) / log_file
    if process == "pipeline":
        return pipeline_dir(company or "default") / log_file
    if process == "upsert_changes":
        return upsert_changes_dir(company or "default") / log_file
    if process == "permits":
        return permits_dir(company) / log_file
    base = Path(__file__).parent.parent / "logging" / process.replace(" ", "_")
    base.mkdir(parents=True, exist_ok=True)
    return base / log_file


def write_transcript_pipeline(record: Dict[str, Any]) -> None:
    rec = copy(record)
    log_file = rec.pop("file", "pipeline_run.jsonl")
    path = transcript_pipeline_dir() / log_file
    _write_jsonl(path, rec)


def write_scheduler(record: Dict[str, Any]) -> None:
    rec = copy(record)
    log_file = rec.pop("file", "task_history.jsonl")
    path = scheduler_dir() / log_file
    _write_jsonl(path, rec)


def write_event_detection(
    record: Dict[str, Any],
    company: str | None = None,
    rule_name: str | None = None,
) -> None:
    rec = copy(record)
    log_file = rec.pop("file", "scan.jsonl")
    if rule_name:
        base = event_detection_historical_dir(rule_name, company)
    else:
        base = event_detection_dir(company)
    path = base / log_file
    rec["company"] = company
    _write_jsonl(path, rec)


def write_extraction(record: Dict[str, Any], company: str | None = None) -> None:
    rec = copy(record)
    log_file = rec.pop("file", "custom_extraction.jsonl")
    base = extraction_dir(company)
    path = base / log_file
    rec["company"] = company
    _write_jsonl(path, rec)


def write_pipeline(
    log_file: str,
    record: Dict[str, Any],
    company: str,
) -> None:
    base = pipeline_dir(company)
    path = base / log_file
    record["company"] = company
    _write_jsonl(path, record)


def write_upsert_changes(
    log_file: str,
    record: Dict[str, Any],
    company: str,
) -> None:
    base = upsert_changes_dir(company)
    path = base / log_file
    record["company"] = company
    _write_jsonl(path, record)


def write_permits(record: Dict[str, Any], company: str | None = None) -> None:
    rec = copy(record)
    log_file = rec.pop("file", "permit_processing.jsonl")
    base = permits_dir(company)
    path = base / log_file
    rec["company"] = company
    _write_jsonl(path, rec)


def get_log_path(process: str, log_file: str, company: str | None = None) -> Path:
    return _path_for(process, log_file, company)
