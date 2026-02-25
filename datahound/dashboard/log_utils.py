"""Utilities for loading and aggregating pipeline logs for the Events Dashboard."""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

LOG_ENCODING = "utf-8"
TIMESTAMP_FIELDS = ("ts", "timestamp", "time", "created_at")

# --------- Helpers ---------


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        dt = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(dt):
        return None
    return dt.tz_localize(None)


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding=LOG_ENCODING, errors="ignore").splitlines()
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    iterable: Iterable[str] = lines if limit is None else lines[-limit:]
    for line in iterable:
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _normalize_cutoff(cutoff: datetime) -> datetime:
    if cutoff.tzinfo:
        return cutoff.astimezone().replace(tzinfo=None)
    return cutoff


def _clear_log_file(path: Path, cutoff: Optional[datetime], direction: str) -> None:
    if not path.exists():
        return
    if cutoff is None:
        path.unlink(missing_ok=True)
        return
    cutoff_clean = _normalize_cutoff(cutoff)
    trimmed: List[str] = []
    for raw in path.read_text(encoding=LOG_ENCODING, errors="ignore").splitlines():
        try:
            record = json.loads(raw)
        except Exception:
            continue
        timestamp: Optional[datetime] = None
        for field in TIMESTAMP_FIELDS:
            if field in record:
                timestamp = _parse_timestamp(record[field])
                if timestamp:
                    break
        if not timestamp:
            continue
        if direction == "after":
            if timestamp <= cutoff_clean:
                trimmed.append(json.dumps(record))
        else:
            if timestamp >= cutoff_clean:
                trimmed.append(json.dumps(record))
    if trimmed:
        path.write_text("\n".join(trimmed) + "\n", encoding=LOG_ENCODING)
    else:
        path.unlink(missing_ok=True)


def _path(company: str, *parts: str) -> Path:
    return Path("data") / company / "logs" / Path(*parts)


def _central_path(company: str, process: str, *parts: str) -> Path:
    from central_logging.config import LOG_ROOT
    return LOG_ROOT / process / company.replace(" ", "_").replace("/", "_") / Path(*parts)


def _normalize_timestamps(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], utc=True, errors="coerce").dt.tz_localize(None)
    return df


# --------- Loaders ---------


def load_download_logs(company: str, limit: int = 5000) -> pd.DataFrame:
    from central_logging.config import pipeline_dir
    path = pipeline_dir(company) / "download.jsonl"
    if not path.exists():
        path = _path(company, "pipeline", "download.jsonl")
    rows = _read_jsonl(path, limit)
    if not rows:
        return pd.DataFrame(columns=["ts", "company", "file_type", "status", "filename"])
    df = pd.DataFrame(rows)
    df = _normalize_timestamps(df, "ts")
    df["stage"] = "download"
    return df


def load_prepare_logs(company: str, limit: int = 5000) -> pd.DataFrame:
    from central_logging.config import pipeline_dir
    path = pipeline_dir(company) / "prepare.jsonl"
    if not path.exists():
        path = Path("data") / company / "downloads" / "logs" / "prepare_log.jsonl"
    rows = _read_jsonl(path, limit)
    if not rows:
        return pd.DataFrame(columns=["ts", "company", "file_type", "status", "source"])
    df = pd.DataFrame(rows)
    df = _normalize_timestamps(df, "ts")
    df["stage"] = "prepare"
    return df


def load_upsert_logs(company: str, limit: int = 20000) -> pd.DataFrame:
    from central_logging.config import pipeline_dir
    path = pipeline_dir(company) / "integrated_upsert.jsonl"
    if not path.exists():
        path = _path(company, "integrated_upsert_log.jsonl")
    rows = _read_jsonl(path, limit)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "level", "message", "file_type"])
    df = pd.DataFrame(rows)
    df = _normalize_timestamps(df, "timestamp")
    df.rename(columns={"timestamp": "ts"}, inplace=True)
    df["stage"] = "integrated_upsert"
    return df


def load_upsert_operations(company: str, limit: int = 2000) -> pd.DataFrame:
    rows = _read_jsonl(_path(company, "integrated_upsert_operations.jsonl"), limit)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "operation", "duration_seconds"])
    df = pd.DataFrame(rows)
    df = _normalize_timestamps(df, "timestamp")
    df.rename(columns={"timestamp": "ts"}, inplace=True)
    df["stage"] = "integrated_upsert_operations"
    return df


def load_core_data_logs(company: str, limit: int = 2000) -> pd.DataFrame:
    rows = _read_jsonl(_path(company, "core_data_operations.jsonl"), limit)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "operation", "duration_seconds"])
    df = pd.DataFrame(rows)
    df = _normalize_timestamps(df, "timestamp")
    df.rename(columns={"timestamp": "ts"}, inplace=True)
    df["stage"] = "core_data"
    return df


def load_historical_scan_logs(company: str, limit: int = 5000) -> pd.DataFrame:
    from central_logging.config import event_detection_historical_dir
    frames: List[pd.DataFrame] = []
    for event_type in ["canceled_jobs", "unsold_estimates", "lost_customers", "overdue_maintenance"]:
        file_path = event_detection_historical_dir(event_type, company) / "scan.jsonl"
        if not file_path.exists():
            file_path = _path(company, "historical", event_type, "scan.jsonl")
        rows = _read_jsonl(file_path, limit)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = _normalize_timestamps(df, "ts")
        df["event_type"] = event_type
        df["stage"] = "historical_scan"
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ts", "company", "action", "event_type", "stage"])
    return pd.concat(frames, ignore_index=True)


def load_recent_event_changes(company: str) -> pd.DataFrame:
    path = _path(company, "recent_event_changes.json")
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "event", "added_rows"])
    try:
        data = json.loads(path.read_text(encoding=LOG_ENCODING))
    except Exception:
        return pd.DataFrame(columns=["timestamp", "event", "added_rows"])
    df = pd.DataFrame(data)
    df = _normalize_timestamps(df, "timestamp")
    df.rename(columns={"timestamp": "ts"}, inplace=True)
    df["stage"] = "recent_events"
    return df


def load_custom_extraction_logs(company: str | None = None, limit: int = 5000) -> pd.DataFrame:
    from central_logging.config import extraction_dir
    path = extraction_dir(company) / "custom_extraction.jsonl" if company else None
    if path is None or not path.exists():
        path = Path("data/logs/custom_extraction_log.jsonl")
    rows = _read_jsonl(path, limit)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "company", "message", "details"])
    df = pd.DataFrame(rows)
    df = _normalize_timestamps(df, "timestamp")
    df.rename(columns={"timestamp": "ts"}, inplace=True)
    df["stage"] = "custom_extraction"
    return df


def load_scheduler_history(limit: int = 5000) -> pd.DataFrame:
    from central_logging.config import scheduler_dir
    path = scheduler_dir() / "task_history.jsonl"
    if not path.exists():
        path = Path("data/scheduler/task_history.jsonl")
    rows = _read_jsonl(path, limit)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "task_id", "success"])
    df = pd.DataFrame(rows)
    df = _normalize_timestamps(df, "timestamp")
    df.rename(columns={"timestamp": "ts"}, inplace=True)
    df["stage"] = "scheduler"
    return df


# --------- Aggregated API ---------


@lru_cache(maxsize=16)
def load_dashboard_data(company: str, limit: int = 5000) -> Dict[str, pd.DataFrame]:
    """Return a mapping of stage -> DataFrame with normalized log data."""

    data: Dict[str, pd.DataFrame] = {}

    data["download"] = load_download_logs(company, limit)
    data["prepare"] = load_prepare_logs(company, limit)
    data["integrated_upsert"] = load_upsert_logs(company, limit)
    data["integrated_upsert_operations"] = load_upsert_operations(company, limit)
    data["core_data"] = load_core_data_logs(company, limit)
    data["historical_scan"] = load_historical_scan_logs(company, limit)
    data["recent_events"] = load_recent_event_changes(company)
    data["custom_extraction"] = load_custom_extraction_logs(company, limit)
    data["scheduler"] = load_scheduler_history(limit)

    # Add change logs per entity
    for name in [
        "job_changes_log.jsonl",
        "customer_changes_log.jsonl",
        "call_changes_log.jsonl",
        "estimate_changes_log.jsonl",
        "invoice_changes_log.jsonl",
        "location_changes_log.jsonl",
        "membership_changes_log.jsonl",
    ]:
        rows = _read_jsonl(_path(company, name), limit)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = _normalize_timestamps(df, "ts")
        df["stage"] = "change_log"
        df["source_file"] = name
        data[f"change_log::{name.split('_')[0]}"] = df

    return data


def clear_cache() -> None:
    """Clear cached dashboard data."""

    load_dashboard_data.cache_clear()  # type: ignore[attr-defined]


def clear_logs(company: str, cutoff: Optional[datetime] = None, direction: str = "before") -> None:
    from central_logging.config import pipeline_dir, scheduler_dir, event_detection_historical_dir, extraction_dir
    targets: List[Tuple[Path, bool]] = []
    base = Path("data") / company
    plog = pipeline_dir(company)
    targets.append((plog / "download.jsonl", True))
    targets.append((plog / "prepare.jsonl", True))
    targets.append((plog / "integrated_upsert.jsonl", True))
    targets.append((plog / "core_data.jsonl", True))
    targets.append((plog / "customer_profile_build.jsonl", True))
    targets.append((base / "logs" / "integrated_upsert_operations.jsonl", True))
    targets.append((base / "logs" / "core_data_operations.jsonl", True))
    for sub in ["canceled_jobs", "unsold_estimates", "lost_customers", "overdue_maintenance"]:
        targets.append((event_detection_historical_dir(sub, company) / "scan.jsonl", True))
    for name in [
        "job_changes_log.jsonl",
        "customer_changes_log.jsonl",
        "call_changes_log.jsonl",
        "estimate_changes_log.jsonl",
        "invoice_changes_log.jsonl",
        "location_changes_log.jsonl",
        "membership_changes_log.jsonl",
    ]:
        targets.append((base / "logs" / name, True))
    targets.append((base / "downloads" / "logs" / "prepare_log.jsonl", True))
    targets.append((extraction_dir(company) / "custom_extraction.jsonl", True))
    targets.append((scheduler_dir() / "task_history.jsonl", True))
    for path, needs_dir in targets:
        if not path.exists():
            continue
        if cutoff is None:
            path.unlink(missing_ok=True)
            continue
        _clear_log_file(path, cutoff=cutoff, direction=direction)
    clear_cache()

