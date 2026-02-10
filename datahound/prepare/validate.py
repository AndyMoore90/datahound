from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class ValidationIssue:
    kind: str
    detail: str


def validate_against_master(prepared_file: Path, master_columns: List[str]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    try:
        suf = prepared_file.suffix.lower()
        if suf == ".parquet":
            df = pd.read_parquet(prepared_file)
        elif suf == ".csv":
            df = pd.read_csv(prepared_file, nrows=1)
        else:
            df = pd.read_excel(prepared_file, nrows=1)
        cols = list(df.columns)
        if cols != master_columns:
            issues.append(ValidationIssue(
                kind="columns_mismatch",
                detail=f"expected={master_columns} got={cols}"
            ))
    except Exception as e:
        issues.append(ValidationIssue(kind="read_error", detail=str(e)))
    return issues


def write_validation_log(log_dir: Path, company: str, file_type: str, source: str, issues: List[ValidationIssue]) -> None:
    # all logs under data/<company>/logs/pipeline
    logs_root = log_dir.parent / "logs"
    pipeline_dir = logs_root / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    log_path = pipeline_dir / "prepare_validation.jsonl"
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "company": company,
        "file_type": file_type,
        "source": source,
        "status": "ok" if not issues else "issues",
        "issues": [issue.__dict__ for issue in issues],
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


