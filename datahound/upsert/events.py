from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


@dataclass
class Event:
    type_name: str
    id_value: str
    payload: Dict


def detect_job_cancellations(master_df: pd.DataFrame, merged: pd.DataFrame, id_col: str, rules: Dict[str, Any] | None = None) -> List[Event]:
    events: List[Event] = []
    status_col = (rules or {}).get("status_column", "Status")
    # rows where status changed to Canceled
    # merged contains columns like: <col>_x (master), <col>_y (prepared)
    mcol = f"{status_col}_x"
    pcol = f"{status_col}_y"
    if mcol not in merged.columns or pcol not in merged.columns:
        return events
    to_value = (rules or {}).get("to", "Canceled")
    from_exclude = set((rules or {}).get("from_exclude", [to_value]))
    # normalize to strings for comparison
    lhs = merged[mcol].astype(str)
    rhs = merged[pcol].astype(str)
    changed = merged[(lhs != rhs) & (rhs == str(to_value)) & (~lhs.isin({str(v) for v in from_exclude}))]
    for _, row in changed.iterrows():
        idv = str(row[id_col])
        events.append(Event(type_name="cancellation", id_value=idv, payload={"job_id": idv}))
    return events


def detect_unsold_estimates(merged: pd.DataFrame, id_col: str, rules: Dict[str, Any] | None = None) -> List[Event]:
    events: List[Event] = []
    status_col = (rules or {}).get("status_column", "Estimate Status")
    summary_col = (rules or {}).get("summary_column", "Estimate Summary")
    include_values = set((rules or {}).get("include", ["Dismissed", "Open"]))
    exclude_substrings = [str(s) for s in (rules or {}).get("exclude_substrings", ["This is an empty"])]
    mcol = f"{status_col}_x"
    pcol = f"{status_col}_y"
    if mcol not in merged.columns or pcol not in merged.columns:
        return events
    rhs = merged[pcol].astype(str)
    candidates = merged[rhs.isin({str(v) for v in include_values})]
    for _, row in candidates.iterrows():
        summary = str(row.get(f"{summary_col}_y", ""))
        if any(ex in summary for ex in exclude_substrings):
            continue
        idv = str(row[id_col])
        events.append(Event(type_name="unsold_estimate", id_value=idv, payload={"estimate_id": idv}))
    return events


def write_events(company: str, data_dir: Path, events: List[Event]) -> int:
    if not events:
        return 0
    base_dir = Path(data_dir).parent
    events_dir = base_dir / "events"
    logs_dir = base_dir / "logs"
    events_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts_date = datetime.now(UTC).strftime("%Y-%m-%d")
    log_path = logs_dir / "events_log.jsonl"
    counts: Dict[str, int] = {}
    for ev in events:
        counts[ev.type_name] = counts.get(ev.type_name, 0) + 1
        csv_path = events_dir / f"{ev.type_name}_events_{ts_date}.csv"
        write_event_csv(csv_path, ev.payload)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(UTC).isoformat(),
                "company": company,
                "event_type": ev.type_name,
                "payload": ev.payload,
            }) + "\n")
    return sum(counts.values())


def write_event_csv(csv_path: Path, payload: Dict) -> None:
    import csv
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=payload.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(payload)


