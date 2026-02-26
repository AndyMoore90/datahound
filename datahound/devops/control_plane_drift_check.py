#!/usr/bin/env python3
"""Control-plane drift check between JSONL exports and DB records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datahound.storage.bootstrap import get_storage_dal_from_env
from datahound.storage.db.repos.notification_repo import NotificationFilter
from datahound.storage.db.repos.review_repo import ReviewGateFilter

DEFAULT_LOG = Path("logging/cron_monitor/swarm_auto_merge.jsonl")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check JSONL-vs-DB drift for review/notify control-plane records")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--max-missing", type=int, default=0, help="Allowed missing task ids before failing")
    args = ap.parse_args()

    dal = get_storage_dal_from_env()
    if dal is None:
        print("[FAIL] DAL unavailable. Set DATAHOUND_STORAGE_URL.")
        return 2

    logs = load_jsonl(args.log)
    json_tasks = {f"pr:{r.get('pr_number')}" for r in logs if r.get("pr_number")}

    review_repo = dal.dependencies.review_repo
    notif_repo = dal.dependencies.notification_repo
    if review_repo is None or notif_repo is None:
        print("[FAIL] review/notification repos not configured in DAL")
        return 2

    db_review_tasks = {r.task_id for r in review_repo.list_review_gates(ReviewGateFilter())}
    db_notif_tasks = {n.task_id for n in notif_repo.list_notifications(NotificationFilter())}

    missing_review = sorted(json_tasks - db_review_tasks)
    missing_notif = sorted(json_tasks - db_notif_tasks)
    missing_total = max(len(missing_review), len(missing_notif))

    summary = {
        "json_tasks": len(json_tasks),
        "db_review_tasks": len(db_review_tasks),
        "db_notification_tasks": len(db_notif_tasks),
        "missing_review": len(missing_review),
        "missing_notification": len(missing_notif),
        "sample_missing_review": missing_review[:10],
        "sample_missing_notification": missing_notif[:10],
        "max_missing_allowed": args.max_missing,
    }
    print(json.dumps(summary, indent=2))

    if missing_total > args.max_missing:
        print("[FAIL] drift above threshold")
        return 1
    print("[PASS] drift within threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
