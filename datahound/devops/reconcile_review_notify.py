#!/usr/bin/env python3
"""Compare swarm auto-merge JSON logs against DB review/notification records."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datahound.storage.bootstrap import get_storage_dal_from_env
from datahound.storage.db.repos.notification_repo import NotificationFilter
from datahound.storage.db.repos.review_repo import ReviewGateFilter

LOG_PATH = Path("logging/cron_monitor/swarm_auto_merge.jsonl")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> int:
    dal = get_storage_dal_from_env()
    if dal is None:
        print("[FAIL] DATAHOUND_STORAGE_URL not configured; DAL unavailable")
        return 1

    log_rows = load_jsonl(LOG_PATH)
    pr_ids = {f"pr:{row.get('pr_number')}" for row in log_rows if row.get("pr_number")}

    review_rows = list(dal.dependencies.review_repo.list_review_gates(ReviewGateFilter())) if dal.dependencies.review_repo else []
    notif_rows = list(dal.dependencies.notification_repo.list_notifications(NotificationFilter())) if dal.dependencies.notification_repo else []

    review_task_ids = {r.task_id for r in review_rows}
    notif_task_ids = {n.task_id for n in notif_rows}

    missing_review = sorted(pr_ids - review_task_ids)
    missing_notif = sorted(pr_ids - notif_task_ids)

    print("reconciliation_summary")
    print(json.dumps({
        "log_records": len(log_rows),
        "unique_pr_tasks": len(pr_ids),
        "db_review_rows": len(review_rows),
        "db_notification_rows": len(notif_rows),
        "missing_review_tasks": missing_review[:20],
        "missing_notification_tasks": missing_notif[:20],
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
