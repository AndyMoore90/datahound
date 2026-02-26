#!/usr/bin/env python3
"""Control-plane readiness snapshot for Phase 5 cutover."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass

from datahound.storage.bootstrap import get_storage_dal_from_env
from datahound.storage.control_plane import get_control_plane_source
from datahound.storage.db.repos.notification_repo import NotificationFilter
from datahound.storage.db.repos.review_repo import ReviewGateFilter


@dataclass
class ReadinessReport:
    control_plane_source: str
    storage_url_present: bool
    dal_available: bool
    has_scheduler_repo: bool
    has_run_repo: bool
    has_event_repo: bool
    has_review_repo: bool
    has_notification_repo: bool
    review_rows: int
    notification_rows: int
    ok: bool
    notes: list[str]


def main() -> int:
    source = get_control_plane_source()
    storage_url_present = bool(os.getenv("DATAHOUND_STORAGE_URL") or os.getenv("DATABASE_URL"))
    dal = get_storage_dal_from_env()

    has_scheduler = bool(dal and dal.dependencies.scheduler_repo)
    has_run = bool(dal and dal.dependencies.run_repo)
    has_event = bool(dal and dal.dependencies.event_repo)
    has_review = bool(dal and dal.dependencies.review_repo)
    has_notif = bool(dal and dal.dependencies.notification_repo)

    review_rows = 0
    notif_rows = 0
    notes: list[str] = []

    if dal and dal.dependencies.review_repo:
        review_rows = len(list(dal.dependencies.review_repo.list_review_gates(ReviewGateFilter())))
    if dal and dal.dependencies.notification_repo:
        notif_rows = len(list(dal.dependencies.notification_repo.list_notifications(NotificationFilter())))

    ok = True
    if source == "db" and not dal:
        ok = False
        notes.append("control-plane source is db but DAL is unavailable")
    if source == "db" and not (has_scheduler and has_run and has_event and has_review and has_notif):
        ok = False
        notes.append("one or more required DB repos are missing")
    if source == "json":
        notes.append("control-plane forced to json mode")
    if source == "auto" and not dal:
        notes.append("auto mode currently using json fallback")

    report = ReadinessReport(
        control_plane_source=source,
        storage_url_present=storage_url_present,
        dal_available=bool(dal),
        has_scheduler_repo=has_scheduler,
        has_run_repo=has_run,
        has_event_repo=has_event,
        has_review_repo=has_review,
        has_notification_repo=has_notif,
        review_rows=review_rows,
        notification_rows=notif_rows,
        ok=ok,
        notes=notes,
    )

    print(json.dumps(asdict(report), indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
