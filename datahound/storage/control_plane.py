"""Control-plane source selection helpers.

DATAHOUND_CONTROL_PLANE_SOURCE:
- db: prefer DB (default when DAL available)
- json: force JSON/log source
- auto: DB if available, else JSON
"""
from __future__ import annotations

import os

from .bootstrap import get_storage_dal_from_env


def get_control_plane_source() -> str:
    mode = (os.getenv("DATAHOUND_CONTROL_PLANE_SOURCE") or "auto").strip().lower()
    if mode not in {"auto", "db", "json"}:
        mode = "auto"

    if mode == "json":
        return "json"

    dal = get_storage_dal_from_env()
    if dal is not None:
        return "db"

    return "json"
