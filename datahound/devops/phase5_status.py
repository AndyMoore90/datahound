#!/usr/bin/env python3
"""Human-friendly Phase 5 status snapshot."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict


@dataclass
class Item:
    name: str
    ok: bool
    detail: str


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    import subprocess

    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = (p.stdout or p.stderr or "").strip()
    return p.returncode, out[:1200]


def main() -> int:
    items: list[Item] = []

    source = (os.getenv("DATAHOUND_CONTROL_PLANE_SOURCE") or "auto").strip().lower()
    items.append(Item("control_plane_source_valid", source in {"auto", "db", "json"}, f"source={source}"))

    has_db = bool(os.getenv("DATAHOUND_STORAGE_URL") or os.getenv("DATABASE_URL"))
    items.append(Item("db_url_present", has_db, "DATAHOUND_STORAGE_URL/DATABASE_URL present" if has_db else "missing DB URL"))

    checks = [
        ("preflight", ["python3", "-m", "datahound.preflight"]),
        ("control_plane_readiness", ["python3", "-m", "datahound.devops.control_plane_readiness"]),
        ("control_plane_drift_check", ["python3", "-m", "datahound.devops.control_plane_drift_check", "--max-missing", "0"]),
    ]

    for name, cmd in checks:
        code, out = run_cmd(cmd)
        items.append(Item(name, code == 0, out.splitlines()[-1] if out else f"exit={code}"))

    ok = all(i.ok for i in items)
    payload = {
        "ok": ok,
        "items": [asdict(i) for i in items],
    }
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
