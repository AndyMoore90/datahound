#!/usr/bin/env python3
"""Run Phase 5 exit checks in one command."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, asdict


@dataclass
class Check:
    name: str
    command: list[str]
    ok: bool = False
    output: str = ""
    code: int = 0


def run_check(chk: Check) -> Check:
    try:
        proc = subprocess.run(chk.command, capture_output=True, text=True, timeout=120)
        chk.code = proc.returncode
        chk.ok = proc.returncode == 0
        chk.output = (proc.stdout or proc.stderr or "").strip()[:4000]
    except Exception as exc:
        chk.ok = False
        chk.code = 99
        chk.output = str(exc)
    return chk


def main() -> int:
    checks = [
        Check("preflight", [sys.executable, "-m", "datahound.preflight"]),
        Check("control_plane_readiness", [sys.executable, "-m", "datahound.devops.control_plane_readiness"]),
        Check("control_plane_drift_check", [sys.executable, "-m", "datahound.devops.control_plane_drift_check", "--max-missing", "0"]),
    ]

    results = [run_check(c) for c in checks]
    ok = all(c.ok for c in results)

    payload = {
        "ok": ok,
        "checks": [asdict(c) for c in results],
    }
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
