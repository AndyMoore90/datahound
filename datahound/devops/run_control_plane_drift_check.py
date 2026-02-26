#!/usr/bin/env python3
"""Cron-friendly wrapper for control-plane drift check."""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    max_missing = os.getenv("DRIFT_MAX_MISSING", "0")
    cmd = [sys.executable, "-m", "datahound.devops.control_plane_drift_check", "--max-missing", max_missing]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
