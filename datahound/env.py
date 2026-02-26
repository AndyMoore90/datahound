"""Lightweight .env loader for local runtime reliability."""
from __future__ import annotations

import os
from pathlib import Path


def load_env_fallback(root: Path | None = None) -> int:
    """Load KEY=VALUE pairs from .env into os.environ if missing.

    Returns number of keys injected.
    """
    base = root or Path(__file__).resolve().parents[1]
    env_path = base / ".env"
    if not env_path.exists():
        return 0

    injected = 0
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            injected += 1
    return injected
