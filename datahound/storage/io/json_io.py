"""Interfaces for interacting with JSON + JSONL artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Protocol, Sequence

JsonDict = MutableMapping[str, Any]


class JsonIO(Protocol):
    """Protocol describing structured JSON IO helpers."""

    def load_json(self, path: Path) -> JsonDict:
        ...

    def dump_json(self, path: Path, payload: Mapping[str, Any]) -> Path:
        ...

    def append_jsonl(self, path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
        ...

    def validate(self, payload: Mapping[str, Any], schema: Mapping[str, Any]) -> Sequence[str]:
        """Return validation error messages (empty when valid)."""
        ...
