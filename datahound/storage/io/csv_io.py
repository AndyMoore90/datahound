"""Interfaces for interacting with CSV assets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


@dataclass(slots=True)
class CsvOptions:
    delimiter: str = ","
    quotechar: str = '"'
    encoding: str = "utf-8"
    header: Sequence[str] | None = None


class CsvIO(Protocol):
    """Protocol describing csv helpers."""

    def read_csv(self, path: Path, *, options: CsvOptions | None = None) -> "pd.DataFrame":
        ...

    def write_csv(
        self,
        path: Path,
        frame: "pd.DataFrame",
        *,
        options: CsvOptions | None = None,
    ) -> Path:
        ...

    def atomic_write(self, target_path: Path, data: bytes) -> Path:
        ...
