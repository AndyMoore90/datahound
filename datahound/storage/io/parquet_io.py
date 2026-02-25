"""Interfaces for interacting with parquet datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

try:  # Optional dependency hint for type checkers only.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas not required for scaffolding
    pd = None  # type: ignore


@dataclass(slots=True)
class ParquetWriteOptions:
    """Options controlling how parquet outputs are materialized."""

    overwrite: bool = False
    partition_cols: Sequence[str] | None = None
    compression: str | None = None


class ParquetIO(Protocol):
    """Protocol describing parquet helpers."""

    def read_table(self, path: Path, columns: Sequence[str] | None = None) -> "pd.DataFrame":
        ...

    def write_table(
        self,
        path: Path,
        frame: "pd.DataFrame",
        *,
        options: ParquetWriteOptions | None = None,
    ) -> Path:
        ...

    def atomic_replace(self, final_path: Path, temp_dir: Path) -> Path:
        ...

    def list_partitions(self, dataset_root: Path) -> Iterable[Path]:
        ...
