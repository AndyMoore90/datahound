from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AuditChange:
    id_value: str
    column: str
    old_value: str
    new_value: str


@dataclass
class UpsertResult:
    type_name: str
    examined_rows: int
    updated_rows: int
    new_rows: int
    audit_changes: List[AuditChange] = field(default_factory=list)
    events_emitted: int = 0
    output_master_path: Optional[Path] = None
    # timings and flags
    ms_read_master: int = 0
    ms_read_prepared: int = 0
    ms_diff: int = 0
    ms_write: int = 0
    used_inplace_write: bool = False
    dry_run: bool = False


@dataclass
class UpsertConfig:
    id_column_by_type: Dict[str, str] = field(default_factory=dict)
    event_rules: Dict[str, Dict] = field(default_factory=dict)


