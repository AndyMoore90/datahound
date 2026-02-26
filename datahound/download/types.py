from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any
import json
from datahound.upsert.types import UpsertConfig


@dataclass
class GmailConfig:
    scopes: List[str]
    credentials_path: Path
    token_path: Path
    query_by_type: Dict[str, str]
    link_prefixes: List[str]


@dataclass
class PermitConfig:
    austin_base_url: str = "https://data.austintexas.gov/resource/3syk-w9eu.csv"
    default_lookback_hours: int = 168


@dataclass
class ScheduleConfig:
    id: str
    enabled: bool
    job: str  # 'gmail' | 'permit_austin'
    interval_minutes: int
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DownloadConfig:
    company: str
    data_dir: Path
    gmail: GmailConfig
    allowed_extensions: List[str] = field(default_factory=lambda: [".xlsx", ".xls", ".csv"])
    mark_as_read: bool = True
    permit: PermitConfig = field(default_factory=PermitConfig)
    schedules: List[ScheduleConfig] = field(default_factory=list)
    prepare: "PrepareConfig" | None = None
    upsert: UpsertConfig = field(default_factory=UpsertConfig)


@dataclass
class PrepareTypeRule:
    drop_last_row: bool = True
    column_renames: Dict[str, str] = field(default_factory=dict)
    date_columns: List[str] = field(default_factory=list)
    date_formats: List[str] = field(default_factory=list)
    date_placeholders: List[str] = field(default_factory=list)
    excel_serial: bool = False


@dataclass
class PrepareConfig:
    tables_dir: Path
    file_type_to_master: Dict[str, str]
    type_rules: Dict[str, PrepareTypeRule] = field(default_factory=dict)


def _norm_path(value: str) -> Path:
    """Normalize config paths so Windows-style separators work on Linux/macOS too."""
    return Path(str(value).replace("\\", "/"))


def load_config(config_path: Path) -> DownloadConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    gmail_cfg = cfg["gmail"]
    schedules_cfg: List[ScheduleConfig] = []
    for s in cfg.get("schedules", []):
        schedules_cfg.append(ScheduleConfig(
            id=s.get("id", "default"),
            enabled=bool(s.get("enabled", False)),
            job=s.get("job", "gmail"),
            interval_minutes=int(s.get("interval_minutes", 60)),
            params=s.get("params", {}),
        ))
    # Prepare defaults
    default_mapping = {
        "jobs": "Jobs.xlsx",
        "customers": "Customers.xlsx",
        "memberships": "Memberships.xlsx",
        "calls": "Calls.xlsx",
        "estimates": "Estimates.xlsx",
        "invoices": "Invoices.xlsx",
        "locations": "Locations.xlsx",
    }
    prepare_cfg = cfg.get("prepare", {})
    tables_dir_default = Path(f"companies/{cfg['company']}/tables")
    rules_cfg = prepare_cfg.get("type_rules", {})
    rules: Dict[str, PrepareTypeRule] = {}
    for k, v in rules_cfg.items():
        rules[k] = PrepareTypeRule(
            drop_last_row=bool(v.get("drop_last_row", True)),
            column_renames=v.get("column_renames", {}),
            date_columns=v.get("date_columns", []),
            date_formats=v.get("date_formats", []),
            date_placeholders=v.get("date_placeholders", []),
            excel_serial=bool(v.get("excel_serial", False)),
        )

    # Upsert defaults
    default_id_map = {
        "jobs": "Job ID",
        "customers": "Customer ID",
        "memberships": "Membership ID",
        "calls": "Call ID",
        "estimates": "Estimate ID",
        "invoices": "Invoice ID",
        "locations": "Location ID",
    }
    upsert_cfg = cfg.get("upsert", {})

    return DownloadConfig(
        company=cfg["company"],
        data_dir=_norm_path(cfg["data_dir"]),
        gmail=GmailConfig(
            scopes=gmail_cfg["scopes"],
            credentials_path=_norm_path(gmail_cfg["credentials_path"]),
            token_path=_norm_path(gmail_cfg["token_path"]),
            query_by_type=gmail_cfg["query_by_type"],
            link_prefixes=gmail_cfg.get("link_prefixes", []),
        ),
        allowed_extensions=cfg.get("allowed_extensions", [".xlsx", ".xls", ".csv"]),
        mark_as_read=cfg.get("mark_as_read", True),
        permit=PermitConfig(
            austin_base_url=cfg.get("permit", {}).get("austin_base_url", "https://data.austintexas.gov/resource/3syk-w9eu.csv"),
            default_lookback_hours=int(cfg.get("permit", {}).get("default_lookback_hours", 168)),
        ),
        schedules=schedules_cfg,
        prepare=PrepareConfig(
            tables_dir=_norm_path(prepare_cfg.get("tables_dir", str(tables_dir_default))),
            file_type_to_master=prepare_cfg.get("file_type_to_master", default_mapping),
            type_rules=rules,
        ),
        upsert=UpsertConfig(
            id_column_by_type=upsert_cfg.get("id_column_by_type", default_id_map),
        ),
    )


def config_to_dict(config: DownloadConfig) -> Dict:
    d = asdict(config)
    d["data_dir"] = str(config.data_dir)
    d["gmail"]["credentials_path"] = str(config.gmail.credentials_path)
    d["gmail"]["token_path"] = str(config.gmail.token_path)
    if config.prepare:
        d["prepare"]["tables_dir"] = str(config.prepare.tables_dir)
    if config.upsert:
        d["upsert"] = d.get("upsert", {})
        # ensure id map exists
        if "id_column_by_type" not in d["upsert"]:
            d["upsert"]["id_column_by_type"] = config.upsert.id_column_by_type
        # include event rules
        d["upsert"]["event_rules"] = config.upsert.event_rules
    return d


def save_config(config: DownloadConfig, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, indent=2)


def new_company_config_template(company: str) -> Dict:
    return {
        "company": company,
        "data_dir": f"data/{company}/downloads",
        "allowed_extensions": [".xlsx", ".xls", ".csv"],
        "mark_as_read": True,
        "gmail": {
            "scopes": ["https://www.googleapis.com/auth/gmail.modify"],
            "credentials_path": f"secrets/{company}/google/credentials.json",
            "token_path": f"secrets/{company}/google/gmail_token.json",
            "link_prefixes": ["https://go.servicetitan.com/PublicResource/File/"],
            "query_by_type": {}
        },
        "permit": {
            "austin_base_url": "https://data.austintexas.gov/resource/3syk-w9eu.csv",
            "default_lookback_hours": 168
        },
        "schedules": []
    }


# Global configuration for non-company-specific downloads/schedules
@dataclass
class GlobalConfig:
    permits_data_dir: Path = Path("global_data/permits")
    permit: PermitConfig = field(default_factory=PermitConfig)
    schedules: List[ScheduleConfig] = field(default_factory=list)


def load_global_config(config_path: Path = Path("config/global.json")) -> GlobalConfig:
    if not config_path.exists():
        return GlobalConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    schedules_cfg: List[ScheduleConfig] = []
    for s in cfg.get("schedules", []):
        schedules_cfg.append(ScheduleConfig(
            id=s.get("id", "default"),
            enabled=bool(s.get("enabled", False)),
            job=s.get("job", "permit_austin"),
            interval_minutes=int(s.get("interval_minutes", 60)),
            params=s.get("params", {}),
        ))
    return GlobalConfig(
        permits_data_dir=_norm_path(cfg.get("permits_data_dir", "global_data/permits")),
        permit=PermitConfig(
            austin_base_url=cfg.get("permit", {}).get("austin_base_url", "https://data.austintexas.gov/resource/3syk-w9eu.csv"),
            default_lookback_hours=int(cfg.get("permit", {}).get("default_lookback_hours", 168)),
        ),
        schedules=schedules_cfg,
    )


def save_global_config(global_config: GlobalConfig, config_path: Path = Path("config/global.json")) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(global_config)
    d["permits_data_dir"] = str(global_config.permits_data_dir)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


