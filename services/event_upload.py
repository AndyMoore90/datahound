#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from argparse import ArgumentParser
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from zoneinfo import ZoneInfo

import gspread
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gspread.exceptions import APIError, WorksheetNotFound
from gspread.utils import rowcol_to_a1
from oauth2client.service_account import ServiceAccountCredentials


BATCH_APPEND_SIZE = 100
RETRY_ATTEMPTS = 5
BACKOFF_INITIAL_SECONDS = 5
QUOTA_RETRY_DELAY_SECONDS = 60
UPDATE_THROTTLE_SECONDS = 60
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_COMPANY = "McCullough Heating and Air"
CACHE_DIR = PROJECT_ROOT / "data" / _COMPANY / "recent_events" / "cache"
REGISTRY_PATH = CACHE_DIR / "recent_event_registry.json"
PACIFIC_ZONE = ZoneInfo("America/Los_Angeles")
WINDOW_START = dt_time(8, 0)
WINDOW_END = dt_time(17, 0)

CANONICAL_COLUMN_ORDER: Dict[str, List[str]] = {
  "recent_cancellations": [
    "event_type",
    "entity_type",
    "entity_id",
    "detected_at",
    "status",
    "completion_date",
    "location_id",
    "cancellation_age_months",
    "Job ID",
    "Job Class",
    "Customer ID",
    "Summary",
    "Customer Phone",
    "profile_id",
    "customer_id",
    "location_ids",
    "location_count",
    "customer_addresses",
    "normalized_addresses",
    "duplicate_addresses",
    "address_count",
    "job_ids",
    "job_count",
    "estimate_ids",
    "estimate_count",
    "invoice_ids",
    "invoice_count",
    "call_ids",
    "call_count",
    "membership_ids",
    "membership_count",
    "rfm_recency",
    "rfm_frequency",
    "rfm_monetary",
    "rfm_score",
    "rfm_segment",
    "household_income",
    "property_value",
    "permit_matches",
    "permit_count",
    "competitor_permits",
    "competitor_permit_count",
    "is_marketable",
    "do_not_call",
    "do_not_service",
    "customer_tier",
    "customer_segment",
    "created_at",
    "updated_at",
    "enriched_at",
    "extraction_timestamp",
    "extraction_config",
    "source_event_type",
  ],
  "recent_unsold_estimates": [
    "event_type",
    "entity_type",
    "entity_id",
    "detected_at",
    "estimate_status",
    "creation_date",
    "opportunity_status",
    "location_id",
    "Estimate ID",
    "Customer ID",
    "Summary",
    "Customer Phone",
    "profile_id",
    "customer_id",
    "location_ids",
    "location_count",
    "customer_addresses",
    "normalized_addresses",
    "duplicate_addresses",
    "address_count",
    "job_ids",
    "job_count",
    "estimate_ids",
    "estimate_count",
    "invoice_ids",
    "invoice_count",
    "call_ids",
    "call_count",
    "membership_ids",
    "membership_count",
    "rfm_recency",
    "rfm_frequency",
    "rfm_monetary",
    "rfm_score",
    "rfm_segment",
    "household_income",
    "property_value",
    "permit_matches",
    "permit_count",
    "competitor_permits",
    "competitor_permit_count",
    "is_marketable",
    "do_not_call",
    "do_not_service",
    "customer_tier",
    "customer_segment",
    "created_at",
    "updated_at",
    "enriched_at",
    "extraction_timestamp",
    "extraction_config",
    "source_event_type",
  ],
  "recent_lost_customers": [
    "event_type",
    "entity_type",
    "entity_id",
    "detected_at",
    "Customer ID",
    "location_id",
    "Customer Phone",
    "profile_id",
    "customer_id",
    "location_ids",
    "location_count",
    "customer_addresses",
    "normalized_addresses",
    "duplicate_addresses",
    "address_count",
    "job_ids",
    "job_count",
    "estimate_ids",
    "estimate_count",
    "invoice_ids",
    "invoice_count",
    "call_ids",
    "call_count",
    "membership_ids",
    "membership_count",
    "rfm_recency",
    "rfm_frequency",
    "rfm_monetary",
    "rfm_score",
    "rfm_segment",
    "household_income",
    "property_value",
    "permit_matches",
    "permit_count",
    "competitor_permits",
    "competitor_permit_count",
    "is_marketable",
    "do_not_call",
    "do_not_service",
    "customer_tier",
    "customer_segment",
    "created_at",
    "updated_at",
    "enriched_at",
    "extraction_timestamp",
    "extraction_config",
    "source_event_type",
  ],
  "recent_overdue_maintenance": [
    "event_type",
    "entity_type",
    "entity_id",
    "detected_at",
    "Customer ID",
    "location_id",
    "Customer Phone",
    "profile_id",
    "customer_id",
    "location_ids",
    "location_count",
    "customer_addresses",
    "normalized_addresses",
    "duplicate_addresses",
    "address_count",
    "job_ids",
    "job_count",
    "estimate_ids",
    "estimate_count",
    "invoice_ids",
    "invoice_count",
    "call_ids",
    "call_count",
    "membership_ids",
    "membership_count",
    "rfm_recency",
    "rfm_frequency",
    "rfm_monetary",
    "rfm_score",
    "rfm_segment",
    "household_income",
    "property_value",
    "permit_matches",
    "permit_count",
    "competitor_permits",
    "competitor_permit_count",
    "is_marketable",
    "do_not_call",
    "do_not_service",
    "customer_tier",
    "customer_segment",
    "created_at",
    "updated_at",
    "enriched_at",
    "extraction_timestamp",
    "extraction_config",
    "source_event_type",
  ],
}


def normalize_column_order(event_name: str, frame: pd.DataFrame) -> pd.DataFrame:
  canonical_order = CANONICAL_COLUMN_ORDER.get(event_name)
  if not canonical_order:
    return frame
  
  existing_columns = list(frame.columns)
  ordered_columns: List[str] = []
  other_columns: List[str] = []
  
  for col in canonical_order:
    if col in existing_columns:
      ordered_columns.append(col)
  
  for col in existing_columns:
    if col not in ordered_columns:
      other_columns.append(col)
  
  final_order = ordered_columns + sorted(other_columns)
  return frame[final_order]


def ensure_cache_dir() -> None:
  CACHE_DIR.mkdir(parents=True, exist_ok=True)


def event_cache_path(event_name: str) -> Path:
  return CACHE_DIR / f"{event_name}.parquet"


def load_cached_frame(event_name: str) -> Optional[pd.DataFrame]:
  path = event_cache_path(event_name)
  if not path.exists():
    return None
  try:
    return pd.read_parquet(path)
  except Exception as error:
    print(f"Failed to read cached parquet for {event_name}: {error}. Removing cache.")
    try:
      path.unlink(missing_ok=True)
    except Exception:
      pass
    return None


def save_cached_frame(event_name: str, frame: pd.DataFrame) -> None:
  ensure_cache_dir()
  path = event_cache_path(event_name)
  frame.to_parquet(path, index=False)


def clear_cached_frame(event_name: str) -> None:
  path = event_cache_path(event_name)
  try:
    path.unlink(missing_ok=True)
  except Exception:
    pass


EVENT_NAMES: Sequence[str] = (
  "recent_cancellations",
  "recent_lost_customers",
  "recent_unsold_estimates",
  "recent_overdue_maintenance",
  "recent_second_chance_leads",
)


SHEET_ENV_KEYS: Dict[str, Dict[str, List[str]]] = {
  "recent_cancellations": {
    "name": ["RECENT_CANCELLATIONS_SHEET_NAME", "CANCELLATIONS_SHEET", "CANCELLATIONS_SHEET_NAME"],
    "id": ["RECENT_CANCELLATIONS_SHEET_ID", "CANCELLATIONS_SHEET_ID"],
  },
  "recent_lost_customers": {
    "name": ["RECENT_LOST_CUSTOMERS_SHEET_NAME", "LOST_CUSTOMERS_SHEET", "LOST_CUSTOMERS_SHEET_NAME"],
    "id": ["RECENT_LOST_CUSTOMERS_SHEET_ID", "LOST_CUSTOMERS_SHEET_ID"],
  },
  "recent_unsold_estimates": {
    "name": ["RECENT_UNSOLD_ESTIMATES_SHEET_NAME", "UNSOLD_ESTIMATES_SHEET", "UNSOLD_ESTIMATES_SHEET_NAME"],
    "id": ["RECENT_UNSOLD_ESTIMATES_SHEET_ID", "UNSOLD_ESTIMATES_SHEET_ID"],
  },
  "recent_overdue_maintenance": {
    "name": ["RECENT_OVERDUE_MAINTENANCE_SHEET_NAME", "OVERDUE_MAINTENANCE_SHEET", "OVERDUE_MAINTENANCE_SHEET_NAME"],
    "id": ["RECENT_OVERDUE_MAINTENANCE_SHEET_ID", "OVERDUE_MAINTENANCE_SHEET_ID"],
  },
  "recent_second_chance_leads": {
    "name": ["RECENT_SECOND_CHANCE_LEADS_SHEET_NAME", "SECOND_CHANCE_SHEET", "SECOND_CHANCE_SHEET_NAME"],
    "id": ["RECENT_SECOND_CHANCE_LEADS_SHEET_ID", "SECOND_CHANCE_SHEET_ID"],
  },
}


PRIORITY_IDENTIFIER_FIELDS: Dict[str, Tuple[str, ...]] = {
  "recent_second_chance_leads": ("Event Type", "Customer Phone"),
  "recent_cancellations": ("Event Type", "entity_id", "Customer ID", "customer_id"),
  "recent_lost_customers": ("Event Type", "entity_id", "Customer ID", "customer_id"),
  "recent_unsold_estimates": ("Event Type", "entity_id", "Customer ID", "customer_id"),
  "recent_overdue_maintenance": ("Event Type", "entity_id", "Customer ID", "customer_id"),
}

# Columns we treat as identifier-like for formatting and matching
ID_COLUMNS: Tuple[str, ...] = (
  "Customer ID",
  "customer_id",
  "entity_id",
  "Job ID",
  "Estimate ID",
  "Location ID",
  "Customer Phone",
  "customer_phone",
)

VOLATILE_COLUMNS: Tuple[str, ...] = (
  "extraction_timestamp",
  "Analysis Timestamp",
)


def normalize_header_name(name: str) -> str:
  return "".join(character for character in name.lower() if character.isalnum())


VOLATILE_NORMALIZED = {normalize_header_name(column) for column in VOLATILE_COLUMNS}


def utc_now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


def sleep_latch() -> None:
  if UPDATE_THROTTLE_SECONDS > 0:
    sleep(UPDATE_THROTTLE_SECONDS)


def seconds_until_window(now: datetime) -> Optional[float]:
  local_now = now.astimezone(PACIFIC_ZONE)
  if WINDOW_START <= local_now.time() < WINDOW_END:
    return None
  start_today = local_now.replace(
    hour=WINDOW_START.hour,
    minute=WINDOW_START.minute,
    second=0,
    microsecond=0,
  )
  if local_now < start_today:
    return (start_today - local_now).total_seconds()
  next_start = start_today + timedelta(days=1)
  return (next_start - local_now).total_seconds()


def normalize_sheet_value(column: str, value: Any) -> str:
  if value is None:
    return ""
  text = str(value).strip()
  if not text:
    return ""
  normalized_column = normalize_header_name(column)
  if normalized_column in ID_COLUMNS:
    if text.startswith("'"):
      text = text[1:]
    return text.replace(" ", "")
  upper = text.upper()
  if upper in {"TRUE", "FALSE"}:
    return upper
  try:
    numeric = float(text)
    if numeric.is_integer():
      return str(int(numeric))
    return str(numeric)
  except ValueError:
    pass
  return text


INVALID_EVENTS_SHEET_NAME = "Invalid Events"

EVENT_IDENTIFIER_FIELDS: Dict[str, Tuple[str, ...]] = {
  "recent_second_chance_leads": ("Event Type", "Customer Phone"),
}
DEFAULT_IDENTIFIER_FIELDS: Tuple[str, ...] = (
  "event_type",
  "entity_id",
  "customer_id",
  "Customer ID",
)


def chunked(values: Sequence[Any], size: int) -> Iterable[List[Any]]:
  if size <= 0:
    size = BATCH_APPEND_SIZE
  for start in range(0, len(values), size):
    yield list(values[start : start + size])


def call_with_retry(operation: Any) -> Any:
  delay = BACKOFF_INITIAL_SECONDS
  for attempt in range(1, RETRY_ATTEMPTS + 1):
    try:
      return operation()
    except APIError as error:
      response = getattr(error, "response", None)
      status_code = getattr(response, "status_code", None)
      error_str = str(error).lower()
      transient_statuses = {429, 500, 502, 503, 504}
      transient_tokens = (
        "quota exceeded",
        "rate limit",
        "service is currently unavailable",
      )
      if status_code in transient_statuses or any(token in error_str for token in transient_tokens):
        if attempt == RETRY_ATTEMPTS:
          raise
        sleep(delay)
        delay = min(delay * 2, QUOTA_RETRY_DELAY_SECONDS)
        continue
      if attempt == RETRY_ATTEMPTS:
        raise
      sleep(delay)
      delay *= 2
  raise RuntimeError("Exceeded retry attempts")


def load_env(base_dir: Path) -> None:
  env_path = base_dir / ".env"
  if env_path.exists():
    load_dotenv(env_path)


def get_gspread_client(base_dir: Path) -> gspread.Client:
  load_env(base_dir)
  credentials_path = os.environ.get("GOOGLE_CREDS")
  if not credentials_path:
    raise ValueError("GOOGLE_CREDS must be set in environment")
  scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
  ]
  credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
  return gspread.authorize(credentials)


def get_sheet_config(event_key: str) -> Dict[str, str]:
  key_options = SHEET_ENV_KEYS.get(event_key)
  if not key_options:
    raise ValueError(f"Unknown event key: {event_key}")
  sheet_name: Optional[str] = None
  sheet_id: Optional[str] = None
  for candidate in key_options["name"]:
    sheet_name = os.environ.get(candidate)
    if sheet_name:
      break
  for candidate in key_options["id"]:
    sheet_id = os.environ.get(candidate)
    if sheet_id:
      break
  if not sheet_name or not sheet_id:
    missing = {
      "name": [key for key in key_options["name"] if not os.environ.get(key)],
      "id": [key for key in key_options["id"] if not os.environ.get(key)],
    }
    raise ValueError(
      "Missing environment variables for "
      + f"{event_key}: name keys {missing['name']} or id keys {missing['id']}"
    )
  return {"sheet_name": sheet_name, "sheet_id": sheet_id}


def ensure_headers(worksheet: gspread.Worksheet, headers: List[str]) -> None:
  if not headers:
    return

  def _resize_if_needed() -> None:
    current_rows = worksheet.row_count
    current_cols = worksheet.col_count
    target_rows = max(current_rows, 1)
    target_cols = max(current_cols, len(headers))
    if target_rows != current_rows or target_cols != current_cols:
      worksheet.resize(rows=target_rows, cols=target_cols)

  call_with_retry(_resize_if_needed)
  header_range = f"A1:{rowcol_to_a1(1, len(headers))}"
  call_with_retry(lambda: worksheet.update([headers], range_name=header_range))


def get_or_create_worksheet(
  spreadsheet: gspread.Spreadsheet,
  sheet_name: str,
  headers: List[str],
) -> gspread.Worksheet:
  try:
    worksheet = call_with_retry(lambda: spreadsheet.worksheet(sheet_name))
  except WorksheetNotFound:
    worksheet = call_with_retry(
      lambda: spreadsheet.add_worksheet(
        title=sheet_name,
        rows=max(1000, len(headers) + 1),
        cols=max(1, len(headers)),
      )
    )
  ensure_headers(worksheet, headers)
  return worksheet


def format_literal(value: str) -> str:
  text = value.strip()
  if not text:
    return text
  if text.replace(" ", "").isdigit():
    return f"'{text}"
  return text


def format_export_value(column: str, value: Any) -> str:
  if value is None:
    return ""
  if isinstance(value, (pd.Timestamp, datetime)):
    return value.isoformat()
  if isinstance(value, float):
    if pd.isna(value):
      return ""
    if value.is_integer():
      return str(int(value))
    return str(value)
  if isinstance(value, (int, np.integer)):
    return str(value)
  if isinstance(value, (bool, np.bool_)):
    return "TRUE" if value else "FALSE"
  if isinstance(value, (list, tuple, set)):
    if not value:
      return ""
    return ", ".join(format_export_value(column, item) for item in value)
  if isinstance(value, np.ndarray):
    if value.size == 0:
      return ""
    return ", ".join(format_export_value(column, item) for item in value.tolist())
  if isinstance(value, pd.Series):
    return ", ".join(format_export_value(column, item) for item in value.tolist())
  text = str(value).strip()
  if text.lower() == "nan":
    return ""
  if column in ID_COLUMNS:
    return format_literal(text)
  if normalize_header_name(column) in ID_COLUMNS:
    return format_literal(text)
  return text


def prepare_cell_value(column: str, value: Any) -> Tuple[Any, str]:
  normalized_column = normalize_header_name(column)
  if value is None:
    return "", ""
  if isinstance(value, (float, np.floating)) and np.isnan(value):
    return "", ""
  if isinstance(value, pd.Timestamp):
    text = value.isoformat()
    return text, text
  if isinstance(value, (list, tuple, set)):
    text = ", ".join(str(item).strip() for item in value if str(item).strip())
    return text, text
  if isinstance(value, np.ndarray):
    return prepare_cell_value(column, list(value))
  if isinstance(value, pd.Series):
    return prepare_cell_value(column, value.tolist())
  if normalized_column in ID_COLUMNS:
    text = format_literal(str(value).strip())
    return text, text
  if isinstance(value, (bool, np.bool_)):
    text = "TRUE" if bool(value) else "FALSE"
    return bool(value), text
  if isinstance(value, (int, np.integer)):
    numeric = int(value)
    return numeric, str(numeric)
  if isinstance(value, (float, np.floating)):
    if value.is_integer():
      numeric = int(value)
      return numeric, str(numeric)
    numeric = float(value)
    return numeric, str(numeric)
  text = str(value).strip()
  if normalized_column in ID_COLUMNS:
    text = format_literal(text)
  return text, text


def prepare_rows(
  frame: pd.DataFrame,
  headers: Sequence[str],
) -> Tuple[List[List[Any]], List[Dict[str, str]]]:
  if frame.empty:
    return [], []
  write_rows: List[List[Any]] = []
  compare_maps: List[Dict[str, str]] = []
  for _, row in frame.iterrows():
    write_row: List[Any] = []
    compare_map: Dict[str, str] = {}
    for column in headers:
      write_value, compare_value = prepare_cell_value(column, row.get(column, ""))
      write_row.append(write_value)
      compare_map[column] = compare_value
    write_rows.append(write_row)
    compare_maps.append(compare_map)
  return write_rows, compare_maps


def compute_row_hash(headers: Sequence[str], compare_map: Dict[str, str]) -> str:
  normalized: Dict[str, str] = {}
  for header in headers:
    normalized_header = normalize_header_name(header)
    if normalized_header in VOLATILE_NORMALIZED:
      continue
    value = compare_map.get(header, "")
    if not value:
      continue
    normalized[normalized_header] = compare_map.get(header, "")
  return json.dumps(normalized, sort_keys=True)


def sheet_row_to_map(headers: Sequence[str], row_values: Sequence[Any]) -> Dict[str, str]:
  normalized: Dict[str, str] = {}
  for index, header in enumerate(headers):
    value = row_values[index] if index < len(row_values) else ""
    normalized[header] = normalize_sheet_value(header, value)
  return normalized


def normalize_sheet_key(raw: Any) -> str:
  if raw is None:
    return ""
  text = str(raw).strip()
  if text.startswith("'"):
    text = text[1:]
  return text.replace(" ", "")


def clean_phone_number(value: Any) -> str:
  if value is None:
    return ""
  if isinstance(value, (float, np.floating)) and np.isnan(value):
    return ""
  if isinstance(value, (list, tuple, set)):
    iterator = (str(item).strip() for item in value if str(item).strip())
    first_value = next(iterator, "")
  else:
    first_value = str(value).strip()
  if not first_value:
    return ""
  primary = first_value.split(",")[0]
  digits = re.findall(r"\d", primary)
  if not digits:
    return ""
  number = "".join(digits)
  if len(number) == 11 and number.startswith("1"):
    number = number[1:]
  if len(number) > 10:
    number = number[:10]
  return number


def extract_row_key(value: Any) -> str:
  if value is None:
    return ""
  if isinstance(value, float):
    if pd.isna(value):
      return ""
    if value.is_integer():
      value = int(value)
  if isinstance(value, (int, np.integer)):
    return str(value)
  text = str(value).strip()
  if text.startswith("'"):
    text = text[1:]
  return text.replace(" ", "")


def update_rows_in_batches(
  worksheet: gspread.Worksheet,
  headers: List[str],
  updates: List[Tuple[int, List[str]]],
  batch_size: int,
) -> None:
  if not updates:
    return
  _ = batch_size
  sorted_updates = sorted(updates, key=lambda item: item[0])
  data: List[Dict[str, Any]] = []
  group: List[List[Any]] = []
  start_row: Optional[int] = None
  for row_index, row_data in sorted_updates:
    if start_row is None:
      start_row = row_index
      group = [row_data]
      continue
    if row_index == start_row + len(group):
      group.append(row_data)
      continue
    end_row = start_row + len(group) - 1
    range_name = f"{rowcol_to_a1(start_row, 1)}:{rowcol_to_a1(end_row, len(headers))}"
    data.append({"range": range_name, "values": group})
    start_row = row_index
    group = [row_data]
  if group:
    end_row = start_row + len(group) - 1
    range_name = f"{rowcol_to_a1(start_row, 1)}:{rowcol_to_a1(end_row, len(headers))}"
    data.append({"range": range_name, "values": group})
  if data:
    call_with_retry(lambda: worksheet.batch_update(data, value_input_option="RAW"))


def append_rows_in_batches(
  worksheet: gspread.Worksheet,
  rows: List[List[Any]],
  batch_size: int,
) -> None:
  if not rows:
    return
  for chunk in chunked(rows, batch_size):
    call_with_retry(
      lambda chunk=chunk: worksheet.append_rows(
        chunk,
        value_input_option="RAW",
      )
    )


def load_registry(registry_path: Path) -> Dict[str, Dict[str, str]]:
  if not registry_path.exists():
    return {}
  try:
    raw_data = json.loads(registry_path.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    backup = registry_path.with_suffix(".invalid.json")
    registry_path.rename(backup)
    return {}
  if not isinstance(raw_data, dict):
    return {}
  sanitized: Dict[str, Dict[str, str]] = {}
  for event_key, entries in raw_data.items():
    if isinstance(entries, dict):
      event_entries: Dict[str, str] = {}
      for identifier, hash_value in entries.items():
        if isinstance(identifier, str) and isinstance(hash_value, str):
          event_entries[identifier] = hash_value
      sanitized[event_key] = event_entries
    else:
      sanitized[event_key] = {}
  return sanitized


def save_registry(registry_path: Path, registry: Dict[str, Dict[str, str]]) -> None:
  registry_path.parent.mkdir(parents=True, exist_ok=True)
  registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def append_invalid_rows(
  spreadsheet: gspread.Spreadsheet,
  rows: List[List[str]],
  headers: List[str],
  batch_size: int,
) -> None:
  if not rows:
    return
  worksheet = get_or_create_worksheet(spreadsheet, INVALID_EVENTS_SHEET_NAME, headers)
  ensure_headers(worksheet, headers)
  append_rows_in_batches(worksheet, rows, batch_size)


def format_literal(value: str) -> str:
  text = value.strip()
  if not text:
    return text
  if text.replace(" ", "").isdigit():
    return f"'{text}"
  return text


def event_identifier_fields(event_name: str, headers: Sequence[str]) -> Tuple[str, ...]:
  normalized_headers = {normalize_header_name(header): header for header in headers}
  priority_fields = PRIORITY_IDENTIFIER_FIELDS.get(event_name, ())
  collected: List[str] = []
  for field in priority_fields:
    normalized = normalize_header_name(field)
    header = normalized_headers.get(normalized)
    if header and header not in collected:
      collected.append(header)
  if not collected:
    collected = list(headers[:3])
  return tuple(collected)


def build_identifier(event_name: str, row: Dict[str, Any], headers: Sequence[str]) -> str:
  fields = event_identifier_fields(event_name, headers)
  entries = {field: str(row.get(field, "")).strip() for field in fields}
  return json.dumps(entries, sort_keys=True)


def upload_event(
  client: gspread.Client,
  base_dir: Path,
  event_name: str,
  batch_size: int,
  registry: Dict[str, Dict[str, str]],
  clear_sheet: bool,
  full_logs: bool,
) -> None:
  parquet_path = base_dir / f"{event_name}.parquet"
  if not parquet_path.exists():
    print(f"Skipping {event_name}: {parquet_path.name} not found")
    return

  print(f"\nProcessing {event_name}")
  frame = pd.read_parquet(parquet_path)
  print(f"Loaded {len(frame)} rows from {parquet_path.name}")

  frame = normalize_column_order(event_name, frame)

  if event_name in {"recent_unsold_estimates", "recent_cancellations", "recent_overdue_maintenance"}:
    frame = frame.copy()
    for column in ("Customer Phone", "customer_phone"):
      if column in frame.columns:
        frame[column] = frame[column].apply(clean_phone_number)

  headers = [str(column) for column in frame.columns]
  cached_frame = None if clear_sheet else load_cached_frame(event_name)
  if cached_frame is not None and not cached_frame.empty:
    cached_frame = normalize_column_order(event_name, cached_frame)
  write_rows, compare_maps = prepare_rows(frame, headers)

  sheet_config = get_sheet_config(event_name)
  spreadsheet = client.open_by_key(sheet_config["sheet_id"])
  worksheet = get_or_create_worksheet(spreadsheet, sheet_config["sheet_name"], headers)
  ensure_headers(worksheet, headers)

  if clear_sheet:
    print(f"Clearing sheet '{sheet_config['sheet_name']}' per --clear option")
    call_with_retry(worksheet.clear)
    ensure_headers(worksheet, headers)
    clear_cached_frame(event_name)
    cached_frame = None

  existing_values = call_with_retry(worksheet.get_all_values)
  existing_rows = existing_values[1:] if existing_values else []
  existing_row_data: Dict[int, List[Any]] = {}
  existing_map: Dict[str, Tuple[int, Dict[str, str]]] = {}
  id_to_identifier: Dict[int, str] = {}
  for offset, row_values in enumerate(existing_rows, start=2):
    padded = list(row_values[: len(headers)])
    if len(padded) < len(headers):
      padded.extend(["" for _ in range(len(headers) - len(padded))])
    existing_row_data[offset] = padded
    compare_map = sheet_row_to_map(headers, padded)
    identifier = build_identifier(event_name, compare_map, headers)
    normalized_identifier = normalize_sheet_value("identifier", identifier)
    if normalized_identifier and normalized_identifier not in existing_map:
      existing_map[normalized_identifier] = (offset, compare_map)
      id_to_identifier[offset] = identifier

  updates: List[Tuple[int, List[Any]]] = []
  appends: List[List[Any]] = []
  archived_rows: List[List[Any]] = []

  registry_entries = registry.setdefault(event_name, {})
  current_hashes: Dict[str, str] = {}
  updated_keys: Dict[str, List[str]] = {}

  cached_compare_map: Dict[str, Dict[str, str]] = {}
  cached_rows_for_archival: Dict[str, List[Any]] = {}
  if cached_frame is not None and not cached_frame.empty:
    cached_write_rows, cached_compare_rows = prepare_rows(cached_frame, headers)
    for row_values, compare_map in zip(cached_write_rows, cached_compare_rows):
      identifier = build_identifier(event_name, compare_map, headers)
      cached_compare_map[identifier] = compare_map
      cached_rows_for_archival[identifier] = row_values

  for row_idx, row_values in enumerate(write_rows):
    compare_map = compare_maps[row_idx]
    identifier = build_identifier(event_name, compare_map, headers)
    normalized_identifier = normalize_sheet_value("identifier", identifier)
    target_row: Optional[int] = None
    previous_compare: Optional[Dict[str, str]] = None
    if normalized_identifier in existing_map:
      target_row, previous_compare = existing_map[normalized_identifier]
    if target_row is not None:
      new_hash = compute_row_hash(headers, compare_map)
      previous_hash = registry_entries.get(identifier)
      if previous_hash != new_hash:
        if previous_compare:
          changed_columns = [
            header
            for header in headers
            if compare_map.get(header, "") != previous_compare.get(header, "")
          ]
          updated_keys[identifier] = changed_columns or headers
        updates.append((target_row, row_values))
      current_hashes[identifier] = new_hash
    else:
      new_hash = compute_row_hash(headers, compare_map)
      if registry_entries.get(identifier) != new_hash:
        appends.append(row_values)
      current_hashes[identifier] = new_hash

  previous_keys = set(registry_entries.keys())
  missing_keys = previous_keys - set(current_hashes.keys())
  for missing_key in missing_keys:
    cached_compare = cached_compare_map.get(missing_key)
    cached_row = cached_rows_for_archival.get(missing_key)
    if cached_row is not None:
      archived_rows.append(cached_row)
    elif cached_compare:
      cached_write_rows, _ = prepare_rows(pd.DataFrame([cached_compare]), headers)
      if cached_write_rows:
        archived_rows.append(cached_write_rows[0])
    else:
      archived_rows.append([missing_key])
  registry[event_name] = current_hashes
  save_cached_frame(event_name, frame)

  print(
    f"Updating {len(updates)} rows and appending {len(appends)} rows to '{sheet_config['sheet_name']}'"
  )
  update_rows_in_batches(worksheet, headers, updates, batch_size)
  append_rows_in_batches(worksheet, appends, batch_size)
  if archived_rows:
    append_invalid_rows(spreadsheet, archived_rows, headers, batch_size)
  if updated_keys:
    for identifier, columns in updated_keys.items():
      if full_logs:
        print(f"  - Updated {event_name} identifier {identifier}: columns {columns}")
      else:
        pass
        # print(f"  - Updated columns {columns}")
  print(f"Completed upload for {event_name}")


def parse_args() -> ArgumentParser:
  parser = ArgumentParser()
  parser.add_argument(
    "--events",
    nargs="*",
    default=list(EVENT_NAMES),
    help="Specific recent event keys to upload. Defaults to all events.",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=BATCH_APPEND_SIZE,
    help="Number of rows to append per batch request.",
  )
  parser.add_argument(
    "--clear",
    action="store_true",
    help="Clear existing data in each sheet before uploading (for testing/reset).",
  )
  parser.add_argument(
    "--interval",
    type=int,
    default=0,
    help="Run continuously, re-executing every N minutes (0 = run once).",
  )
  parser.add_argument(
    "--max-runs",
    type=int,
    default=0,
    help="Optional cap on number of runs when using --interval (0 = infinite).",
  )
  parser.add_argument(
    "--full-logs",
    action="store_true",
    help="Print full identifier details for updated rows.",
  )
  parser.add_argument(
    "--bypass-schedule",
    action="store_true",
    help="Bypass the time window restriction (8:00 AM - 5:00 PM PT) for testing or special cases.",
  )
  return parser


def main() -> None:
  parser = parse_args()
  args = parser.parse_args()

  events = [event for event in args.events if event in EVENT_NAMES]
  if not events:
    print("No valid events specified. Nothing to do.")
    return

  base_dir = PROJECT_ROOT / "data" / _COMPANY / "recent_events"
  client = get_gspread_client(base_dir)
  ensure_cache_dir()

  interval_seconds = max(0, args.interval * 60)
  max_runs = max(0, args.max_runs)
  run_number = 0

  try:
    while True:
      if not args.bypass_schedule:
        wait_seconds = seconds_until_window(datetime.now(timezone.utc))
        if wait_seconds is not None:
          minutes = int(wait_seconds // 60)
          seconds = int(wait_seconds % 60)
          print(
            f"Outside allowed window ({WINDOW_START} - {WINDOW_END} PT). "
            + f"Sleeping {minutes}m {seconds}s before retry."
          )
          time.sleep(wait_seconds)
          continue

      run_number += 1
      registry = load_registry(REGISTRY_PATH)

      started = utc_now_iso()
      print("#" * 80)
      print(f"RECENT EVENT UPLOAD - Started at {started}")
      print(f"Events: {', '.join(events)}")

      for event_name in events:
        upload_event(
          client,
          base_dir,
          event_name,
          args.batch_size,
          registry,
          args.clear,
          args.full_logs,
        )

      save_registry(REGISTRY_PATH, registry)
      finished = utc_now_iso()
      print("#" * 80)
      print(f"Upload completed at {finished}")

      if interval_seconds <= 0:
        break
      if max_runs and run_number >= max_runs:
        break

      print(f"Sleeping {interval_seconds} seconds before next run...")
      time.sleep(interval_seconds)
      args.clear = False
  except KeyboardInterrupt:
    print("Loop interrupted by user.")


if __name__ == "__main__":
  main()


