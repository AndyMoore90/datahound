from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]


def _default_second_chance_dir(company: str = "McCullough Heating and Air") -> Path:
    return ROOT_DIR / "data" / company / "second_chance"


def parse_date_string(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid date value: {value}")


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if value is None or isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return None
    return None


def _parse_boolean(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "yes"}:
            return True
        if lowered in {"false", "f", "0", "no"}:
            return False
    return None


def _parse_call_ids(value: Any) -> Set[str]:
    if value is None or isinstance(value, float) and pd.isna(value):
        return set()
    data = value
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return set()
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, Iterable):
        return set()
    call_ids: Set[str] = set()
    for item in data:
        if isinstance(item, dict) and item.get("call_id") is not None:
            call_ids.add(str(item["call_id"]).strip())
    return call_ids


def _parse_primary_call_date(value: Any) -> Optional[date]:
    if value is None or isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
    return None


def _load_processed_customers(path: Path, start: Optional[date], end: Optional[date]) -> Dict[str, Any]:
    customers: Dict[str, Any] = {}
    total_transcripts = 0
    call_ids: Set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            last_processed = _parse_iso_datetime(record.get("last_processed"))
            record_date = last_processed.date() if last_processed else None
            if start and (not record_date or record_date < start):
                continue
            if end and (not record_date or record_date > end):
                continue
            processed_ids = [str(item).strip() for item in record.get("processed_call_ids", []) if item]
            total_transcripts += len(processed_ids)
            call_ids.update(processed_ids)
            record["processed_call_ids"] = processed_ids
            record["last_processed_dt"] = last_processed
            customers[str(record.get("customer_phone", "")).strip()] = record

    return {
        "records": customers,
        "transcripts": total_transcripts,
        "call_ids": call_ids,
    }


def _load_lead_records(
    path: Path,
    call_ids: Set[str],
    customer_phones: Set[str],
    start: Optional[date],
    end: Optional[date],
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if df.empty:
        return df

    # Backward-compatibility for older/partial lead schemas.
    required_columns = [
        "customer_phone",
        "referenced_transcripts",
        "primary_call_date",
        "was_customer_call",
        "was_service_request",
        "was_booked",
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    df["customer_phone"] = df["customer_phone"].astype(str).str.strip()
    df = df[df["customer_phone"].isin(customer_phones)]
    if df.empty:
        return df

    df["call_ids"] = df["referenced_transcripts"].apply(_parse_call_ids)
    df = df[df["call_ids"].map(lambda ids: bool(ids.intersection(call_ids)))]
    if df.empty:
        return df

    df["primary_call_date_parsed"] = df["primary_call_date"].apply(_parse_primary_call_date)
    if start or end:
        df = df[df["primary_call_date_parsed"].notna()]
        if start:
            df = df[df["primary_call_date_parsed"].map(lambda value: value and value >= start)]
        if end:
            df = df[df["primary_call_date_parsed"].map(lambda value: value and value <= end)]
        if df.empty:
            return df

    df["was_customer_call"] = df["was_customer_call"].apply(_parse_boolean)
    df["was_service_request"] = df["was_service_request"].apply(_parse_boolean)
    df["was_booked"] = df["was_booked"].apply(_parse_boolean)

    return df


def _aggregate_boolean(series: pd.Series) -> Optional[bool]:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return bool(cleaned.any())


def _build_customer_classification(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["customer_phone", "was_customer_call", "was_service_request", "was_booked"])
    grouped = (
        df.groupby("customer_phone")[["was_customer_call", "was_service_request", "was_booked"]]
        .agg(_aggregate_boolean)
        .reset_index()
    )
    return grouped


def _summarize_classification(grouped: pd.DataFrame, column: str) -> Dict[str, int]:
    true_count = grouped[column].map(lambda value: value is True).sum()
    false_count = grouped[column].map(lambda value: value is False).sum()
    unknown_count = len(grouped) - true_count - false_count
    return {"true": int(true_count), "false": int(false_count), "unknown": int(unknown_count)}


def _ensure_classification_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["customer_phone", "was_customer_call", "was_service_request", "was_booked"]
    for col in required:
        if col not in df.columns:
            df[col] = None
    return df


def _build_funnel_rows(grouped: pd.DataFrame) -> Dict[str, int]:
    grouped = _ensure_classification_columns(grouped.copy())
    customer_true = grouped.loc[grouped["was_customer_call"].map(lambda value: value is True)]
    service_true = customer_true.loc[customer_true["was_service_request"].map(lambda value: value is True)]
    service_booked = service_true.loc[service_true["was_booked"].map(lambda value: value is True)]
    return {
        "customers": int(len(customer_true)),
        "service_requests": int(len(service_true)),
        "booked": int(len(service_booked)),
    }


def _build_timeline(grouped_daily: pd.DataFrame) -> Dict[str, Any]:
    if grouped_daily.empty:
        return {"daily": []}
    records = []
    for current_date, group in grouped_daily.groupby("primary_call_date_parsed"):
        day_counts = _summarize_classification(group, "was_customer_call")
        service_counts = _summarize_classification(group.loc[group["was_customer_call"].map(lambda value: value is True)], "was_service_request")
        booking_counts = _summarize_classification(group.loc[group["was_service_request"].map(lambda value: value is True)], "was_booked")
        not_booked_counts = {
            "true": max(service_counts.get("true", 0) - booking_counts.get("true", 0), 0),
            "false": booking_counts.get("true", 0),
            "unknown": service_counts.get("unknown", 0),
        }
        records.append(
            {
                "date": current_date.isoformat(),
                "customers": day_counts,
                "service_requests": service_counts,
                "bookings": booking_counts,
                "not_booked": not_booked_counts,
            }
        )
    return {"daily": records}


def compute_transcript_metrics(
    processed_file: Optional[Path] = None,
    leads_file: Optional[Path] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    company: str = "McCullough Heating and Air",
) -> Dict[str, Any]:
    data_dir = _default_second_chance_dir(company)
    if processed_file is None:
        processed_file = data_dir / "processed_customers.jsonl"
    if leads_file is None:
        leads_file = data_dir / "second_chance_leads.parquet"
    if not processed_file.exists():
        raise FileNotFoundError(f"processed JSONL not found: {processed_file}")
    if not leads_file.exists():
        raise FileNotFoundError(f"second chance leads parquet not found: {leads_file}")
    if start_date and end_date and start_date > end_date:
        raise ValueError("start date cannot be after end date")

    processed_payload = _load_processed_customers(processed_file, start_date, end_date)
    customers: Dict[str, Any] = processed_payload["records"]
    transcript_count: int = processed_payload["transcripts"]
    call_ids: Set[str] = processed_payload["call_ids"]

    leads_df = _load_lead_records(
        leads_file,
        call_ids,
        set(customers.keys()),
        start_date,
        end_date,
    )

    classification_df = _ensure_classification_columns(_build_customer_classification(leads_df))

    customers_summary = _summarize_classification(classification_df, "was_customer_call") if not classification_df.empty else {"true": 0, "false": 0, "unknown": 0}
    service_summary = {"true": 0, "false": 0, "unknown": 0}
    booking_summary = {"true": 0, "false": 0, "unknown": 0}

    if not classification_df.empty:
        customer_subset = classification_df.loc[classification_df["was_customer_call"].map(lambda value: value is True)]
        service_summary = _summarize_classification(customer_subset, "was_service_request")
        service_subset = customer_subset.loc[customer_subset["was_service_request"].map(lambda value: value is True)]
        booking_summary = _summarize_classification(service_subset, "was_booked")

    funnel = _build_funnel_rows(classification_df)

    daily_grouped = pd.DataFrame()
    if not leads_df.empty:
        daily_grouped = (
            leads_df.dropna(subset=["primary_call_date_parsed"])
            .groupby(["primary_call_date_parsed", "customer_phone"])[["was_customer_call", "was_service_request", "was_booked"]]
            .agg(_aggregate_boolean)
            .reset_index()
        )
    timeline = _build_timeline(daily_grouped) if not daily_grouped.empty else {"daily": []}

    result = {
        "filters": {
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
        },
        "totals": {
            "transcripts_processed": transcript_count,
            "unique_callers_processed": len(customers),
            "callers_with_lead_data": len(classification_df),
        },
        "customers": customers_summary,
        "service_requests": service_summary,
        "bookings": booking_summary,
        "funnel": funnel,
        "timeline": timeline,
        "customer_details": classification_df.to_dict(orient="records") if not classification_df.empty else [],
    }

    return result


def backfill_processed_customers(company: str = "McCullough Heating and Air") -> Dict[str, int]:
    data_dir = _default_second_chance_dir(company)
    leads_file = data_dir / "second_chance_leads.parquet"
    profiles_file = data_dir / "customer_call_profiles.json"
    processed_file = data_dir / "processed_customers.jsonl"

    if not leads_file.exists():
        return {"status": "skipped", "reason": "no leads file", "backfilled": 0}
    if not profiles_file.exists():
        return {"status": "skipped", "reason": "no profiles file", "backfilled": 0}

    persistent_df = pd.read_parquet(leads_file)
    if persistent_df.empty:
        return {"status": "skipped", "reason": "leads file empty", "backfilled": 0}

    with open(profiles_file, "r", encoding="utf-8") as fh:
        profiles_list = json.load(fh)

    customer_call_mapping: Dict[str, list] = {}
    for profile in profiles_list:
        phone = profile.get("customer_phone")
        call_ids = [c.get("call_id", "") for c in profile.get("calls", [])]
        if phone and call_ids:
            customer_call_mapping[phone] = call_ids

    existing_phones: set = set()
    if processed_file.exists():
        with open(processed_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    existing_phones.add(record.get("customer_phone", ""))
                except json.JSONDecodeError:
                    continue

    backfilled = 0
    with open(processed_file, "a", encoding="utf-8") as fh:
        for _, row in persistent_df.iterrows():
            phone = str(row.get("customer_phone", "")).strip()
            if not phone or phone in existing_phones:
                continue
            if phone not in customer_call_mapping:
                continue
            call_ids = customer_call_mapping[phone]
            entry = {
                "customer_phone": phone,
                "processed_call_ids": call_ids,
                "last_processed": str(row.get("analysis_timestamp", "")),
                "total_calls_processed": len(call_ids),
            }
            fh.write(json.dumps(entry) + "\n")
            existing_phones.add(phone)
            backfilled += 1

    return {"status": "ok", "backfilled": backfilled, "total_processed": len(existing_phones)}
