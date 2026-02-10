import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests


def _append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _build_austin_query(start_iso: str, end_iso: str) -> str:
    base = (
        "SELECT\n"
        "  `permittype`,\n"
        "  `permit_type_desc`,\n"
        "  `permit_number`,\n"
        "  `permit_class_mapped`,\n"
        "  `permit_class`,\n"
        "  `work_class`,\n"
        "  `condominium`,\n"
        "  `permit_location`,\n"
        "  `description`,\n"
        "  `tcad_id`,\n"
        "  `legal_description`,\n"
        "  `applieddate`,\n"
        "  `issue_date`,\n"
        "  `day_issued`,\n"
        "  `calendar_year_issued`,\n"
        "  `fiscal_year_issued`,\n"
        "  `issued_in_last_30_days`,\n"
        "  `issue_method`,\n"
        "  `status_current`,\n"
        "  `statusdate`,\n"
        "  `expiresdate`,\n"
        "  `completed_date`,\n"
        "  `total_existing_bldg_sqft`,\n"
        "  `remodel_repair_sqft`,\n"
        "  `total_new_add_sqft`,\n"
        "  `total_valuation_remodel`,\n"
        "  `total_job_valuation`,\n"
        "  `number_of_floors`,\n"
        "  `housing_units`,\n"
        "  `building_valuation`,\n"
        "  `building_valuation_remodel`,\n"
        "  `electrical_valuation`,\n"
        "  `electrical_valuation_remodel`,\n"
        "  `mechanical_valuation`,\n"
        "  `mechanical_valuation_remodel`,\n"
        "  `plumbing_valuation`,\n"
        "  `plumbing_valuation_remodel`,\n"
        "  `medgas_valuation`,\n"
        "  `medgas_valuation_remodel`,\n"
        "  `original_address1`,\n"
        "  `original_city`,\n"
        "  `original_state`,\n"
        "  `original_zip`,\n"
        "  `council_district`,\n"
        "  `jurisdiction`,\n"
        "  `link`,\n"
        "  `project_id`,\n"
        "  `masterpermitnum`,\n"
        "  `latitude`,\n"
        "  `longitude`,\n"
        "  `location`,\n"
        "  `contractor_trade`,\n"
        "  `contractor_company_name`,\n"
        "  `contractor_full_name`,\n"
        "  `contractor_phone`,\n"
        "  `contractor_address1`,\n"
        "  `contractor_address2`,\n"
        "  `contractor_city`,\n"
        "  `contractor_zip`,\n"
        "  `applicant_full_name`,\n"
        "  `applicant_org`,\n"
        "  `applicant_phone`,\n"
        "  `applicant_address1`,\n"
        "  `applicant_address2`,\n"
        "  `applicant_city`,\n"
        "  `applicantzip`,\n"
        "  `certificate_of_occupancy`,\n"
        "  `total_lot_sq_ft`\n"
        "WHERE\n"
        "  `applieddate`\n"
        "    BETWEEN '" + start_iso + "' :: floating_timestamp\n"
        "    AND '" + end_iso + "' :: floating_timestamp"
    )
    return base


def download_austin_permits(company: str, data_dir: Path, base_url: str, start_dt: datetime, end_dt: datetime) -> Optional[str]:
    start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S")
    query = _build_austin_query(start_iso, end_iso)
    url = f"{base_url}?$query={quote(query)}"
    resp = requests.get(url, timeout=60)
    logs_dir = data_dir.parent / "logs"
    if resp.status_code != 200:
        _append_log(logs_dir / "global_permits_log.jsonl", {
            "ts": datetime.now(UTC).isoformat(),
            "company": company,
            "file_type": "Austin_Permits",
            "status": "error",
            "error": f"HTTP {resp.status_code}",
            "url": url,
        })
        return None
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    fname = f"permits_austin_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}_{ts}.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    outpath = data_dir / fname
    with open(outpath, "wb") as f:
        f.write(resp.content)
    _append_log(logs_dir / "global_permits_log.jsonl", {
        "ts": datetime.now(UTC).isoformat(),
        "company": company,
        "file_type": "Austin_Permits",
        "status": "downloaded_permits",
        "filename": fname,
        "start": start_iso,
        "end": end_iso,
        "url": url,
    })
    return fname


