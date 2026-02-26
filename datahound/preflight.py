"""Startup preflight checks for DataHound Pro."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize_path(path_str: str) -> Path:
    cleaned = path_str.replace("\\", "/")
    path = Path(cleaned).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_env_file(path: Path = PROJECT_ROOT / ".env") -> dict:
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"").strip("'")
    return values


def _find_deepseek_key() -> Tuple[Optional[str], Optional[str]]:
    env_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key, "environment"

    env_file = _load_env_file()
    file_key = env_file.get("DEEPSEEK_API_KEY") or env_file.get("OPENAI_API_KEY")
    if file_key:
        return file_key, ".env file"

    config_path = PROJECT_ROOT / "config" / "global.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            cfg_key = payload.get("deepseek_api_key")
            if cfg_key:
                return cfg_key, "config/global.json"
        except json.JSONDecodeError:
            pass

    return None, None


def check_required_env() -> List[CheckResult]:
    results: List[CheckResult] = []

    creds_env = os.getenv("GOOGLE_CREDS")
    if not creds_env:
        results.append(
            CheckResult(
                "Google credentials env",
                False,
                "Set GOOGLE_CREDS to the path of your Google service account JSON.",
            )
        )
    else:
        creds_path = _normalize_path(creds_env)
        if creds_path.exists():
            results.append(
                CheckResult(
                    "Google credentials env",
                    True,
                    f"Found Google credentials at {creds_path}.",
                )
            )
        else:
            results.append(
                CheckResult(
                    "Google credentials env",
                    False,
                    f"GOOGLE_CREDS points to {creds_path}, but the file is missing.",
                )
            )

    sheet_id = (
        os.getenv("SECOND_CHANCE_SHEET_ID")
        or os.getenv("RECENT_SECOND_CHANCE_LEADS_SHEET_ID")
    )
    if sheet_id:
        results.append(
            CheckResult(
                "Second chance sheet ID",
                True,
                "Found second chance sheet ID in environment.",
            )
        )
    else:
        results.append(
            CheckResult(
                "Second chance sheet ID",
                False,
                "Set SECOND_CHANCE_SHEET_ID (or RECENT_SECOND_CHANCE_LEADS_SHEET_ID) to the Google Sheet ID.",
            )
        )

    deepseek_key, source = _find_deepseek_key()
    if deepseek_key:
        results.append(
            CheckResult(
                "LLM API key",
                True,
                f"Found LLM API key via {source}.",
            )
        )
    else:
        results.append(
            CheckResult(
                "LLM API key",
                False,
                "Set DEEPSEEK_API_KEY (or OPENAI_API_KEY) to enable LLM analysis.",
            )
        )

    return results


def check_gmail_files() -> List[CheckResult]:
    results: List[CheckResult] = []
    companies_dir = PROJECT_ROOT / "companies"
    if not companies_dir.exists():
        return results

    for config_file in companies_dir.glob("*/config.json"):
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results.append(
                CheckResult(
                    f"Gmail config ({config_file.parent.name})",
                    False,
                    "Company config could not be read. Fix JSON formatting.",
                )
            )
            continue

        gmail_cfg = config.get("gmail", {})
        company = config.get("company", config_file.parent.name)
        credentials_path = gmail_cfg.get("credentials_path")
        token_path = gmail_cfg.get("token_path")

        if credentials_path:
            cred_path = _normalize_path(credentials_path)
            if cred_path.exists():
                results.append(
                    CheckResult(
                        f"Gmail credentials ({company})",
                        True,
                        f"Found Gmail credentials at {cred_path}.",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        f"Gmail credentials ({company})",
                        False,
                        f"Missing Gmail credentials. Place the OAuth JSON at {cred_path}.",
                    )
                )

        if token_path:
            tok_path = _normalize_path(token_path)
            if tok_path.exists():
                results.append(
                    CheckResult(
                        f"Gmail token ({company})",
                        True,
                        f"Found Gmail token at {tok_path}.",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        f"Gmail token ({company})",
                        False,
                        "Gmail token missing. Run the Gmail auth flow in the Data Pipeline page to generate it.",
                    )
                )

    return results


def check_control_plane_config() -> List[CheckResult]:
    results: List[CheckResult] = []
    source = (os.getenv("DATAHOUND_CONTROL_PLANE_SOURCE") or "auto").strip().lower()
    if source not in {"auto", "db", "json"}:
        results.append(
            CheckResult(
                "Control-plane source",
                False,
                "DATAHOUND_CONTROL_PLANE_SOURCE must be one of: auto, db, json.",
            )
        )
        return results

    storage_url = os.getenv("DATAHOUND_STORAGE_URL") or os.getenv("DATABASE_URL")
    if source == "db" and not storage_url:
        results.append(
            CheckResult(
                "Control-plane DB configuration",
                False,
                "DATAHOUND_CONTROL_PLANE_SOURCE=db requires DATAHOUND_STORAGE_URL (or DATABASE_URL).",
            )
        )
    elif source == "json" and storage_url:
        results.append(
            CheckResult(
                "Control-plane DB configuration",
                True,
                "Control-plane source forced to json; DB URL is present but will be ignored.",
            )
        )
    elif storage_url:
        results.append(
            CheckResult(
                "Control-plane DB configuration",
                True,
                "DB URL detected; control-plane can run in db mode.",
            )
        )
    else:
        results.append(
            CheckResult(
                "Control-plane DB configuration",
                True,
                "No DB URL detected; control-plane will use JSON fallback.",
            )
        )

    return results


def check_connectivity() -> List[CheckResult]:
    results: List[CheckResult] = []

    deepseek_key, _ = _find_deepseek_key()
    if deepseek_key:
        try:
            resp = requests.get(
                "https://api.deepseek.com/v1/models",
                headers={"Authorization": f"Bearer {deepseek_key}"},
                timeout=5,
            )
            if resp.status_code == 200:
                results.append(
                    CheckResult(
                        "DeepSeek connectivity",
                        True,
                        "Reached DeepSeek API successfully.",
                    )
                )
            elif resp.status_code in {401, 403}:
                results.append(
                    CheckResult(
                        "DeepSeek connectivity",
                        False,
                        "DeepSeek API key was rejected. Double-check the key value.",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        "DeepSeek connectivity",
                        False,
                        f"DeepSeek API returned status {resp.status_code}. Try again later.",
                    )
                )
        except requests.RequestException:
            results.append(
                CheckResult(
                    "DeepSeek connectivity",
                    False,
                    "Could not reach DeepSeek API. Check internet access or firewall rules.",
                )
            )
    else:
        results.append(
            CheckResult(
                "DeepSeek connectivity",
                False,
                "Skipped because no LLM API key is configured.",
            )
        )

    try:
        resp = requests.get("https://www.googleapis.com/discovery/v1/apis", timeout=5)
        if resp.status_code == 200:
            results.append(
                CheckResult(
                    "Google API connectivity",
                    True,
                    "Reached Google APIs endpoint successfully.",
                )
            )
        else:
            results.append(
                CheckResult(
                    "Google API connectivity",
                    False,
                    f"Google APIs returned status {resp.status_code}.",
                )
            )
    except requests.RequestException:
        results.append(
            CheckResult(
                "Google API connectivity",
                False,
                "Could not reach Google APIs. Check internet access or firewall rules.",
            )
        )

    return results


def run_preflight() -> int:
    checks = []
    checks.extend(check_required_env())
    checks.extend(check_gmail_files())
    checks.extend(check_control_plane_config())
    checks.extend(check_connectivity())

    print("DataHound Pro preflight checks\n")
    failures = 0
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"[{status}] {check.name} - {check.message}")
        if not check.ok:
            failures += 1

    if failures:
        print("\nPreflight failed. Fix the items marked FAIL and rerun this command.")
        return 1

    print("\nPreflight passed. You're ready to start the app.")
    return 0


def main() -> None:
    sys.exit(run_preflight())


if __name__ == "__main__":
    main()
