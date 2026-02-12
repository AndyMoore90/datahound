#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import time
import unicodedata
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_DIR.parent
DEFAULT_SECRETS_DIR = PROJECT_ROOT / "secrets" / "McCullough Heating and Air" / "google"
DEFAULT_CREDENTIALS = DEFAULT_SECRETS_DIR / "credentials.json"
DEFAULT_TOKEN = DEFAULT_SECRETS_DIR / "sheets_token.json"
DEFAULT_ARCHIVE_DIR = Path("Z:/datahound_backups/sms_csv_archive")
EMOJI_RANGES = (
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FAFF),
    (0x2600, 0x27BF),
)
EMOJI_EXTRA_CODES = {
    0x2764,
    0x2B50,
    0x23F3,
    0x231A,
}
EMOJI_MODIFIER_RANGE = (0x1F3FB, 0x1F3FF)
EMOJI_IGNORED_CODES = {
    0xFE0F,
    0xFE0E,
    0x200D,
    0x200B,
}
SPACE_LIKE_CODES = {
    0x2009,
    0x200A,
    0x2006,
    0x2005,
    0x2004,
    0x2003,
    0x2002,
    0x2001,
    0x2000,
}
PUNCTUATION_REPLACEMENTS = {
    0x2018: "'",
    0x2019: "'",
    0x201A: "'",
    0x201C: '"',
    0x201D: '"',
    0x201E: '"',
    0x2013: "-",
    0x2014: "-",
    0x2026: "...",
}


def is_emoji_codepoint(codepoint: int) -> bool:
    if any(start <= codepoint <= end for start, end in EMOJI_RANGES):
        return True
    if codepoint in EMOJI_EXTRA_CODES:
        return True
    return False


def is_modifier_codepoint(codepoint: int) -> bool:
    start, end = EMOJI_MODIFIER_RANGE
    return start <= codepoint <= end


def load_credentials(credentials_path: Path, token_path: Path) -> Credentials:
    print(f"Using credentials at {credentials_path}")
    print(f"Using token at {token_path}")
    creds: Optional[Credentials] = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as exc:
            print(f"Failed to load token: {exc}")
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                print("Refreshed existing token")
            except Exception as exc:
                print(f"Failed to refresh token: {exc}")
                creds = None
        if not creds or not creds.valid:
            if not credentials_path.exists():
                available = list(credentials_path.parent.glob("*"))
                print(f"Credentials directory contents: {available}")
                raise FileNotFoundError(f"Missing credentials file at {credentials_path}")
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        print(f"Saved token to {token_path}")
    return creds


def download_csv(sheet_id: str, creds: Credentials, output_path: Path, gid: Optional[str]) -> Path:
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError("Credentials are invalid and cannot be refreshed")
    params = {"format": "csv"}
    if gid:
        params["gid"] = gid
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export"
    headers = {"Authorization": f"Bearer {creds.token}"}
    response = requests.get(url, headers=headers, params=params, timeout=60)
    if response.status_code == 403:
        raise PermissionError("Access denied. Ensure the authenticated account can view the sheet.")
    if response.status_code == 404:
        raise FileNotFoundError("Sheet not found. Confirm the sheet ID and permissions.")
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path


def resolve_output_path(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = PROJECT_ROOT / "data" / "McCullough Heating and Air" / "sms_exports"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"google_sheet_{timestamp}.csv"


def transform_emojis(text: str) -> str:
    result: list[str] = []
    for char in text:
        codepoint = ord(char)
        if codepoint in EMOJI_IGNORED_CODES:
            continue
        if codepoint in SPACE_LIKE_CODES:
            result.append(" ")
            continue
        if is_emoji_codepoint(codepoint) or is_modifier_codepoint(codepoint):
            try:
                name = unicodedata.name(char)
            except ValueError:
                name = f"emoji_{codepoint:04x}"
            token = name.lower().replace(" ", "_").replace("-", "_")
            result.append(f":{token}:")
            continue
        if codepoint in PUNCTUATION_REPLACEMENTS:
            result.append(PUNCTUATION_REPLACEMENTS[codepoint])
            continue
        result.append(char)
    return "".join(result)


def clean_csv_file(path: Path) -> None:
    print(f"Cleaning {path}")
    with path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    if not rows:
        print("Downloaded file is empty")
        return
    trimmed_rows = rows[1:] if len(rows) > 1 else []
    if not trimmed_rows:
        print("No header row after removing first line")
        path.write_text("", encoding="utf-8")
        return
    header = trimmed_rows[0]
    data_rows = trimmed_rows[1:]
    if not header:
        print("Header row is empty")
        path.write_text("", encoding="utf-8")
        return
    cleaned_records: list[dict[str, str]] = []
    for row in data_rows:
        record = {header[i]: row[i] if i < len(row) else "" for i in range(len(header))}
        sms_raw = record.get("SMSText", "")
        sms_processed = transform_emojis(sms_raw)
        if sms_processed.strip():
            record["SMSText"] = sms_processed
            cleaned_records.append(record)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=header)
        writer.writeheader()
        writer.writerows(cleaned_records)
    print(f"Kept {len(cleaned_records)} rows in {path}")


def archive_file(path: Path, archive_dir: Path) -> None:
    if not path.exists():
        print(f"File already moved or missing: {path}")
        return
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / path.name
    counter = 1
    while target.exists():
        target = archive_dir / f"{path.stem}_{counter}{path.suffix}"
        counter += 1
    shutil.move(str(path), str(target))
    print(f"Archived {path} to {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet-id", default="1WJpjFg4vrnBN-Gu9Fl9v0aLyN7owagozQXj1io4j6lc")
    parser.add_argument("--gid")
    parser.add_argument("--credentials", default=str(DEFAULT_CREDENTIALS))
    parser.add_argument("--token", default=str(DEFAULT_TOKEN))
    parser.add_argument("--output")
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE_DIR))
    parser.add_argument("--interval-minutes", type=int, default=30)
    parser.add_argument("--keep-files", type=int, default=6)
    parser.add_argument("--run-once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    credentials_path = Path(args.credentials).expanduser().resolve()
    token_path = Path(args.token).expanduser().resolve()
    archive_dir = Path(args.archive).expanduser().resolve()
    keep_files = max(args.keep_files, 0)
    interval_minutes = max(args.interval_minutes, 1)
    history: deque[Path] = deque()
    iteration = 0
    while True:
        iteration += 1
        started = datetime.now().isoformat()
        print(f"Iteration {iteration} started at {started}")
        try:
            creds = load_credentials(credentials_path, token_path)
            output_path = resolve_output_path(args.output)
            saved_path = download_csv(args.sheet_id, creds, output_path, args.gid)
            print(f"Saved CSV to {saved_path}")
            clean_csv_file(saved_path)
            history.append(saved_path)
            while keep_files and len(history) > keep_files:
                old_path = history.popleft()
                archive_file(old_path, archive_dir)
        except Exception as exc:
            print(f"Iteration {iteration} error: {exc}")
        if args.run_once:
            break
        print(f"Sleeping for {interval_minutes} minutes")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    main()
