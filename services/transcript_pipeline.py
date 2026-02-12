#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Set

import pandas as pd
import subprocess
import sys
import traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from datahound.extract.engine import CustomExtractionEngine
from datahound.extract.types import ExtractionConfig
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from json_repair import repair_json

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_DIR = Path(__file__).resolve().parent
DEFAULT_SECRETS_DIR = PROJECT_ROOT / "secrets" / "McCullough Heating and Air" / "google"
DEFAULT_CREDENTIALS = DEFAULT_SECRETS_DIR / "credentials.json"
DEFAULT_TOKEN = DEFAULT_SECRETS_DIR / "mcc_transcript_token.json"
DEFAULT_ARCHIVE_DIR = Path("Z:/datahound_backups/mcc_transcript_archive")

# DeepSeek API Configuration
DEEPSEEK_API_KEY = "sk-7f665faefb854ce69aad1e049887942c"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"
TEMPERATURE = 0.1
MAX_TOKENS = 12000
REQUEST_DELAY_SECONDS = 1.0
CONCURRENT_API_CALLS = 50
MAX_PROFILES_TO_ANALYZE = None  # Set to a number (e.g., 5) to limit for testing, None for all profiles

# Processing tracking
PROCESSING_TRACKING_FILE = "processed_customers.jsonl"
FAILURE_LOG_FILE = "analysis_failures.jsonl"

# Output directory structure
SECOND_CHANCE_DATA_DIR = PROJECT_ROOT / "data" / "McCullough Heating and Air" / "second_chance"
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
    default_dir = SECOND_CHANCE_DATA_DIR
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"mcc_transcript_{timestamp}.csv"


def normalize_phone(phone):
    """Normalize phone numbers: remove non-digits, strip leading '1'"""
    if pd.isna(phone):
        return None
    digits = re.sub(r'\D', '', str(phone))
    return digits[1:] if digits.startswith('1') else digits


def clean_transcript_text(text: str) -> str:
    """Clean transcript text to prevent CSV fragmentation"""
    if pd.isna(text) or not text:
        return ""
    
    # Convert to string
    text_str = str(text)
    
    # Replace problematic characters that can fragment CSV
    text_str = text_str.replace('\r\n', ' ')  # Windows line endings
    text_str = text_str.replace('\r', ' ')    # Mac line endings
    text_str = text_str.replace('\n', ' ')    # Unix line endings
    text_str = text_str.replace('\t', ' ')    # Tabs
    
    # Replace multiple spaces with single space
    text_str = re.sub(r'\s+', ' ', text_str)
    
    # Remove or replace control characters (except common ones)
    text_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text_str)
    
    # Trim whitespace
    text_str = text_str.strip()
    
    return text_str


def load_existing_second_chance_leads() -> pd.DataFrame:
    """Load existing second chance leads from persistent files"""
    parquet_file = SECOND_CHANCE_DATA_DIR / "second_chance_leads.parquet"
    csv_file = SECOND_CHANCE_DATA_DIR / "second_chance_leads.csv"
    
    if parquet_file.exists():
        try:
            df = pd.read_parquet(parquet_file)
            print(f"Loaded {len(df)} existing second chance leads from Parquet")
            return df
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
    
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} existing second chance leads from CSV")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
    
    print("No existing second chance leads file found, starting fresh")
    return pd.DataFrame()


def save_persistent_second_chance_leads(df: pd.DataFrame, changes_log: List[Dict[str, Any]]) -> None:
    """Save second chance leads to persistent files and log changes"""
    if df.empty:
        print("No data to save")
        return
    
    # Ensure directory exists
    SECOND_CHANCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet (primary format)
    parquet_file = SECOND_CHANCE_DATA_DIR / "second_chance_leads.parquet"
    df['call_id'] = df.get('call_id', "").fillna("").astype(str)
    df.to_parquet(parquet_file, index=False, engine='pyarrow')
    print(f"Saved {len(df)} records to {parquet_file}")
    
    # Save to CSV (legacy support)
    csv_file = SECOND_CHANCE_DATA_DIR / "second_chance_leads.csv"
    df.to_csv(csv_file, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"Saved {len(df)} records to {csv_file}")
    
    # Save changes log
    if changes_log:
        changes_file = SECOND_CHANCE_DATA_DIR / "second_chance_leads_changes.json"
        with open(changes_file, 'w') as f:
            json.dump(changes_log, f, indent=2)
        print(f"Logged {len(changes_log)} changes to {changes_file}")


def load_existing_invalidated_leads() -> pd.DataFrame:
    """Load existing invalidated second chance leads from persistent files"""
    parquet_file = SECOND_CHANCE_DATA_DIR / "invalidated_second_chance_leads.parquet"
    csv_file = SECOND_CHANCE_DATA_DIR / "invalidated_second_chance_leads.csv"
    
    if parquet_file.exists():
        try:
            df = pd.read_parquet(parquet_file)
            print(f"Loaded {len(df)} existing invalidated leads from Parquet")
            return df
        except Exception as e:
            print(f"Error loading invalidated Parquet file: {e}")
    
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} existing invalidated leads from CSV")
            return df
        except Exception as e:
            print(f"Error loading invalidated CSV file: {e}")
    
    print("No existing invalidated leads file found, starting fresh")
    return pd.DataFrame()


def save_persistent_invalidated_leads(df: pd.DataFrame) -> None:
    """Save invalidated second chance leads to persistent files"""
    if df.empty:
        print("No invalidated data to save")
        return
    
    # Ensure directory exists
    SECOND_CHANCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet (primary format)
    parquet_file = SECOND_CHANCE_DATA_DIR / "invalidated_second_chance_leads.parquet"
    df['call_id'] = df.get('call_id', "").fillna("").astype(str)
    df.to_parquet(parquet_file, index=False, engine='pyarrow')
    print(f"Saved {len(df)} invalidated records to {parquet_file}")
    
    # Save to CSV (legacy support)
    csv_file = SECOND_CHANCE_DATA_DIR / "invalidated_second_chance_leads.csv"
    df.to_csv(csv_file, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"Saved {len(df)} invalidated records to {csv_file}")


def update_persistent_files(new_results: List[Dict[str, Any]], customer_profiles: List[Dict[str, Any]]) -> None:
    """Update persistent second chance leads files with new results"""
    # Load existing data
    existing_df = load_existing_second_chance_leads()
    changes_log = []
    
    # Convert new results to DataFrame
    if not new_results:
        print("No new results to process")
        return
    
    # Separate valid and invalidated leads
    valid_rows = []
    invalidated_rows = []
    
    for result in new_results:
        # Check if this lead is invalidated
        invalidated = bool(result.get('invalidated', False))
        
        # Extract referenced transcript text and calculate length
        referenced_texts = []
        total_length = 0
        
        # Get call details from the first referenced transcript
        primary_call_details = {}
        primary_call_id = ""
        referenced_transcripts = result.get('referenced_transcripts', [])
        if referenced_transcripts:
            primary_call_id = referenced_transcripts[0].get('call_id', '')
            primary_call_details = get_call_details_for_call_id(primary_call_id, customer_profiles)
        
        for transcript_ref in result.get('referenced_transcripts', []):
            call_id = transcript_ref.get('call_id', '')
            snippet = transcript_ref.get('snippet', '')
            
            # Find the full transcript for this call_id
            full_transcript = get_full_transcript_for_call_id(call_id, customer_profiles)
            
            # Use full transcript if available, otherwise use snippet
            transcript_text = full_transcript if full_transcript else snippet
            referenced_texts.append(transcript_text)
            total_length += len(transcript_text)
        
        # Join all referenced transcripts with separator
        referenced_text_combined = ' | '.join(referenced_texts)
        
        # Get verification data
        verification_data = result.get('verification', {})
        
        # Safely extract all fields with defaults
        customer_call_analysis = result.get('customer_call_analysis', {})
        recommendations = result.get('recommendations', {})
        
        # Base row data (common to both valid and invalidated)
        base_row = {
            'customer_phone': result.get('customer_phone', ''),
            'analysis_timestamp': result.get('analysis_timestamp', ''),
            'reasoning': result.get('reasoning', ''),
            'referenced_transcripts': json.dumps(result.get('referenced_transcripts', [])),
            'referenced_transcript_text': referenced_text_combined,
            'referenced_transcript_length': total_length,
            'was_customer_call': bool(customer_call_analysis.get('was_customer_call', False)),
            'was_service_request': bool(customer_call_analysis.get('was_service_request', False)),
            'was_booked': bool(customer_call_analysis.get('was_booked', False)),
            'booking_failure_reason': customer_call_analysis.get('booking_failure_reason', ''),
            'conversion_potential': customer_call_analysis.get('conversion_potential', ''),
            'immediate_action': recommendations.get('immediate_action', ''),
            'follow_up_strategy': recommendations.get('follow_up_strategy', ''),
            'agent_training_notes': recommendations.get('agent_training_notes', ''),
            'call_id': primary_call_id,
            'primary_call_direction': primary_call_details.get('direction', ''),
            'primary_call_date': primary_call_details.get('call_date', ''),
            'primary_call_time': primary_call_details.get('call_time', ''),
            'primary_call_agent_name': primary_call_details.get('agent_name', '')
        }
        
        if invalidated:
            # Invalidated lead - include invalidation fields, exclude from main output
            invalidated_row = base_row.copy()
            invalidated_row['is_second_chance_lead'] = bool(verification_data.get('is_valid_second_chance_lead', result.get('is_second_chance_lead', False)))
            invalidated_row['invalidated'] = True
            invalidated_row['invalidation_reason'] = result.get('invalidation_reason', '')
            invalidated_row['new_job_activity_detected'] = bool(verification_data.get('new_job_activity_detected', False))
            invalidated_row['invalidating_job_data'] = json.dumps(verification_data.get('invalidating_job')) if verification_data.get('invalidating_job') else None
            invalidated_rows.append(invalidated_row)
        else:
            # Valid lead - exclude invalidation fields
            valid_row = base_row.copy()
            valid_row['is_second_chance_lead'] = bool(verification_data.get('is_valid_second_chance_lead', result.get('is_second_chance_lead', False)))
            valid_row['new_job_activity_detected'] = bool(verification_data.get('new_job_activity_detected', False))
            valid_row['invalidating_job_data'] = json.dumps(verification_data.get('invalidating_job')) if verification_data.get('invalidating_job') else None
            valid_rows.append(valid_row)
    
    # Process valid leads
    new_df = pd.DataFrame(valid_rows)
    
    # Process updates and new records
    updated_count = 0
    appended_count = 0
    
    for _, new_row in new_df.iterrows():
        customer_phone = new_row['customer_phone']
        
        # Check if customer already exists (only if existing_df is not empty)
        if not existing_df.empty and 'customer_phone' in existing_df.columns:
            existing_mask = existing_df['customer_phone'] == customer_phone
            if existing_mask.any():
                # Update existing record
                existing_idx = existing_df[existing_mask].index[0]
                old_row = existing_df.loc[existing_idx].to_dict()
                
                # Update the row
                existing_df.loc[existing_idx] = new_row
                
                # Log the change
                change_entry = {
                    'timestamp': new_row['analysis_timestamp'],
                    'customer_phone': customer_phone,
                    'action': 'updated',
                    'changes': {}
                }
                
                # Track specific field changes
                for key in new_row.index:
                    if key in old_row and str(old_row[key]) != str(new_row[key]):
                        change_entry['changes'][key] = {
                            'old_value': str(old_row[key]),
                            'new_value': str(new_row[key])
                        }
                
                changes_log.append(change_entry)
                updated_count += 1
            else:
                # Append new record
                existing_df = pd.concat([existing_df, new_row.to_frame().T], ignore_index=True)
                
                # Log the addition
                change_entry = {
                    'timestamp': new_row['analysis_timestamp'],
                    'customer_phone': customer_phone,
                    'action': 'added',
                    'new_record': new_row.to_dict()
                }
                changes_log.append(change_entry)
                appended_count += 1
        else:
            # No existing data, append new record
            existing_df = pd.concat([existing_df, new_row.to_frame().T], ignore_index=True)
            
            # Log the addition
            change_entry = {
                'timestamp': new_row['analysis_timestamp'],
                'customer_phone': customer_phone,
                'action': 'added',
                'new_record': new_row.to_dict()
            }
            changes_log.append(change_entry)
            appended_count += 1
    
    # Save updated files for valid leads
    save_persistent_second_chance_leads(existing_df, changes_log)
    
    print(f"Persistent files updated: {updated_count} records updated, {appended_count} records added")
    
    # Process invalidated leads separately
    if invalidated_rows:
        invalidated_df = pd.DataFrame(invalidated_rows)
        existing_invalidated_df = load_existing_invalidated_leads()
        
        # Merge with existing invalidated leads (append new ones)
        if not existing_invalidated_df.empty:
            # Remove duplicates based on customer_phone
            existing_phones = set(existing_invalidated_df['customer_phone'].astype(str))
            new_invalidated_df = invalidated_df[~invalidated_df['customer_phone'].astype(str).isin(existing_phones)]
            if not new_invalidated_df.empty:
                combined_invalidated_df = pd.concat([existing_invalidated_df, new_invalidated_df], ignore_index=True)
            else:
                combined_invalidated_df = existing_invalidated_df
        else:
            combined_invalidated_df = invalidated_df
        
        save_persistent_invalidated_leads(combined_invalidated_df)
        print(f"Invalidated leads saved: {len(invalidated_rows)} new invalidated records")


def get_call_details_for_call_id(call_id: str, customer_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find call details (direction, call_date, call_time, agent_name) for a given call_id"""
    for profile in customer_profiles:
        for call in profile.get('calls', []):
            if call.get('call_id') == call_id:
                return {
                    'direction': call.get('direction', ''),
                    'call_date': call.get('call_date', ''),
                    'call_time': call.get('call_time', ''),
                    'agent_name': call.get('agent_name', '')
                }
    return {
        'direction': '',
        'call_date': '',
        'call_time': '',
        'agent_name': ''
    }


def get_full_transcript_for_call_id(call_id: str, customer_profiles: List[Dict[str, Any]]) -> str:
    """Find the full transcript text for a given call_id"""
    for profile in customer_profiles:
        for call in profile.get('calls', []):
            if call.get('call_id') == call_id:
                transcript_list = call.get('transcript', [])
                if transcript_list:
                    # Reconstruct full transcript from list of speaker-message dictionaries
                    full_transcript = []
                    for entry in transcript_list:
                        speaker = entry.get('speaker', '')
                        message = entry.get('message', '')
                        full_transcript.append(f"{speaker}: {message}")
                    return ' '.join(full_transcript)
    return ""


def backfill_processed_customers_from_persistent_files() -> None:
    """Backfill processed_customers.jsonl from existing persistent second chance leads files"""
    print("Backfilling processed_customers.jsonl from existing persistent files...")
    
    # Load existing persistent files
    persistent_df = load_existing_second_chance_leads()
    if persistent_df.empty:
        print("No existing persistent files found, skipping backfill")
        return
    
    # Load existing customer profiles to get call IDs
    customer_profiles_dict = load_existing_customer_profiles()
    if not customer_profiles_dict:
        print("No existing customer profiles found, skipping backfill")
        return
    
    # Create a mapping of customer phone to their call IDs
    customer_call_mapping = {}
    for customer_phone, profile in customer_profiles_dict.items():
        call_ids = [call.get('call_id', '') for call in profile.get('calls', [])]
        if customer_phone and call_ids:
            customer_call_mapping[customer_phone] = call_ids
    
    # Process each customer in the persistent files
    backfilled_count = 0
    for _, row in persistent_df.iterrows():
        customer_phone = row.get('customer_phone', '')
        analysis_timestamp = row.get('analysis_timestamp', '')
        
        if customer_phone in customer_call_mapping:
            call_ids = customer_call_mapping[customer_phone]
            
            # Check if this customer is already in processed_customers.jsonl
            if not is_customer_already_processed(customer_phone):
                # Add to processed_customers.jsonl
                processed_customer_data = {
                    'customer_phone': customer_phone,
                    'processed_call_ids': call_ids,
                    'last_processed': analysis_timestamp,
                    'total_calls_processed': len(call_ids)
                }
                
                save_processed_customer_data(processed_customer_data)
                backfilled_count += 1
                print(f"Backfilled customer {customer_phone} with {len(call_ids)} call IDs")
    
    print(f"Backfill completed: {backfilled_count} customers added to processed_customers.jsonl")


def is_customer_already_processed(customer_phone: str) -> bool:
    """Check if a customer is already in processed_customers.jsonl"""
    processed_file = SECOND_CHANCE_DATA_DIR / "processed_customers.jsonl"
    if not processed_file.exists():
        return False
    
    try:
        with open(processed_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        if data.get('customer_phone') == customer_phone:
                            return True
                    except json.JSONDecodeError:
                        # Skip malformed lines silently to avoid spam
                        continue
    except Exception as e:
        print(f"Error reading processed_customers.jsonl: {e}")
    
    return False


def save_processed_customer_data(customer_data: Dict[str, Any]) -> None:
    """Save customer data to processed_customers.jsonl"""
    processed_file = SECOND_CHANCE_DATA_DIR / "processed_customers.jsonl"
    
    try:
        with open(processed_file, 'a') as f:
            f.write(json.dumps(customer_data) + '\n')
    except Exception as e:
        print(f"Error saving to processed_customers.jsonl: {e}")


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file"""
    import hashlib
    
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256()
            while chunk := f.read(8192):
                file_hash.update(chunk)
            return file_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return ""


def is_duplicate_download(file_path: Path) -> bool:
    """Check if the downloaded file is identical to the previous download"""
    hash_file = SECOND_CHANCE_DATA_DIR / "last_download_hash.txt"
    
    if not file_path.exists():
        return False
    
    current_hash = calculate_file_hash(file_path)
    if not current_hash:
        return False
    
    # Check if we have a previous hash
    if hash_file.exists():
        try:
            with open(hash_file, 'r') as f:
                previous_hash = f.read().strip()
            
            if current_hash == previous_hash:
                print(f"Downloaded file is identical to previous download (hash: {current_hash[:16]}...)")
                # Delete the duplicate file to save storage
                try:
                    file_path.unlink()
                    print(f"Deleted duplicate file: {file_path}")
                except Exception as e:
                    print(f"Error deleting duplicate file: {e}")
                return True
        except Exception as e:
            print(f"Error reading hash file: {e}")
    
    # Save current hash for next comparison
    try:
        with open(hash_file, 'w') as f:
            f.write(current_hash)
        print(f"Saved new download hash: {current_hash[:16]}...")
    except Exception as e:
        print(f"Error saving hash file: {e}")
    
    return False


def archive_old_downloaded_files(current_file: Path) -> None:
    """Archive old downloaded CSV files to backup directory"""
    backup_dir = Path("Z:/datahound_backups/mcc_transcription_archive")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all old transcript files in the second_chance_data directory
    transcript_files = list(SECOND_CHANCE_DATA_DIR.glob("mcc_transcript_*.csv"))
    
    # Remove the current file from the list
    transcript_files = [f for f in transcript_files if f != current_file]
    
    if not transcript_files:
        print("No old transcript files to archive")
        return
    
    archived_count = 0
    for old_file in transcript_files:
        try:
            # Create backup filename with timestamp
            backup_filename = old_file.name
            backup_path = backup_dir / backup_filename
            
            # Move the file to backup directory
            shutil.move(str(old_file), str(backup_path))
            print(f"Archived {old_file.name} to {backup_path}")
            archived_count += 1
        except Exception as e:
            print(f"Error archiving {old_file.name}: {e}")
    
    print(f"Archived {archived_count} old transcript files to {backup_dir}")


def filter_by_phone_length(path: Path) -> pd.DataFrame:
    """Filter CSV to keep only rows with 10-digit phone numbers"""
    print(f"Filtering rows by phone number length in {path}")
    
    try:
        # Read the saved CSV file
        df = pd.read_csv(path)
        
        if df.empty:
            print("No data to filter")
            return df
        
        # Filter for rows with exactly 10-digit phone numbers
        phone_mask = df['Contact Phone'].apply(
            lambda x: pd.notna(x) and len(str(int(x))) == 10
        )
        
        filtered_df = df[phone_mask].copy()
        
        # Save the filtered data back to the same file
        filtered_df.to_csv(path, index=False, sep=',', quoting=csv.QUOTE_ALL, quotechar='"')
        
        removed_count = len(df) - len(filtered_df)
        print(f"Phone filtering: kept {len(filtered_df)} rows, removed {removed_count} rows")
        
        return filtered_df
        
    except Exception as e:
        print(f"Error filtering by phone length: {e}")
        # If filtering fails, return the original dataframe
        try:
            return pd.read_csv(path)
        except:
            return pd.DataFrame()


def validate_and_clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and remove corrupted rows from the dataframe"""
    print(f"Validating {len(df)} rows for corruption...")
    
    # Define expected patterns for each column
    validation_rules = {
        'Transcription ID': lambda x: int(x) > 0,
        'Contact Phone': lambda x: len(str(int(x))) == 10,
        'Talk Time (sec)': lambda x: int(x) >= 0,
        'User': lambda x: isinstance(x, str) and len(x.strip()) > 0,
        'Direction': lambda x: isinstance(x, str) and x.strip() in ['Inbound', 'Outbound'],
        'Call Date': lambda x: isinstance(x, str) and len(x.strip()) > 0,
        'Call Time': lambda x: isinstance(x, str) and len(x.strip()) > 0,
    }
    
    # Track validation results
    valid_rows = []
    corrupted_count = 0
    
    for idx, row in df.iterrows():
        is_valid = True
        corruption_reasons = []
        
        # Check each validation rule
        for column, validator in validation_rules.items():
            if column in row:
                try:
                    result = validator(row[column])
                    if not result:
                        is_valid = False
                        corruption_reasons.append(f"{column}: invalid value ({row[column]})")
                except Exception as e:
                    is_valid = False
                    corruption_reasons.append(f"{column}: validation error ({e})")
            else:
                is_valid = False
                corruption_reasons.append(f"{column}: missing column")
        
        # Additional checks for data integrity
        # Check if Transcription field contains reasonable content
        if 'Transcription' in row:
            transcription = str(row['Transcription'])
            # Transcription should contain dialogue markers (Agent:, Contact:)
            if not ('Agent:' in transcription or 'Contact:' in transcription):
                is_valid = False
                corruption_reasons.append("Transcription: missing dialogue markers")
            
            # Transcription shouldn't be too short (less than 50 chars is suspicious)
            if len(transcription) < 50:
                is_valid = False
                corruption_reasons.append("Transcription: too short")
            
            # Check for fragmented transcription content (indicates row splitting)
            # Look for patterns that suggest the transcription was split across rows
            fragmented_patterns = [
                'rvice and. in the me I\'m told th no specifi hey my uh the guy wi go over ar definitely OK um so I\'ll defi thank you we\'ll see y',
                'Interaction Agent: It\'s a great day here. Agent: I\'m calling Heating and Air. Agent: This is Monica. Agent: Am I speaking with Jason?'
            ]
            
            if any(pattern in transcription for pattern in fragmented_patterns):
                is_valid = False
                corruption_reasons.append("Transcription: fragmented content detected")
        
        # Check if Talkdesk Activity contains expected pattern
        if 'Talkdesk Activity: Talkdesk Activity Name' in row:
            activity = str(row['Talkdesk Activity: Talkdesk Activity Name'])
            if not ('Interaction' in activity or 'Inbound' in activity or 'Outbound' in activity):
                is_valid = False
                corruption_reasons.append("Talkdesk Activity: invalid pattern")
        
        if is_valid:
            valid_rows.append(row)
        else:
            corrupted_count += 1
            if corrupted_count <= 5:  # Only show first 5 corruption examples
                print(f"Corrupted row {idx}: {', '.join(corruption_reasons)}")
    
    if corrupted_count > 5:
        print(f"... and {corrupted_count - 5} more corrupted rows")
    
    print(f"Validation complete: {len(valid_rows)} valid rows, {corrupted_count} corrupted rows removed")
    
    # Return validated dataframe
    if valid_rows:
        return pd.DataFrame(valid_rows).reset_index(drop=True)
    else:
        return pd.DataFrame()


def clean_csv_formatting(path: Path) -> None:
    """Clean CSV formatting issues like unescaped newlines and quotes"""
    print(f"Cleaning CSV formatting in {path}")
    
    try:
        # Read the file as text
        with path.open('r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace carriage returns
        content = content.replace('\r', '')
        
        # Fix the main issue: unescaped newlines within quoted fields
        # The pattern is: quote, content with newlines, quote
        # We need to replace newlines within quotes with spaces
        
        lines = content.split('\n')
        cleaned_lines = []
        in_quoted_field = False
        current_line = ""
        
        for line in lines:
            # Check if this line starts a quoted field
            if line.count('"') > 0 and not in_quoted_field:
                # Count quotes to see if this line contains a complete quoted field
                quote_count = line.count('"')
                if quote_count % 2 == 0:
                    # Complete quoted field on this line
                    cleaned_lines.append(line)
                else:
                    # Start of multi-line quoted field
                    in_quoted_field = True
                    current_line = line
            elif in_quoted_field:
                # Continue building the multi-line quoted field
                current_line += " " + line
                # Check if this line ends the quoted field
                if current_line.count('"') % 2 == 0:
                    # End of quoted field
                    cleaned_lines.append(current_line)
                    current_line = ""
                    in_quoted_field = False
            else:
                # Normal line
                cleaned_lines.append(line)
        
        # Add any remaining line
        if current_line:
            cleaned_lines.append(current_line)
        
        # Write the cleaned content back
        with path.open('w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        
        print(f"CSV formatting cleaned, reduced from {len(lines)} to {len(cleaned_lines)} lines")
        
    except Exception as e:
        print(f"Error cleaning CSV formatting: {e}")
        # If cleaning fails, continue with original file
        pass


def process_transcripts(path: Path) -> None:
    """Process transcript CSV file with normalization and filtering"""
    print(f"Processing transcripts in {path}")
    
    try:
        # First, clean the raw CSV file to handle formatting issues
        clean_csv_formatting(path)
        
        # Read CSV with error handling for malformed data
        try:
            df = pd.read_csv(path, skiprows=[0], sep=',', quoting=csv.QUOTE_ALL, quotechar='"', engine='python')
        except Exception as e:
            print(f"Error reading CSV with quotes: {e}")
            # Try reading without quotes
            try:
                df = pd.read_csv(path, skiprows=[0], sep=',', engine='python', on_bad_lines='skip')
            except Exception as e2:
                print(f"Error reading CSV without quotes: {e2}")
                # Last resort: read with minimal processing
                df = pd.read_csv(path, skiprows=[0], sep=',', engine='python', on_bad_lines='skip', quoting=csv.QUOTE_NONE)
        
        if df.empty:
            print("No data to process")
            return
        
        # Add unique Transcription ID column
        df.insert(0, 'Transcription ID', range(1, len(df) + 1))
        
        # Strip quote characters from Transcription column
        if 'Transcription' in df.columns:
            df['Transcription'] = df['Transcription'].str.replace('"', '', regex=False)
        
        # Remove rows with empty Contact Phone
        df = df[df['Contact Phone'].notna()]
        
        # Normalize phone numbers
        df['Contact Phone'] = df['Contact Phone'].apply(normalize_phone)
        
        # Filter for 10-digit phone numbers, excluding None values
        df = df[df['Contact Phone'].notna() & (df['Contact Phone'].str.len() == 10)]
        
        # Split Start Time into Call Date and Call Time
        if 'Start Time' in df.columns:
            df[['Call Date', 'Call Time']] = df['Start Time'].str.split(', ', expand=True)
        
        # Fill empty Direction values based on Talkdesk Activity
        if 'Direction' in df.columns and 'Talkdesk Activity: Talkdesk Activity Name' in df.columns:
            df['Direction'] = df.apply(
                lambda row: (
                    'Inbound' if 'Inbound' in str(row['Talkdesk Activity: Talkdesk Activity Name']) 
                    else 'Outbound' if 'Outbound' in str(row['Talkdesk Activity: Talkdesk Activity Name']) 
                    else row['Direction']
                ), axis=1
            )
        
        # Filter for rows with no NaN values
        complete_rows_df = df.dropna()
        
        # Validate and clean corrupted rows
        validated_df = validate_and_clean_rows(complete_rows_df)
        
        # NEW: Filter out transcripts shorter than 250 characters
        print(f"Filtering transcripts by length (minimum 250 characters)...")
        initial_count = len(validated_df)
        validated_df = validated_df[validated_df['Transcription'].str.len() >= 250]
        filtered_count = len(validated_df)
        removed_count = initial_count - filtered_count
        print(f"Transcription length filtering: kept {filtered_count} rows, removed {removed_count} rows with <250 characters")
        
        # Save processed data
        validated_df.to_csv(path, index=False, sep=',', quoting=csv.QUOTE_ALL, quotechar='"')
        
        # Post-save filtering: remove rows without 10-digit phone numbers
        final_df = filter_by_phone_length(path)
        
        print(f"Processed {len(validated_df)} validated rows, removed {len(df) - len(validated_df)} corrupted/incomplete rows")
        print(f"Final output: {len(final_df)} rows with 10-digit phone numbers")
        
    except Exception as e:
        print(f"Error processing transcripts: {e}")
        # If processing fails, keep the original file
        pass


def clean_csv_file(path: Path) -> None:
    """Process downloaded CSV file with transcript-specific logic"""
    process_transcripts(path)


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


class SecondChanceLeadAnalyzer:
    def __init__(self, api_key: str = "", jobs_df: pd.DataFrame = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=DEEPSEEK_BASE_URL
        )
        self.analysis_results = []
        self.results_lock = Lock()
        self.processed_customers = load_processed_customers()
        self.jobs_df = jobs_df if jobs_df is not None else pd.DataFrame()
    
    def create_second_chance_prompt(self, customer_profile: Dict[str, Any]) -> str:
        return f"""You are an expert HVAC industry analyst reviewing call transcripts for McCullough Heating and Air to identify "second chance leads".

CUSTOMER PROFILE:
- Phone: {customer_profile['customer_phone']}
- Total Calls: {customer_profile['total_calls']}
- Invalid Calls Dropped: {customer_profile['invalid_calls']}
- Total Talk Time: {customer_profile['summary']['total_talk_time_seconds']} seconds
- Call Date Range: {customer_profile['summary']['date_range']['first_call']} to {customer_profile['summary']['date_range']['last_call']}
- Agents Involved: {', '.join(customer_profile['summary']['agents_involved'])}

CALL TRANSCRIPTS:
{self._format_calls_for_prompt(customer_profile['calls'])}

ANALYSIS REQUIREMENTS:
Follow this EXACT step-by-step process to determine if this customer represents a "second chance lead":

**STEP 1: Determine if this was a customer call**
- Was the caller a customer seeking HVAC services? (not a sales call, verification call, or spam)
- Set was_customer_call = true/false

**STEP 2: Determine if this was a service request**
- Did the customer request HVAC service, maintenance, or repair?
- Set was_service_request = true/false

**STEP 3: Determine if the service was booked**
- Was the service request scheduled/appointed/booked?
- Set was_booked = true/false

**STEP 4: Apply the logic rules**
- If was_customer_call = false → is_second_chance_lead = false
- If was_service_request = false → is_second_chance_lead = false
- If was_booked = true → is_second_chance_lead = false
- ONLY if ALL THREE are true (was_customer_call=true, was_service_request=true, was_booked=false) → is_second_chance_lead = true

**STEP 5: Check for invalidation reasons**
- Review the call transcripts for ANY of the following invalidation reasons:
  * Call was to confirm they received parts
  * Call was to reschedule an appointment
  * Call was because they missed the technician's call
  * Call was to request an invoice
  * Call was to approve an estimate
  * Job was outside the service area
  * Accidental call
  * Call was to ask about a past estimate
  * Call was to sign up for a membership
  * Call was to request the phone number for the Texas Department of Licensing
  * Job was for a manufactured or mobile home
  * Call was about an issue from a previous job
  * Caller was not the homeowner
- If ANY of these reasons apply, set invalidated = true and provide the specific invalidation_reason
- If NONE of these reasons apply, set invalidated = false and leave invalidation_reason as empty string
- CRITICAL: If invalidated = true, invalidation_reason MUST contain the specific reason. If invalidated = false, invalidation_reason MUST be empty string.

**STEP 6: Write consistent reasoning**
- Your reasoning MUST match your is_second_chance_lead determination
- If is_second_chance_lead = true, explain WHY it meets all criteria
- If is_second_chance_lead = false, explain WHY it fails one or more criteria
- NEVER write reasoning that contradicts your determination

**CRITICAL CONSISTENCY RULES**: 
1. Your reasoning text MUST align with your is_second_chance_lead boolean value. If you determine is_second_chance_lead = true, your reasoning must explain why it IS a second chance lead. If you determine is_second_chance_lead = false, your reasoning must explain why it is NOT a second chance lead.
2. The invalidated field and invalidation_reason field MUST be consistent: if invalidated = true, invalidation_reason must contain text. If invalidated = false, invalidation_reason must be empty string.

Provide a JSON response with EXACTLY this structure:

{{
    "is_second_chance_lead": "boolean (true/false)",
    "invalidated": "boolean (true if any invalidation reason applies, false otherwise)",
    "invalidation_reason": "string (specific invalidation reason if invalidated=true, empty string if invalidated=false)",
    "reasoning": "string (detailed explanation that MUST match your is_second_chance_lead determination)",
    "referenced_transcripts": [
        {{
            "call_id": "number",
            "snippet": "string (exact transcript excerpt that supports your determination)",
            "relevance": "string (why this snippet is relevant)"
        }}
    ],
    "customer_call_analysis": {{
        "was_customer_call": "boolean (true if caller was a customer)",
        "was_service_request": "boolean (true if call was for service)",
        "was_booked": "boolean (true if service was scheduled)",
        "booking_failure_reason": "string (why booking failed if applicable)",
        "conversion_potential": "string (High/Medium/Low)"
    }},
    "recommendations": {{
        "immediate_action": "string",
        "follow_up_strategy": "string",
        "agent_training_notes": "string"
    }}
}}

Provide ONLY the JSON response with no additional text."""

    def _format_calls_for_prompt(self, calls: List[Dict[str, Any]]) -> str:
        formatted_calls = []
        for call in calls:
            call_header = f"\n--- CALL {call['call_id']} ({call['call_date']} at {call['call_time']}) ---"
            call_header += f"\nAgent: {call['agent_name']} | Direction: {call['direction']} | Duration: {call['talk_time_seconds']}s\n"
            
            conversation = []
            for exchange in call['transcript']:
                role = "AGENT" if exchange['role'] == 'agent' else "CUSTOMER"
                conversation.append(f"{role}: {exchange['message']}")
            
            formatted_calls.append(call_header + "\n".join(conversation))
        
        return "\n".join(formatted_calls)

    def _fix_json_issues(self, content: str) -> str:
        """Fix common JSON parsing issues using json-repair library"""
        try:
            # Use the json-repair library to fix common JSON issues
            repaired_content = repair_json(content)
            return repaired_content
        except Exception as e:
            print(f"JSON repair failed: {e}")
            # Fallback to basic fixes
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            return content

    def _validate_analysis_logic(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and correct analysis logic inconsistencies"""
        # Safely extract all fields with defaults
        customer_call_analysis = analysis_data.get('customer_call_analysis', {})
        was_customer_call = customer_call_analysis.get('was_customer_call', False)
        was_service_request = customer_call_analysis.get('was_service_request', False)
        was_booked = customer_call_analysis.get('was_booked', False)
        is_second_chance_lead = analysis_data.get('is_second_chance_lead', False)
        reasoning = analysis_data.get('reasoning', '').lower()
        
        # Check for reasoning-text contradictions
        contradiction_detected = False
        contradiction_reason = ""
        
        if is_second_chance_lead:
            # If marked as second chance lead, check for negative reasoning
            negative_phrases = [
                "does not meet the criteria",
                "does not qualify",
                "not a second chance lead",
                "cannot be considered",
                "fails to meet",
                "does not represent",
                "is not a second chance"
            ]
            
            for phrase in negative_phrases:
                if phrase in reasoning:
                    contradiction_detected = True
                    contradiction_reason = f"Reasoning contradicts determination: '{phrase}' found in reasoning but is_second_chance_lead=True"
                    break
        
        else:
            # If marked as NOT second chance lead, check for positive reasoning
            positive_phrases = [
                "is a second chance lead",
                "qualifies as a second chance",
                "meets the criteria",
                "represents a second chance",
                "should be considered"
            ]
            
            for phrase in positive_phrases:
                if phrase in reasoning:
                    contradiction_detected = True
                    contradiction_reason = f"Reasoning contradicts determination: '{phrase}' found in reasoning but is_second_chance_lead=False"
                    break
        
        # Apply boolean logic validation
        boolean_logic_error = False
        logic_error_reason = ""
        
        if is_second_chance_lead:
            # For a second chance lead, ALL three conditions must be met:
            # 1. was_customer_call = TRUE
            # 2. was_service_request = TRUE  
            # 3. was_booked = FALSE
            
            if not was_customer_call:
                boolean_logic_error = True
                logic_error_reason = "was_customer_call=False"
            elif not was_service_request:
                boolean_logic_error = True
                logic_error_reason = "was_service_request=False"
            elif was_booked:
                boolean_logic_error = True
                logic_error_reason = "was_booked=True"
        
        # Validate invalidation fields
        invalidated = analysis_data.get('invalidated', False)
        invalidation_reason = analysis_data.get('invalidation_reason', '')
        invalidation_contradiction = False
        invalidation_contradiction_reason = ""
        
        if invalidated:
            if not invalidation_reason or invalidation_reason.strip() == "":
                invalidation_contradiction = True
                invalidation_contradiction_reason = "invalidated=true but invalidation_reason is empty"
        else:
            if invalidation_reason and invalidation_reason.strip() != "":
                invalidation_contradiction = True
                invalidation_contradiction_reason = "invalidated=false but invalidation_reason contains text"
        
        if invalidation_contradiction:
            customer_phone = analysis_data.get('customer_phone', 'unknown')
            print(f"INVALIDATION CONTRADICTION: Customer {customer_phone} - {invalidation_contradiction_reason}")
            if invalidated and not invalidation_reason:
                analysis_data['invalidated'] = False
                analysis_data['invalidation_reason'] = ""
            elif not invalidated and invalidation_reason:
                analysis_data['invalidation_reason'] = ""
        
        if invalidated and not invalidation_contradiction:
            customer_phone = analysis_data.get('customer_phone', 'unknown')
            print(f"INVALIDATION DETECTED: Customer {customer_phone} - {invalidation_reason}")
            analysis_data['is_second_chance_lead'] = False
        
        # Apply corrections
        if contradiction_detected or boolean_logic_error:
            customer_phone = analysis_data.get('customer_phone', 'unknown')
            
            if contradiction_detected:
                print(f"REASONING CONTRADICTION: Customer {customer_phone} - {contradiction_reason}")
                analysis_data['is_second_chance_lead'] = False
                analysis_data['reasoning'] = f"LOGIC CORRECTED: Reasoning contradiction detected. Original reasoning: '{analysis_data.get('reasoning', '')}' - Corrected to FALSE due to contradictory reasoning."
            
            elif boolean_logic_error:
                print(f"BOOLEAN LOGIC ERROR: Customer {customer_phone} marked as second chance lead but {logic_error_reason}. Correcting to FALSE.")
                analysis_data['is_second_chance_lead'] = False
                analysis_data['reasoning'] = f"LOGIC CORRECTED: {analysis_data.get('reasoning', '')} - Cannot be a second chance lead because {logic_error_reason}."
        
        return analysis_data

    def _get_full_transcript_for_call_id(self, call_id: str, customer_profiles: List[Dict[str, Any]]) -> str:
        """Find the full transcript text for a given call_id"""
        for profile in customer_profiles:
            for call in profile.get('calls', []):
                if call.get('call_id') == call_id:
                    # Reconstruct the full transcript from the formatted conversation
                    transcript_parts = []
                    for exchange in call.get('transcript', []):
                        role = exchange.get('role', '')
                        message = exchange.get('message', '')
                        if role == 'agent':
                            transcript_parts.append(f"Agent: {message}")
                        elif role == 'customer':
                            transcript_parts.append(f"Contact: {message}")
                    return '\n'.join(transcript_parts)
        return ""

    def _get_call_details_for_call_id(self, call_id: str, customer_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find call details (direction, call_date, call_time, agent_name) for a given call_id"""
        for profile in customer_profiles:
            for call in profile.get('calls', []):
                if call.get('call_id') == call_id:
                    return {
                        'direction': call.get('direction', ''),
                        'call_date': call.get('call_date', ''),
                        'call_time': call.get('call_time', ''),
                        'agent_name': call.get('agent_name', '')
                    }
        return {
            'direction': '',
            'call_date': '',
            'call_time': '',
            'agent_name': ''
        }

    def analyze_customer_profile(self, customer_profile: Dict[str, Any], max_retries: int = 2) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            print("Error: API key not provided")
            return None
        
        prompt = self.create_second_chance_prompt(customer_profile)
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert HVAC industry analyst. Provide responses in valid JSON format only. Ensure all strings are properly escaped with \\n for newlines and \\\" for quotes. The JSON must be complete and valid."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False
                )
                
                content = response.choices[0].message.content.strip()
                
                # Remove markdown code blocks if present
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                # Try to fix common JSON issues
                content = self._fix_json_issues(content)
                
                # Parse JSON response
                analysis_data = json.loads(content)
                
                # Add customer phone for tracking
                analysis_data['customer_phone'] = customer_profile['customer_phone']
                analysis_data['analysis_timestamp'] = datetime.now().isoformat()
                primary_call_id = ""
                referenced_transcripts = analysis_data.get('referenced_transcripts') or []
                if referenced_transcripts:
                    primary_call_id = str(referenced_transcripts[0].get('call_id', ''))
                analysis_data['call_id'] = primary_call_id
                
                # Set default values for invalidation fields if missing
                if 'invalidated' not in analysis_data:
                    analysis_data['invalidated'] = False
                if 'invalidation_reason' not in analysis_data:
                    analysis_data['invalidation_reason'] = ''
                
                # Validate logic consistency
                analysis_data = self._validate_analysis_logic(analysis_data)
                
                return analysis_data
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    print(f"JSON parse error on attempt {attempt + 1} for customer {customer_profile['customer_phone']}, retrying...")
                    time.sleep(1)  # Brief delay before retry
                    continue
                else:
                    error_msg = f"Failed to parse JSON response after {max_retries + 1} attempts: {e}"
                    print(f"Failed to parse JSON response for customer {customer_profile['customer_phone']}: {e}")
                    print(f"Response content: {content[:500]}...")
                    
                    # Log the failure
                    call_ids = [call['call_id'] for call in customer_profile['calls']]
                    log_analysis_failure(
                        customer_profile['customer_phone'],
                        'JSON_PARSE_ERROR',
                        error_msg,
                        call_ids,
                        datetime.now().isoformat()
                    )
                    return None
            except Exception as e:
                if attempt < max_retries:
                    print(f"API error on attempt {attempt + 1} for customer {customer_profile['customer_phone']}, retrying...")
                    time.sleep(1)  # Brief delay before retry
                    continue
                else:
                    error_msg = f"API error after {max_retries + 1} attempts: {e}"
                    print(f"API error for customer {customer_profile['customer_phone']}: {e}")
                    
                    # Log the failure
                    call_ids = [call['call_id'] for call in customer_profile['calls']]
                    log_analysis_failure(
                        customer_profile['customer_phone'],
                        'API_ERROR',
                        error_msg,
                        call_ids,
                        datetime.now().isoformat()
                    )
                    return None
        
        # If all retries failed, try a simplified prompt as fallback
        return self._fallback_analysis(customer_profile)

    def _fallback_analysis(self, customer_profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fallback analysis with simplified prompt for problematic customers"""
        try:
            # Create a much simpler prompt
            simple_prompt = f"""Analyze this customer call profile and determine if it's a second chance lead.

Customer Phone: {customer_profile['customer_phone']}
Total Calls: {customer_profile['total_calls']}

Call Summary: {customer_profile['summary']}

Check for invalidation reasons: parts confirmation, rescheduling, missed call, invoice request, estimate approval, outside service area, accidental call, past estimate inquiry, membership signup, licensing info request, manufactured/mobile home, previous job issue, not homeowner.

Provide a simple JSON response:
{{
    "is_second_chance_lead": true/false,
    "invalidated": true/false,
    "invalidation_reason": "reason if invalidated=true, empty string if invalidated=false",
    "reasoning": "brief explanation",
    "referenced_transcripts": [{{"call_id": 1, "snippet": "key text", "relevance": "why relevant"}}],
    "customer_call_analysis": {{
        "was_customer_call": true/false,
        "was_service_request": true/false,
        "was_booked": true/false,
        "booking_failure_reason": "reason if not booked",
        "conversion_potential": "High/Medium/Low"
    }},
    "recommendations": {{
        "immediate_action": "action",
        "follow_up_strategy": "strategy",
        "agent_training_notes": "notes"
    }}
}}"""

            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an HVAC analyst. Provide simple, valid JSON only."},
                    {"role": "user", "content": simple_prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent output
                max_tokens=8000,  # Reduced token limit
                stream=False
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up the response
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON response
            analysis_data = json.loads(content)
            
            # Add customer phone for tracking
            analysis_data['customer_phone'] = customer_profile['customer_phone']
            analysis_data['analysis_timestamp'] = datetime.now().isoformat()
            primary_call_id = ""
            referenced_transcripts = analysis_data.get('referenced_transcripts') or []
            if referenced_transcripts:
                primary_call_id = str(referenced_transcripts[0].get('call_id', ''))
            analysis_data['call_id'] = primary_call_id
            
            # Set default values for invalidation fields if missing
            if 'invalidated' not in analysis_data:
                analysis_data['invalidated'] = False
            if 'invalidation_reason' not in analysis_data:
                analysis_data['invalidation_reason'] = ''
            
            analysis_data['fallback_analysis'] = True  # Mark as fallback
            
            # Validate logic consistency
            analysis_data = self._validate_analysis_logic(analysis_data)
            
            print(f"Fallback analysis successful for customer {customer_profile['customer_phone']}")
            return analysis_data
            
        except Exception as e:
            print(f"Fallback analysis also failed for customer {customer_profile['customer_phone']}: {e}")
            
            # Log the failure
            call_ids = [call['call_id'] for call in customer_profile['calls']]
            log_analysis_failure(
                customer_profile['customer_phone'],
                'FALLBACK_FAILED',
                f"Both main and fallback analysis failed: {e}",
                call_ids,
                datetime.now().isoformat()
            )
            return None

    def analyze_customer_safe(self, customer_profile: Dict[str, Any], customer_num: int, total_customers: int) -> Optional[Dict[str, Any]]:
        customer_phone = customer_profile['customer_phone']
        
        # Check if customer has new calls to process
        new_calls = get_new_calls_for_customer(customer_phone, customer_profile['calls'], self.processed_customers)
        
        if not new_calls:
            print(f"Skipping customer {customer_num}/{total_customers}: {customer_phone} (no new calls)")
            return {"skipped": True, "customer_phone": customer_phone, "reason": "no_new_calls"}
        
        # Update customer profile to only include new calls
        customer_profile_with_new_calls = customer_profile.copy()
        customer_profile_with_new_calls['calls'] = new_calls
        customer_profile_with_new_calls['total_calls'] = len(new_calls)
        
        print(f"Analyzing customer {customer_num}/{total_customers}: {customer_phone} ({len(new_calls)} new calls)")
        
        analysis = self.analyze_customer_profile(customer_profile_with_new_calls)
        
        if analysis:
            # Perform verification for second chance leads
            if analysis.get('is_second_chance_lead') and not self.jobs_df.empty:
                # Get the most recent call date for verification
                latest_call_date = max(call['call_date'] for call in customer_profile_with_new_calls['calls'])
                verification_result = verify_second_chance_lead(customer_phone, latest_call_date, self.jobs_df)
                analysis['verification'] = verification_result
                
                if verification_result['new_job_activity_detected']:
                    print(f"Second chance lead invalidated for {customer_phone} - job activity detected after {latest_call_date}")
            else:
                # If not a second chance lead, ensure verification fields are set to default values
                analysis['verification'] = {
                    'is_valid_second_chance_lead': False,
                    'new_job_activity_detected': False,
                    'invalidating_job': None,
                    'total_matching_jobs': 0
                }
            
            with self.results_lock:
                self.analysis_results.append(analysis)
            
            # Save processed customer information (only new calls)
            call_ids = {call['call_id'] for call in customer_profile_with_new_calls['calls']}
            save_processed_customer(customer_phone, call_ids, analysis['analysis_timestamp'])
            
            print(f"Analysis completed for {customer_phone}")
            return analysis
        else:
            print(f"Analysis failed for {customer_phone}")
            return None

    def process_all_customers(self, customer_profiles: List[Dict[str, Any]], concurrent_calls: int = 5) -> None:
        total_customers = len(customer_profiles)
        print(f"Processing {total_customers} customer profiles with {concurrent_calls} concurrent API calls...")
        
        # Create numbered profiles for tracking
        numbered_profiles = [(i + 1, profile) for i, profile in enumerate(customer_profiles)]
        
        completed_count = 0
        failed_count = 0
        skipped_count = 0
        
        with ThreadPoolExecutor(max_workers=concurrent_calls) as executor:
            # Submit all tasks
            future_to_profile = {
                executor.submit(self.analyze_customer_safe, profile, num, total_customers): (num, profile)
                for num, profile in numbered_profiles
            }
            
            # Process completed tasks
            for future in as_completed(future_to_profile):
                num, profile = future_to_profile[future]
                try:
                    result = future.result()
                    if result:
                        if result.get("skipped"):
                            skipped_count += 1
                        else:
                            completed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"Thread error for customer {profile['customer_phone']}: {e}")
                    failed_count += 1
        
        print(f"\nAnalysis Summary:")
        print(f"Completed: {completed_count} customers")
        print(f"Skipped: {skipped_count} customers (no new calls)")
        print(f"Failed: {failed_count} customers")
        
        if completed_count + failed_count > 0:
            print(f"Success Rate: {(completed_count / (completed_count + failed_count) * 100):.1f}%")

    def save_to_parquet(self, output_file: str, customer_profiles: List[Dict[str, Any]] = None) -> None:
        """Save results to Parquet format (recommended for transcript data)"""
        if not self.analysis_results:
            print("No analysis results to save")
            return
        
        print(f"Saving results to {output_file}...")
        
        # Prepare data for DataFrame - exclude invalidated leads
        rows = []
        invalidated_rows = []
        
        for result in self.analysis_results:
            # Skip invalidated leads - they go to separate files
            if bool(result.get('invalidated', False)):
                continue
            
            # Extract referenced transcript text and calculate length
            referenced_texts = []
            total_length = 0
            
            # Get call details from the first referenced transcript (primary call that triggered the second chance lead)
            primary_call_details = {}
            referenced_transcripts = result.get('referenced_transcripts', [])
            if referenced_transcripts:
                primary_call_id = referenced_transcripts[0].get('call_id', '')
                primary_call_details = self._get_call_details_for_call_id(primary_call_id, customer_profiles)
            
            for transcript_ref in result.get('referenced_transcripts', []):
                call_id = transcript_ref.get('call_id', '')
                snippet = transcript_ref.get('snippet', '')
                
                # Find the full transcript for this call_id
                full_transcript = self._get_full_transcript_for_call_id(call_id, customer_profiles)
                
                # Use full transcript if available, otherwise use snippet
                transcript_text = full_transcript if full_transcript else snippet
                referenced_texts.append(transcript_text)
                total_length += len(transcript_text)
            
            # Join all referenced transcripts with separator
            referenced_text_combined = ' | '.join(referenced_texts)
            
            # Get verification data
            verification_data = result.get('verification', {})
            
            # Safely extract all fields with defaults
            customer_call_analysis = result.get('customer_call_analysis', {})
            recommendations = result.get('recommendations', {})
            
            row = {
                'customer_phone': result.get('customer_phone', ''),
                'analysis_timestamp': result.get('analysis_timestamp', ''),
                'is_second_chance_lead': bool(verification_data.get('is_valid_second_chance_lead', result.get('is_second_chance_lead', False))),
                'reasoning': result.get('reasoning', ''),
                'referenced_transcripts': json.dumps(result.get('referenced_transcripts', [])),
                'referenced_transcript_text': referenced_text_combined,
                'referenced_transcript_length': total_length,
                'was_customer_call': bool(customer_call_analysis.get('was_customer_call', False)),
                'was_service_request': bool(customer_call_analysis.get('was_service_request', False)),
                'was_booked': bool(customer_call_analysis.get('was_booked', False)),
                'booking_failure_reason': customer_call_analysis.get('booking_failure_reason', ''),
                'conversion_potential': customer_call_analysis.get('conversion_potential', ''),
                'immediate_action': recommendations.get('immediate_action', ''),
                'follow_up_strategy': recommendations.get('follow_up_strategy', ''),
                'agent_training_notes': recommendations.get('agent_training_notes', ''),
                'new_job_activity_detected': bool(verification_data.get('new_job_activity_detected', False)),
                'invalidating_job_data': json.dumps(verification_data.get('invalidating_job')) if verification_data.get('invalidating_job') else None,
                'call_id': str(result.get('call_id', '') or ''),
                'primary_call_direction': primary_call_details.get('direction', ''),
                'primary_call_date': primary_call_details.get('call_date', ''),
                'primary_call_time': primary_call_details.get('call_time', ''),
                'primary_call_agent_name': primary_call_details.get('agent_name', '')
            }
            rows.append(row)
        
        # Create DataFrame and save as Parquet
        df = pd.DataFrame(rows)
        df.to_parquet(output_file, index=False, engine='pyarrow')
        
        print(f"Parquet report saved to {output_file} ({len(rows)} valid leads)")

    def save_to_csv(self, output_file: str, customer_profiles: List[Dict[str, Any]] = None) -> None:
        """Save results to CSV format (legacy support with enhanced escaping)"""
        if not self.analysis_results:
            print("No analysis results to save")
            return
        
        print(f"Saving results to {output_file}...")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'customer_phone',
                'analysis_timestamp',
                'is_second_chance_lead',
                'reasoning',
                'referenced_transcripts',
                'referenced_transcript_text',
                'referenced_transcript_length',
                'was_customer_call',
                'was_service_request',
                'was_booked',
                'booking_failure_reason',
                'conversion_potential',
                'immediate_action',
                'follow_up_strategy',
                'agent_training_notes',
                'new_job_activity_detected',
                'invalidating_job_data',
                'primary_call_direction',
                'primary_call_date',
                'primary_call_time',
                'primary_call_agent_name'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, escapechar='\\')
            writer.writeheader()
            
            valid_count = 0
            for result in self.analysis_results:
                # Skip invalidated leads - they go to separate files
                if bool(result.get('invalidated', False)):
                    continue
                
                valid_count += 1
                # Extract referenced transcript text and calculate length
                referenced_texts = []
                total_length = 0
                
                # Get call details from the first referenced transcript (primary call that triggered the second chance lead)
                primary_call_details = {}
                referenced_transcripts = result.get('referenced_transcripts', [])
                if referenced_transcripts:
                    primary_call_id = referenced_transcripts[0].get('call_id', '')
                    primary_call_details = self._get_call_details_for_call_id(primary_call_id, customer_profiles)
                
                for transcript_ref in referenced_transcripts:
                    call_id = transcript_ref.get('call_id', '')
                    snippet = transcript_ref.get('snippet', '')
                    
                    # Find the full transcript for this call_id
                    full_transcript = self._get_full_transcript_for_call_id(call_id, customer_profiles)
                    
                    # Use full transcript if available, otherwise use snippet
                    transcript_text = full_transcript if full_transcript else snippet
                    referenced_texts.append(transcript_text)
                    total_length += len(transcript_text)
                
                # Join all referenced transcripts with separator
                referenced_text_combined = ' | '.join(referenced_texts)
                
                # Get verification data
                verification_data = result.get('verification', {})
                
                # Safely extract all fields with defaults
                customer_call_analysis = result.get('customer_call_analysis', {})
                recommendations = result.get('recommendations', {})
                
                row = {
                    'customer_phone': result.get('customer_phone', ''),
                    'analysis_timestamp': result.get('analysis_timestamp', ''),
                    'is_second_chance_lead': str(verification_data.get('is_valid_second_chance_lead', result.get('is_second_chance_lead', False))),
                    'reasoning': clean_transcript_text(result.get('reasoning', '')),
                    'referenced_transcripts': json.dumps(result.get('referenced_transcripts', [])),
                    'referenced_transcript_text': clean_transcript_text(referenced_text_combined),
                    'referenced_transcript_length': total_length,
                    'was_customer_call': str(customer_call_analysis.get('was_customer_call', False)),
                    'was_service_request': str(customer_call_analysis.get('was_service_request', False)),
                    'was_booked': str(customer_call_analysis.get('was_booked', False)),
                    'booking_failure_reason': clean_transcript_text(customer_call_analysis.get('booking_failure_reason', '')),
                    'conversion_potential': customer_call_analysis.get('conversion_potential', ''),
                    'immediate_action': clean_transcript_text(recommendations.get('immediate_action', '')),
                    'follow_up_strategy': clean_transcript_text(recommendations.get('follow_up_strategy', '')),
                    'agent_training_notes': clean_transcript_text(recommendations.get('agent_training_notes', '')),
                    'new_job_activity_detected': str(verification_data.get('new_job_activity_detected', False)),
                    'invalidating_job_data': json.dumps(verification_data.get('invalidating_job')) if verification_data.get('invalidating_job') else None,
                    # Add call details from the primary call that triggered the second chance lead
                    'primary_call_direction': primary_call_details.get('direction', ''),
                    'primary_call_date': primary_call_details.get('call_date', ''),
                    'primary_call_time': primary_call_details.get('call_time', ''),
                    'primary_call_agent_name': primary_call_details.get('agent_name', '')
                }
                writer.writerow(row)
        
        print(f"CSV report saved to {output_file} ({valid_count} valid leads)")


def format_transcript_for_llm(transcript: str) -> List[Dict[str, str]]:
    lines = transcript.strip().split('\n')
    formatted_conversation = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('Agent:'):
            formatted_conversation.append({
                'role': 'agent',
                'message': line[7:].strip()
            })
        elif line.startswith('Contact:'):
            formatted_conversation.append({
                'role': 'customer',
                'message': line[9:].strip()
            })
    
    return formatted_conversation


def create_customer_call_profiles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    customer_profiles = []
    
    for contact_phone, group in df.groupby('Contact Phone'):
        customer_calls = []
        
        for _, row in group.iterrows():
            # Generate deterministic call ID
            call_id = generate_call_id(
                str(contact_phone),
                row['Call Date'],
                row['Call Time'],
                row['User'],
                row['Transcription']
            )
            
            # All transcripts should already be >= 250 characters from earlier filtering
            call_data = {
                'call_id': call_id,
                'activity_name': row['Talkdesk Activity: Talkdesk Activity Name'],
                'direction': row['Direction'],
                'talk_time_seconds': int(row['Talk Time (sec)']),
                'start_time': row['Start Time'],
                'call_date': row['Call Date'],
                'call_time': row['Call Time'],
                'agent_name': row['User'],
                'transcript': format_transcript_for_llm(row['Transcription'])
            }
            customer_calls.append(call_data)
        
        if customer_calls:  # Only create profile if there are valid calls
            customer_profile = {
                'customer_phone': str(contact_phone),
                'total_calls': len(customer_calls),
                'invalid_calls': 0,  # No longer tracking invalid calls since filtering happens earlier
                'calls': customer_calls,
                'summary': {
                    'total_talk_time_seconds': sum(call['talk_time_seconds'] for call in customer_calls),
                    'call_directions': list(set(call['direction'] for call in customer_calls)),
                    'agents_involved': list(set(call['agent_name'] for call in customer_calls)),
                    'date_range': {
                        'first_call': min(call['call_date'] for call in customer_calls),
                        'last_call': max(call['call_date'] for call in customer_calls)
                    }
                }
            }
            
            customer_profiles.append(customer_profile)
    
    return customer_profiles


def generate_call_id(customer_phone: str, call_date: str, call_time: str, agent_name: str, transcription: str) -> str:
    """Generate a deterministic call ID based on customer phone and call data"""
    # Create a unique identifier from key call attributes
    call_data = f"{customer_phone}|{call_date}|{call_time}|{agent_name}|{transcription[:100]}"
    
    # Generate SHA-256 hash for uniqueness
    call_hash = hashlib.sha256(call_data.encode('utf-8')).hexdigest()[:16]
    
    # Format as readable ID: phone_last4_hash
    phone_last4 = customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone
    return f"{phone_last4}_{call_hash}"


def load_processed_customers() -> Dict[str, Dict[str, Any]]:
    """Load previously processed customers and their call IDs"""
    tracking_file = SECOND_CHANCE_DATA_DIR / PROCESSING_TRACKING_FILE
    
    if not tracking_file.exists():
        return {}
    
    processed_customers = {}
    
    try:
        with open(tracking_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    customer_phone = record['customer_phone']
                    processed_customers[customer_phone] = record
    except Exception as e:
        print(f"Error loading processed customers: {e}")
        return {}
    
    return processed_customers


def load_existing_customer_profiles() -> Dict[str, Dict[str, Any]]:
    """Load existing customer profiles from the persistent file"""
    profiles_file = SECOND_CHANCE_DATA_DIR / "customer_call_profiles.json"
    
    if not profiles_file.exists():
        return {}
    
    try:
        with open(profiles_file, 'r', encoding='utf-8') as f:
            profiles_data = json.load(f)
            # Convert list to dict keyed by customer phone
            profiles_dict = {}
            for profile in profiles_data:
                phone = profile.get('customer_phone')
                if phone:
                    profiles_dict[phone] = profile
            return profiles_dict
    except Exception as e:
        print(f"Error loading existing customer profiles: {e}")
        return {}


def save_customer_profiles(customer_profiles: List[Dict[str, Any]]) -> None:
    """Save customer profiles to the persistent file"""
    profiles_file = SECOND_CHANCE_DATA_DIR / "customer_call_profiles.json"
    profiles_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(profiles_file, 'w', encoding='utf-8') as f:
            json.dump(customer_profiles, f, indent=2, ensure_ascii=False)
        print(f"Customer profiles saved to: {profiles_file}")
    except Exception as e:
        print(f"Error saving customer profiles: {e}")


def merge_customer_profiles(existing_profiles: Dict[str, Dict[str, Any]], new_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge new customer profiles with existing ones"""
    merged_profiles = []
    
    # Start with existing profiles
    for phone, profile in existing_profiles.items():
        merged_profiles.append(profile)
    
    # Add or update with new profiles
    for new_profile in new_profiles:
        phone = new_profile.get('customer_phone')
        if phone in existing_profiles:
            # Update existing profile with new calls
            existing_profile = existing_profiles[phone]
            existing_calls = existing_profile.get('calls', [])
            new_calls = new_profile.get('calls', [])
            
            # Add only new calls (based on call_id)
            existing_call_ids = {call.get('call_id') for call in existing_calls}
            for new_call in new_calls:
                if new_call.get('call_id') not in existing_call_ids:
                    existing_calls.append(new_call)
            
            # Update the profile
            existing_profile['calls'] = existing_calls
            existing_profile['total_calls'] = len(existing_calls)
            existing_profile['summary']['total_talk_time_seconds'] = sum(call['talk_time_seconds'] for call in existing_calls)
            existing_profile['summary']['agents_involved'] = list(set(call['agent_name'] for call in existing_calls))
            existing_profile['summary']['date_range']['first_call'] = min(call['call_date'] for call in existing_calls)
            existing_profile['summary']['date_range']['last_call'] = max(call['call_date'] for call in existing_calls)
            
            # Update the merged list
            for i, merged_profile in enumerate(merged_profiles):
                if merged_profile.get('customer_phone') == phone:
                    merged_profiles[i] = existing_profile
                    break
        else:
            # Add new profile
            merged_profiles.append(new_profile)
    
    return merged_profiles


def save_processed_customer(customer_phone: str, call_ids: Set[str], analysis_timestamp: str) -> None:
    """Save processed customer information to tracking file"""
    tracking_file = SECOND_CHANCE_DATA_DIR / PROCESSING_TRACKING_FILE
    
    # Load existing processed customers
    processed_customers = load_processed_customers()
    
    # Update or add customer record
    if customer_phone in processed_customers:
        # Merge new call IDs with existing ones
        existing_call_ids = set(processed_customers[customer_phone]['processed_call_ids'])
        merged_call_ids = existing_call_ids.union(call_ids)
        processed_customers[customer_phone] = {
            'customer_phone': customer_phone,
            'processed_call_ids': list(merged_call_ids),
            'last_processed': analysis_timestamp,
            'total_calls_processed': len(merged_call_ids)
        }
    else:
        # Add new customer
        processed_customers[customer_phone] = {
            'customer_phone': customer_phone,
            'processed_call_ids': list(call_ids),
            'last_processed': analysis_timestamp,
            'total_calls_processed': len(call_ids)
        }
    
    # Save updated processed customers
    try:
        with open(tracking_file, 'w', encoding='utf-8') as f:
            for customer_data in processed_customers.values():
                f.write(json.dumps(customer_data) + '\n')
    except Exception as e:
        print(f"Error saving processed customer: {e}")


def log_analysis_failure(customer_phone: str, error_type: str, error_message: str, call_ids: List[str], analysis_timestamp: str) -> None:
    """Log analysis failures for debugging"""
    failure_file = SECOND_CHANCE_DATA_DIR / FAILURE_LOG_FILE
    
    record = {
        'customer_phone': customer_phone,
        'error_type': error_type,
        'error_message': error_message,
        'call_ids': call_ids,
        'failure_timestamp': analysis_timestamp
    }
    
    try:
        with open(failure_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        print(f"Error logging failure: {e}")


def get_new_calls_for_customer(customer_phone: str, customer_calls: List[Dict[str, Any]], processed_customers: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out calls that have already been processed for a customer"""
    if customer_phone not in processed_customers:
        return customer_calls
    
    processed_call_ids = set(processed_customers[customer_phone].get('processed_call_ids', []))
    new_calls = []
    
    for call in customer_calls:
        if call['call_id'] not in processed_call_ids:
            new_calls.append(call)
    
    return new_calls


def load_calls_data() -> pd.DataFrame:
    calls_path = PROJECT_ROOT / "companies" / "McCullough Heating and Air" / "parquet" / "Calls.parquet"
    
    if not calls_path.exists():
        print(f"Warning: Calls.parquet not found at {calls_path}")
        return pd.DataFrame()
    
    try:
        calls_df = pd.read_parquet(calls_path)
        print(f"Loaded {len(calls_df)} calls from Calls.parquet")
        return calls_df
    except Exception as e:
        print(f"Error loading Calls.parquet: {e}")
        return pd.DataFrame()


def load_jobs_data() -> pd.DataFrame:
    """Load Jobs.parquet data for second chance lead verification"""
    jobs_path = PROJECT_ROOT / "companies" / "McCullough Heating and Air" / "parquet" / "Jobs.parquet"
    
    if not jobs_path.exists():
        print(f"Warning: Jobs.parquet not found at {jobs_path}")
        return pd.DataFrame()
    
    try:
        jobs_df = pd.read_parquet(jobs_path)
        print(f"Loaded {len(jobs_df)} jobs from Jobs.parquet")
        return jobs_df
    except Exception as e:
        print(f"Error loading Jobs.parquet: {e}")
        return pd.DataFrame()


def normalize_phone_for_matching(phone: str) -> str:
    """Normalize phone number for matching (remove all non-digits)"""
    if pd.isna(phone):
        return ""
    return re.sub(r'\D', '', str(phone))


def find_matching_jobs(customer_phone: str, transcription_date: str, jobs_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Find jobs that match customer phone and have created date >= transcription date"""
    if jobs_df.empty:
        return []
    
    normalized_customer_phone = normalize_phone_for_matching(customer_phone)
    matching_jobs = []
    
    for _, job in jobs_df.iterrows():
        # Handle Customer Phone column - can be single phone or comma-separated list
        customer_phones_str = str(job.get('Customer Phone', ''))
        if pd.isna(job.get('Customer Phone')):
            continue
            
        # Split by comma and normalize each phone number
        job_phones = [normalize_phone_for_matching(phone.strip()) for phone in customer_phones_str.split(',')]
        
        # Check if customer phone matches any job phone
        if normalized_customer_phone in job_phones:
            # Check if job created date is >= transcription date
            job_created_date = job.get('Created Date', '')
            if pd.notna(job_created_date):
                # Convert both dates to datetime for proper comparison
                try:
                    # Parse transcription date (format: M/D/YYYY)
                    from datetime import datetime
                    transcription_dt = datetime.strptime(transcription_date, '%m/%d/%Y')
                    
                    # Parse job date (format: YYYY-MM-DD HH:MM:SS)
                    if isinstance(job_created_date, str):
                        job_dt = datetime.strptime(job_created_date.split(' ')[0], '%Y-%m-%d')
                    else:
                        job_dt = job_created_date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
                    
                    if job_dt >= transcription_dt:
                        matching_jobs.append(job.to_dict())
                except (ValueError, AttributeError) as e:
                    # If date parsing fails, skip this job
                    continue
    
    return matching_jobs


def verify_second_chance_lead(customer_phone: str, transcription_date: str, jobs_df: pd.DataFrame) -> Dict[str, Any]:
    """Verify if a second chance lead should be invalidated due to subsequent job creation"""
    matching_jobs = find_matching_jobs(customer_phone, transcription_date, jobs_df)
    
    if matching_jobs:
        # Use the most recent job for verification
        latest_job = max(matching_jobs, key=lambda x: str(x.get('Created Date', '')))
        return {
            'is_valid_second_chance_lead': False,
            'new_job_activity_detected': True,
            'invalidating_job': latest_job,
            'total_matching_jobs': len(matching_jobs)
        }
    else:
        return {
            'is_valid_second_chance_lead': True,
            'new_job_activity_detected': False,
            'invalidating_job': None,
            'total_matching_jobs': 0
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet-id", default="1AwSeb7E1x0kN4j5czsuGUf8nSUADayxYVyiJ9bvXbaE")
    parser.add_argument("--gid")
    parser.add_argument("--credentials", default=str(DEFAULT_CREDENTIALS))
    parser.add_argument("--token", default=str(DEFAULT_TOKEN))
    parser.add_argument("--output")
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE_DIR))
    parser.add_argument("--interval-minutes", type=int, default=30)
    parser.add_argument("--keep-files", type=int, default=6)
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--test-limit", type=int, help="Limit number of customer profiles to analyze for testing (prevents excessive API usage)")
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
    
    print("Starting Second Chance Lead Detection Process")
    print("=" * 60)
    
    # Download Google Sheet data once per execution
    print("\nStep 1: Downloading Google Sheet data...")
    creds = load_credentials(credentials_path, token_path)
    output_path = resolve_output_path(args.output)
    saved_path = download_csv(args.sheet_id, creds, output_path, args.gid)
    print(f"Saved CSV to {saved_path}")
    
    # Check if this is a duplicate download
    if is_duplicate_download(saved_path):
        print("Duplicate download detected - skipping entire process")
        print("Execution skipped due to duplicate data")
        return
    
    # Archive old downloaded files since this is a new download
    archive_old_downloaded_files(saved_path)
    
    # Backfill processed customers if needed
    backfill_processed_customers_from_persistent_files()
    
    while True:
        iteration += 1
        started = datetime.now().isoformat()
        print(f"\nIteration {iteration} started at {started}")
        
        try:
            
            # Step 2: Process and clean transcript data
            print("\nStep 2: Processing and cleaning transcript data...")
            clean_csv_file(saved_path)
            
            # Step 3: Load processed data and create customer profiles
            print("\nStep 3: Creating customer call profiles...")
            try:
                df = pd.read_csv(saved_path, sep=',', quoting=csv.QUOTE_ALL, quotechar='"', engine='python')
                print(f"Loaded {len(df)} transcript records")
                
                new_customer_profiles = create_customer_call_profiles(df)
                print(f"Created {len(new_customer_profiles)} new customer profiles")
                
                # Load existing profiles and merge with new ones
                existing_profiles = load_existing_customer_profiles()
                customer_profiles = merge_customer_profiles(existing_profiles, new_customer_profiles)
                
                # Save merged customer profiles to persistent file
                save_customer_profiles(customer_profiles)
                
                # Display profile statistics
                total_invalid_calls = sum(profile.get('invalid_calls', 0) for profile in customer_profiles)
                total_valid_calls = sum(profile['total_calls'] for profile in customer_profiles)
                print(f"Statistics:")
                print(f"   - Total valid calls: {total_valid_calls}")
                print(f"   - Total invalid calls (dropped): {total_invalid_calls}")
                print(f"   - New customer profiles: {len(new_customer_profiles)}")
                print(f"   - Total customer profiles: {len(customer_profiles)}")
                
            except Exception as e:
                print(f"Error processing transcript data: {e}")
                raise
            
            # Step 4: Load Calls.parquet and Jobs.parquet data for additional context
            print("\nStep 4: Loading Calls.parquet and Jobs.parquet data...")
            calls_df = load_calls_data()
            jobs_df = load_jobs_data()
            
            # Step 5: AI Analysis for Second Chance Leads
            print("\nStep 5: AI Analysis for Second Chance Leads...")
            if not DEEPSEEK_API_KEY:
                print("Warning: No DeepSeek API key provided. Skipping AI analysis.")
            else:
                # Apply test limit if specified
                profiles_to_analyze = customer_profiles
                if args.test_limit:
                    profiles_to_analyze = customer_profiles[:args.test_limit]
                    print(f"Testing mode: Limited to {len(profiles_to_analyze)} profiles (out of {len(customer_profiles)} total)")
                elif MAX_PROFILES_TO_ANALYZE:
                    profiles_to_analyze = customer_profiles[:MAX_PROFILES_TO_ANALYZE]
                    print(f"Testing mode: Limited to {len(profiles_to_analyze)} profiles (out of {len(customer_profiles)} total)")
                
                analyzer = SecondChanceLeadAnalyzer(jobs_df=jobs_df)
                analyzer.process_all_customers(profiles_to_analyze, concurrent_calls=CONCURRENT_API_CALLS)
                
                # Step 6: Update persistent files
                print("\nStep 6: Updating persistent second chance leads files...")
                
                # Update persistent files with new results
                update_persistent_files(analyzer.analysis_results, customer_profiles)
                
                # Count second chance leads
                second_chance_count = sum(1 for result in analyzer.analysis_results if result.get('is_second_chance_lead'))
                print(f"Found {second_chance_count} second chance leads out of {len(analyzer.analysis_results)} analyzed customers")
                print(f"Persistent files updated in {SECOND_CHANCE_DATA_DIR}")
                
                # Backfill processed customers again after updating persistent files
                print("\nStep 7: Backfilling processed customers after persistent file updates...")
                backfill_processed_customers_from_persistent_files()

                # Step 8: Run recent second chance leads extraction
                try:
                    project_root = Path(__file__).resolve().parent.parent
                    company = "McCullough Heating and Air"
                    data_dir = project_root / "data" / company
                    parquet_dir = project_root / "companies" / company / "parquet"
                    previous_cwd = Path.cwd()
                    print(f"[DownloadScript] Switching CWD from {previous_cwd} to {project_root}", flush=True)
                    try:
                        os.chdir(project_root)
                        extraction_engine = CustomExtractionEngine(company, data_dir, parquet_dir)
                        result = extraction_engine.extract_second_chance_leads()
                        print("Triggered recent second chance leads extraction", flush=True)
                        print(f"Extraction result: success={result.success}, saved={result.records_saved}, message={result.error_message}", flush=True)
                        if not result.success:
                            print(f"Extraction warning: {result.error_message}")
                    finally:
                        print(f"[DownloadScript] Restoring CWD to {previous_cwd}", flush=True)
                        os.chdir(previous_cwd)
                except Exception as extract_error:
                    print(f"WARNING: Failed to run second chance leads extraction: {extract_error}")
                    import traceback as _traceback
                    _traceback.print_exc()
            
            # Archive old files
            history.append(saved_path)
            while keep_files and len(history) > keep_files:
                old_path = history.popleft()
                archive_file(old_path, archive_dir)
                
            print(f"\nIteration {iteration} completed successfully!")
            
        except Exception as exc:
            print(f"Iteration {iteration} error: {exc}")
            import traceback
            traceback.print_exc()
            
        if args.run_once:
            break
            
        print(f"\nSleeping for {interval_minutes} minutes until next iteration...")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    main()
