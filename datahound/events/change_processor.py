"""
Enhanced change processing system for log-driven event detection
"""

from dataclasses import dataclass, asdict
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import pandas as pd
import json
import hashlib

from .types import EventResult, EventSeverity
from ..upsert.types import AuditChange
from .rules_config import EventRulesManager, RemovalCondition


@dataclass
class EventChangeLog:
    """Structured event change log entry"""
    timestamp: datetime
    company: str
    change_type: str  # "job_canceled", "estimate_dismissed", "maintenance_completed", etc.
    customer_id: str
    entity_type: str  # "job", "estimate", "call", etc.
    entity_id: str
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    metadata: Dict[str, Any]  # Additional context like dates, values, etc.


@dataclass
class RecentEvent:
    """Recent event tracking structure"""
    event_id: str
    customer_id: str
    event_type: str
    detected_at: datetime
    entity_id: str
    details: Dict[str, Any]
    status: str = "active"  # active, archived, resolved
    archive_reason: Optional[str] = None


class ChangeProcessor:
    """Processes audit changes into structured event logs"""
    
    def __init__(self, company: str, data_dir: Path):
        self.company = company
        self.data_dir = data_dir
        self.event_changes_log = data_dir / "logs" / "event_changes_log.jsonl"
        
        # Ensure log directory exists
        self.event_changes_log.parent.mkdir(parents=True, exist_ok=True)
    
    def process_audit_changes(self, changes: List[AuditChange], file_type: str, metadata: Dict[str, Any] = None):
        """Process audit changes and extract event-worthy changes"""
        
        event_changes = []
        
        for change in changes:
            # Determine if this change represents an event
            event_type = self._determine_event_type(change, file_type)
            
            if event_type:
                # Extract customer ID
                customer_id = self._extract_customer_id(change, file_type)
                
                if customer_id:
                    event_change = EventChangeLog(
                        timestamp=datetime.now(UTC),
                        company=self.company,
                        change_type=event_type,
                        customer_id=customer_id,
                        entity_type=file_type.rstrip('s'),  # "jobs" -> "job"
                        entity_id=str(change.id_value),  # Use id_value instead of record_id
                        old_values={change.column: change.old_value},
                        new_values={change.column: change.new_value},
                        metadata=metadata or {}
                    )
                    
                    event_changes.append(event_change)
        
        # Save event changes to log
        if event_changes:
            self._save_event_changes(event_changes)
        
        return event_changes
    
    def _determine_event_type(self, change: AuditChange, file_type: str) -> Optional[str]:
        """Determine if change represents an event worth tracking"""
        
        if file_type == "jobs":
            if change.column.lower() in ["status"]:  # Actual column name is "Status"
                old_status = str(change.old_value).lower()
                new_status = str(change.new_value).lower()
                
                if new_status in ["canceled", "cancelled"]:
                    return "job_canceled"
                elif new_status in ["completed", "finished", "complete"] and old_status not in ["completed", "finished", "complete"]:
                    return "job_completed"
                elif new_status in ["scheduled", "in progress"] and old_status in ["canceled", "cancelled"]:
                    return "job_rescheduled"
        
        elif file_type == "estimates":
            if change.column.lower() in ["estimate status"]:  # Actual column name is "Estimate Status"
                new_status = str(change.new_value).lower()
                
                if new_status in ["dismissed", "declined", "rejected"]:
                    return "estimate_dismissed"
                elif new_status in ["sold", "accepted", "approved"]:
                    return "estimate_sold"
        
        elif file_type == "calls":
            if change.column.lower() in ["call type", "type", "reason"]:
                return "customer_contact"
        
        # Add more event type detection logic as needed
        return None
    
    def _extract_customer_id(self, change: AuditChange, file_type: str) -> Optional[str]:
        """Extract customer ID from change context by looking up the record"""
        
        try:
            # Load the master data for this file type
            # Try both company name formats
            possible_paths = [
                Path(f"companies/{self.company}/parquet/{file_type.capitalize()}.parquet"),
                Path(f"companies/McCullough Heating and Air/parquet/{file_type.capitalize()}.parquet"),
                Path(f"companies/mccullough_hvac/parquet/{file_type.capitalize()}.parquet")
            ]
            
            master_file = None
            for path in possible_paths:
                if path.exists():
                    master_file = path
                    break
            
            if not master_file:
                return None
            
            # Load the data
            df = pd.read_parquet(master_file)
            
            # Find the record using the ID
            id_column = f"{file_type.rstrip('s').capitalize()} ID"  # "jobs" -> "Job ID"
            
            # Try different ID column variations
            possible_id_columns = [
                id_column,
                f"{file_type.rstrip('s')} ID",  # "jobs" -> "job ID" 
                "ID",
                df.columns[0]  # First column is often the ID
            ]
            
            actual_id_column = None
            for col in possible_id_columns:
                if col in df.columns:
                    actual_id_column = col
                    break
            
            if not actual_id_column:
                return None
            
            # Find the record
            record_row = df[df[actual_id_column].astype(str) == str(change.id_value)]
            
            if record_row.empty:
                return None
            
            # Find customer ID column
            customer_columns = [col for col in df.columns if 'customer' in col.lower() and 'id' in col.lower()]
            
            if not customer_columns:
                return None
            
            customer_id = record_row.iloc[0][customer_columns[0]]
            return str(customer_id) if pd.notna(customer_id) else None
            
        except Exception as e:
            # Silently handle errors to avoid breaking main process
            return None
    
    def _save_event_changes(self, event_changes: List[EventChangeLog]):
        """Save event changes to JSONL log file"""
        
        try:
            with open(self.event_changes_log, 'a', encoding='utf-8') as f:
                for event_change in event_changes:
                    # Convert to dict and handle datetime serialization
                    event_dict = asdict(event_change)
                    event_dict['timestamp'] = event_change.timestamp.isoformat()
                    
                    f.write(json.dumps(event_dict) + '\n')
        
        except Exception as e:
            # Log error but don't break the main process
            print(f"Error saving event changes: {e}")


class RecentEventsManager:
    """Manages recent events with automatic archiving"""
    
    def __init__(self, company: str, data_dir: Path):
        self.company = company
        self.data_dir = data_dir
        self.recent_events_dir = data_dir / "recent_events"
        self.archives_dir = self.recent_events_dir / "archives"
        
        # Create directories
        self.recent_events_dir.mkdir(parents=True, exist_ok=True)
        self.archives_dir.mkdir(parents=True, exist_ok=True)
        
        # Load master data for lookups
        self.parquet_dir = data_dir.parent / "parquet"
        self._master_data_cache = {}
        
        # Initialize rules manager
        config_dir = Path("config/events")
        self.rules_manager = EventRulesManager(company, config_dir)
    
    def process_change_logs(self, since: Optional[datetime] = None):
        """Process change logs and update recent events"""
        
        # Load recent change logs
        changes = self._load_recent_changes(since)
        
        for change in changes:
            if change['change_type'] == 'job_canceled':
                self._process_job_cancellation(change)
            elif change['change_type'] == 'job_completed':
                self._process_job_completion(change)
            elif change['change_type'] == 'job_rescheduled':
                self._process_job_rescheduled(change)
            elif change['change_type'] == 'estimate_dismissed':
                self._process_unsold_estimate(change)
            elif change['change_type'] == 'estimate_sold':
                self._process_estimate_sold(change)
            elif change['change_type'] == 'customer_contact':
                self._process_customer_contact(change)
        
        # Check for overdue maintenance based on job completion dates
        self._check_overdue_maintenance()
        
        # Run archiving for all event types
        self._archive_old_events()
    
    def _load_recent_changes(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Load recent change logs"""
        
        event_changes_log = self.data_dir / "logs" / "event_changes_log.jsonl"
        
        if not event_changes_log.exists():
            return []
        
        changes = []
        cutoff_time = since or (datetime.now(UTC) - timedelta(hours=24))  # Default: last 24 hours
        
        try:
            with open(event_changes_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        change = json.loads(line.strip())
                        change_time = datetime.fromisoformat(change['timestamp'].replace('Z', '+00:00'))
                        
                        if change_time >= cutoff_time:
                            changes.append(change)
                    except Exception:
                        continue  # Skip malformed lines
        except Exception:
            pass  # File doesn't exist or can't be read
        
        return changes
    
    def _process_job_cancellation(self, change: Dict[str, Any]):
        """Process job cancellation and add to recent events"""
        
        recent_cancellations = self._load_recent_events("cancellations")
        
        # Check if this cancellation already exists
        existing = recent_cancellations[
            (recent_cancellations['entity_id'] == change['entity_id']) &
            (recent_cancellations['customer_id'] == change['customer_id'])
        ]
        
        if existing.empty:
            # Get job details for metadata
            job_details = self._get_job_details(change['entity_id'])
            
            # Add new cancellation
            new_event = {
                'event_id': self._generate_event_id(change),
                'customer_id': change['customer_id'],
                'event_type': 'job_canceled',
                'detected_at': change['timestamp'],
                'entity_id': change['entity_id'],
                'details': {
                    **change['metadata'],
                    **job_details,
                    'cancellation_reason': change['new_values'].get('status', 'canceled')
                },
                'status': 'active',
                'archive_reason': None
            }
            
            # Add to dataframe and save
            new_df = pd.DataFrame([new_event])
            updated_df = pd.concat([recent_cancellations, new_df], ignore_index=True)
            self._save_recent_events("cancellations", updated_df)
    
    def _process_job_completion(self, change: Dict[str, Any]):
        """Process job completion - remove related cancellations"""
        
        recent_cancellations = self._load_recent_events("cancellations")
        
        # Find cancellations for this job/customer
        to_archive = recent_cancellations[
            (recent_cancellations['entity_id'] == change['entity_id']) |
            (recent_cancellations['customer_id'] == change['customer_id'])
        ]
        
        if not to_archive.empty:
            # Archive these cancellations
            for _, event in to_archive.iterrows():
                self._archive_event(event, "job_completed")
            
            # Remove from active cancellations
            remaining = recent_cancellations[~recent_cancellations.index.isin(to_archive.index)]
            self._save_recent_events("cancellations", remaining)
    
    def _process_job_rescheduled(self, change: Dict[str, Any]):
        """Process job reschedule - remove related cancellations"""
        
        recent_cancellations = self._load_recent_events("cancellations")
        
        # Find cancellations for this job
        to_archive = recent_cancellations[
            recent_cancellations['entity_id'] == change['entity_id']
        ]
        
        if not to_archive.empty:
            # Archive these cancellations
            for _, event in to_archive.iterrows():
                self._archive_event(event, "job_rescheduled")
            
            # Remove from active cancellations
            remaining = recent_cancellations[~recent_cancellations.index.isin(to_archive.index)]
            self._save_recent_events("cancellations", remaining)
    
    def _process_unsold_estimate(self, change: Dict[str, Any]):
        """Process dismissed estimate and add to recent events"""
        
        recent_estimates = self._load_recent_events("unsold_estimates")
        
        # Check if this estimate already exists
        existing = recent_estimates[
            (recent_estimates['entity_id'] == change['entity_id']) &
            (recent_estimates['customer_id'] == change['customer_id'])
        ]
        
        if existing.empty:
            # Get estimate details
            estimate_details = self._get_estimate_details(change['entity_id'])
            
            # Add new unsold estimate
            new_event = {
                'event_id': self._generate_event_id(change),
                'customer_id': change['customer_id'],
                'event_type': 'estimate_dismissed',
                'detected_at': change['timestamp'],
                'entity_id': change['entity_id'],
                'details': {
                    **change['metadata'],
                    **estimate_details,
                    'dismiss_reason': change['new_values'].get('estimate_status', 'dismissed')
                },
                'status': 'active',
                'archive_reason': None
            }
            
            # Add to dataframe and save
            new_df = pd.DataFrame([new_event])
            updated_df = pd.concat([recent_estimates, new_df], ignore_index=True)
            self._save_recent_events("unsold_estimates", updated_df)
    
    def _process_estimate_sold(self, change: Dict[str, Any]):
        """Process sold estimate - remove from unsold estimates"""
        
        recent_estimates = self._load_recent_events("unsold_estimates")
        
        # Find this estimate in unsold list
        to_archive = recent_estimates[
            recent_estimates['entity_id'] == change['entity_id']
        ]
        
        if not to_archive.empty:
            # Archive the estimate
            for _, event in to_archive.iterrows():
                self._archive_event(event, "estimate_sold")
            
            # Remove from active unsold estimates
            remaining = recent_estimates[~recent_estimates.index.isin(to_archive.index)]
            self._save_recent_events("unsold_estimates", remaining)
    
    def _process_customer_contact(self, change: Dict[str, Any]):
        """Process customer contact - could affect lost customer status"""
        
        recent_lost = self._load_recent_events("lost_customers")
        
        # If this customer was marked as lost, consider removing them
        to_archive = recent_lost[
            recent_lost['customer_id'] == change['customer_id']
        ]
        
        if not to_archive.empty:
            # Archive lost customer status due to recent contact
            for _, event in to_archive.iterrows():
                self._archive_event(event, "customer_contact")
            
            # Remove from lost customers
            remaining = recent_lost[~recent_lost.index.isin(to_archive.index)]
            self._save_recent_events("lost_customers", remaining)
    
    def _check_overdue_maintenance(self):
        """Check for overdue maintenance and update recent events"""
        
        try:
            # Load jobs data to find last maintenance dates
            jobs_df = self._load_master_data("jobs")
            if jobs_df is None:
                return
            
            # Find maintenance jobs
            maintenance_jobs = jobs_df[
                jobs_df['Job Type'].str.contains('maintenance|service|repair', case=False, na=False)
            ]
            
            # Group by customer and find last maintenance
            current_time = datetime.now(UTC)
            recent_maintenance = self._load_recent_events("overdue_maintenance")
            
            for customer_id in maintenance_jobs['Customer ID'].unique():
                customer_jobs = maintenance_jobs[maintenance_jobs['Customer ID'] == customer_id]
                
                # Find most recent maintenance
                if 'Completion Date' in customer_jobs.columns:
                    last_dates = pd.to_datetime(customer_jobs['Completion Date'], errors='coerce')
                    last_maintenance = last_dates.max()
                else:
                    last_dates = pd.to_datetime(customer_jobs['Created Date'], errors='coerce')
                    last_maintenance = last_dates.max()
                
                if pd.notna(last_maintenance):
                    months_since = (current_time - last_maintenance.tz_localize(UTC)).days / 30.44
                    
                    # If overdue (>12 months), add to recent events
                    if months_since > 12:
                        existing = recent_maintenance[
                            recent_maintenance['customer_id'] == str(customer_id)
                        ]
                        
                        if existing.empty:
                            # Add new overdue maintenance event
                            new_event = {
                                'event_id': hashlib.md5(f"maintenance_{customer_id}_{current_time}".encode()).hexdigest()[:16],
                                'customer_id': str(customer_id),
                                'event_type': 'overdue_maintenance',
                                'detected_at': current_time.isoformat(),
                                'entity_id': str(customer_id),
                                'details': {
                                    'months_overdue': round(months_since, 1),
                                    'last_maintenance_date': last_maintenance.strftime('%Y-%m-%d'),
                                    'maintenance_type': 'general'
                                },
                                'status': 'active',
                                'archive_reason': None
                            }
                            
                            new_df = pd.DataFrame([new_event])
                            updated_df = pd.concat([recent_maintenance, new_df], ignore_index=True)
                            self._save_recent_events("overdue_maintenance", updated_df)
        
        except Exception as e:
            print(f"Error checking overdue maintenance: {e}")
    
    def _archive_old_events(self):
        """Archive events based on configurable rules"""
        
        current_time = datetime.now(UTC)
        
        # Get removal rules for each event type
        event_types = ["cancellations", "overdue_maintenance", "unsold_estimates", "lost_customers"]
        
        for event_type in event_types:
            removal_rules = self.rules_manager.get_removal_rules_by_event_type(event_type)
            
            # Sort rules by order for proper evaluation
            removal_rules.sort(key=lambda x: x.order)
            
            # Apply each removal rule
            for rule in removal_rules:
                self._apply_removal_rule(event_type, rule, current_time)
    
    def _apply_removal_rule(self, event_type: str, rule, current_time: datetime):
        """Apply a specific removal rule to events"""
        
        try:
            recent_events = self._load_recent_events(event_type)
            if recent_events.empty:
                return
            
            to_archive = []
            
            if rule.removal_condition == RemovalCondition.AGE_BASED:
                # Age-based removal
                max_days = rule.parameters.get('max_days', 30)
                cutoff_time = current_time - timedelta(days=max_days)
                
                old_events = recent_events[
                    pd.to_datetime(recent_events['detected_at']) < cutoff_time
                ]
                
                for _, event in old_events.iterrows():
                    event_dict = event.to_dict()
                    reason = rule.parameters.get('archive_reason', f'aged_out_{max_days}_days')
                    event_dict['archive_reason'] = reason
                    event_dict['invalidation_reason'] = reason
                    to_archive.append(event_dict)
            
            elif rule.removal_condition == RemovalCondition.SUBSEQUENT_ACTIVITY:
                # Subsequent activity removal
                activity_types = rule.parameters.get('activity_types', [rule.parameters.get('activity_type')])
                match_customer = rule.parameters.get('match_customer', True)
                match_entity = rule.parameters.get('match_entity', False)
                
                for _, event in recent_events.iterrows():
                    customer_id = event['customer_id']
                    entity_id = event['entity_id']
                    detected_at = event['detected_at']
                    
                    # Check for subsequent activity
                    has_activity = False
                    for activity_type in activity_types:
                        if activity_type == 'job_completed':
                            if self._has_subsequent_job_activity(customer_id, detected_at):
                                has_activity = True
                                break
                        elif activity_type == 'maintenance_completed':
                            if self._customer_completed_maintenance_recently(customer_id):
                                has_activity = True
                                break
                        elif activity_type == 'customer_contact':
                            if self._has_recent_customer_contact(customer_id, detected_at):
                                has_activity = True
                                break
                    
                    if has_activity:
                        event_dict = event.to_dict()
                        reason = rule.parameters.get('archive_reason', 'subsequent_activity')
                        event_dict['archive_reason'] = reason
                        event_dict['invalidation_reason'] = event_dict.get('invalidation_reason', reason)
                        to_archive.append(event_dict)
            
            elif rule.removal_condition == RemovalCondition.STATUS_RESOLUTION:
                # Status resolution removal
                resolution_event = rule.parameters.get('resolution_event')
                match_entity = rule.parameters.get('match_entity', True)
                
                # This would require tracking resolution events - for now, skip
                # Could be enhanced to check for specific status changes
                pass
            elif rule.removal_condition == RemovalCondition.INBOUND_CALL_ACTIVITY:
                matched_calls = self._find_inbound_call_matches(recent_events, rule.parameters)
                for _, event in matched_calls.iterrows():
                    event_dict = event.to_dict()
                    reason = rule.parameters.get('archive_reason', 'inbound_call_activity')
                    event_dict['archive_reason'] = reason
                    event_dict['invalidation_reason'] = event_dict.get('invalidation_reason', reason)
                    to_archive.append(event_dict)
            elif rule.removal_condition == RemovalCondition.SMS_ACTIVITY:
                matched_sms = self._find_sms_activity_matches(recent_events, rule.parameters)
                for _, event in matched_sms.iterrows():
                    event_dict = event.to_dict()
                    reason = rule.parameters.get('archive_reason', 'sms_follow_up')
                    event_dict['archive_reason'] = reason
                    event_dict['invalidation_reason'] = event_dict.get('invalidation_reason', reason)
                    to_archive.append(event_dict)
            
            # Archive events that match the rule
            if to_archive:
                archive_df = pd.DataFrame(to_archive)
                archive_df['archived_at'] = current_time.isoformat()
                archive_df['status'] = 'archived'
                
                # Move to archive
                self._move_to_archive_df(event_type, archive_df)
                
                # Remove from recent events
                archived_ids = set(archive_df['event_id'])
                remaining = recent_events[~recent_events['event_id'].isin(archived_ids)]
                self._save_recent_events(event_type, remaining)
        
        except Exception as e:
            print(f"Error applying removal rule {rule.rule_id}: {e}")
    
    def _find_inbound_call_matches(self, recent_events: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        try:
            calls_df = self._load_master_data("calls")
            if calls_df is None or calls_df.empty:
                return pd.DataFrame()
            calls_df = calls_df.copy()
            phone_field = params.get('phone_number_column', 'Phone Number')
            caller_phone_field = params.get('caller_phone_column', 'Caller Phone Number')
            # fallback to customer phone field if phone is missing
            if phone_field not in recent_events.columns or recent_events[phone_field].isna().all():
                alt_phone_field = params.get('customer_phone_column', 'Phone Number_customer_core')
                if alt_phone_field in recent_events.columns:
                    phone_field = alt_phone_field
            direction_field = params.get('direction_column', 'Direction')
            call_date_field = params.get('call_date_column', 'Call Date')
            event_date_field = params.get('detection_field', 'Updated At')
            directions = params.get('directions', ['Inbound'])
            normalize_numbers = params.get('normalize_phone_numbers', True)
            require_match_phone = params.get('match_phone_number', True)
            require_match_customer = params.get('match_customer_id', False)
            customer_field = params.get('customer_id_column', 'Customer ID')
            call_customer_field = params.get('call_customer_column', 'Customer ID')
            calls_df[caller_phone_field] = calls_df[caller_phone_field].astype(str).str.strip()
            recent_events[phone_field] = recent_events[phone_field].astype(str).str.strip()
            calls_df[call_date_field] = pd.to_datetime(calls_df[call_date_field], errors='coerce')
            recent_events[event_date_field] = pd.to_datetime(recent_events[event_date_field], errors='coerce')
            if normalize_numbers:
                calls_df[caller_phone_field] = calls_df[caller_phone_field].str.replace(r"[^0-9]", "", regex=True)
                recent_events[phone_field] = recent_events[phone_field].str.replace(r"[^0-9]", "", regex=True)
            calls_df = calls_df[calls_df[direction_field].isin(directions)]
            filtered_calls = calls_df[[caller_phone_field, call_date_field, call_customer_field]].dropna(subset=[call_date_field])
            if filtered_calls.empty:
                return pd.DataFrame()
            matches = []
            match_count = 0
            for _, event in recent_events.iterrows():
                event_phone = event.get(phone_field)
                if require_match_phone and not event_phone:
                    continue
                candidate_calls = filtered_calls
                if require_match_phone and event_phone:
                    candidate_calls = candidate_calls[candidate_calls[caller_phone_field] == event_phone]
                if require_match_customer and customer_field in event and event.get(customer_field):
                    candidate_calls = candidate_calls[candidate_calls[call_customer_field].astype(str) == str(event.get(customer_field))]
                if len(candidate_calls) == 0:
                    continue
                event_detected_at = event.get(event_date_field)
                if event_detected_at is None or pd.isna(event_detected_at):
                    event_detected_at = event.get('updated_at')
                if event_detected_at is None or pd.isna(event_detected_at):
                    continue
                candidate_calls = candidate_calls[candidate_calls[call_date_field] >= event_detected_at]
                window_days = params.get('window_days')
                if window_days is not None:
                    candidate_calls = candidate_calls[candidate_calls[call_date_field] <= event_detected_at + pd.Timedelta(days=window_days)]
                if len(candidate_calls) == 0:
                    continue
                calls_required = params.get('calls_required', 1)
                if len(candidate_calls) >= calls_required:
                    matches.append(event)
                    match_count += 1
            if not matches:
                return pd.DataFrame()
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_processing_stats(
                    "recent_events",
                    "inbound_call_match",
                    {
                        "event_candidates": len(recent_events),
                        "matches": match_count,
                        "rule_phone_field": phone_field,
                        "call_direction_filter": directions
                    }
                )
            return pd.DataFrame(matches)
        except Exception as e:
            print(f"Error finding inbound call matches: {e}")
            return pd.DataFrame()

    def _find_sms_activity_matches(self, recent_events: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        try:
            sms_dir_param = params.get('sms_export_dir') or params.get('sms_recent_dir')
            if sms_dir_param:
                base_dir = Path(sms_dir_param)
                if not base_dir.is_absolute():
                    base_dir = Path.cwd() / base_dir
            else:
                base_dir = Path.cwd() / "data"
            if not base_dir.exists():
                return pd.DataFrame()
            csv_files = sorted(base_dir.glob("google_sheet_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not csv_files:
                return pd.DataFrame()
            latest_file = csv_files[0]
            sms_df = pd.read_csv(latest_file, dtype=str, keep_default_na=False)
            phone_col = params.get('phone_number_column', 'Phone Number')
            sms_phone_col = params.get('sms_phone_column', 'Case: Secondary Contact Phone')
            sms_date_col = params.get('sms_created_column', 'Created Date/Time')
            event_date_field = params.get('detection_field', 'Updated At')
            if phone_col not in recent_events.columns:
                return pd.DataFrame()
            if sms_phone_col not in sms_df.columns or sms_date_col not in sms_df.columns:
                return pd.DataFrame()
            normalize_numbers = params.get('normalize_phone_numbers', True)
            window_minutes = params.get('window_minutes')
            sms_df = sms_df.copy()
            recent_events = recent_events.copy()
            if normalize_numbers:
                sms_df[sms_phone_col] = sms_df[sms_phone_col].astype(str).str.replace(r"[^0-9]", "", regex=True)
                recent_events[phone_col] = recent_events[phone_col].astype(str).str.replace(r"[^0-9]", "", regex=True)
            sms_df[sms_date_col] = pd.to_datetime(sms_df[sms_date_col], format="%m/%d/%Y %H:%M", errors='coerce')
            recent_events[event_date_field] = pd.to_datetime(recent_events[event_date_field], errors='coerce')
            sms_df = sms_df.dropna(subset=[sms_phone_col, sms_date_col])
            if sms_df.empty:
                return pd.DataFrame()
            matches = []
            for _, event in recent_events.iterrows():
                event_phone = event.get(phone_col)
                if not event_phone:
                    continue
                candidates = sms_df[sms_df[sms_phone_col] == event_phone]
                if candidates.empty:
                    continue
                event_dt = event.get(event_date_field)
                if event_dt is None or pd.isna(event_dt):
                    continue
                if window_minutes is not None:
                    window_end = event_dt + pd.Timedelta(minutes=window_minutes)
                    candidates = candidates[(candidates[sms_date_col] >= event_dt) & (candidates[sms_date_col] <= window_end)]
                else:
                    candidates = candidates[candidates[sms_date_col] >= event_dt]
                if not candidates.empty:
                    matches.append(event)
            if not matches:
                return pd.DataFrame()
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_processing_stats(
                    "recent_events",
                    "sms_activity_match",
                    {
                        "event_candidates": len(recent_events),
                        "matches": len(matches),
                        "sms_file": str(latest_file)
                    }
                )
            return pd.DataFrame(matches)
        except Exception as e:
            print(f"Error finding sms activity matches: {e}")
            return pd.DataFrame()
    
    def _archive_events_by_age(self, event_type: str, max_days: int, current_time: datetime):
        """Archive events older than specified days"""
        
        recent_events = self._load_recent_events(event_type)
        if recent_events.empty:
            return
        
        # Find old events
        cutoff_time = current_time - timedelta(days=max_days)
        
        old_events = recent_events[
            pd.to_datetime(recent_events['detected_at']) < cutoff_time
        ]
        
        if not old_events.empty:
            # Archive old events
            for _, event in old_events.iterrows():
                self._archive_event(event, f"aged_out_{max_days}_days")
            
            # Keep only recent events
            remaining = recent_events[~recent_events.index.isin(old_events.index)]
            self._save_recent_events(event_type, remaining)
    
    def _load_recent_events(self, event_type: str) -> pd.DataFrame:
        """Load recent events of specified type"""
        
        file_path = self.recent_events_dir / f"recent_{event_type}.parquet"
        
        if file_path.exists():
            try:
                return pd.read_parquet(file_path)
            except Exception:
                pass
        
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=[
            'event_id', 'customer_id', 'event_type', 'detected_at', 
            'entity_id', 'details', 'status', 'archive_reason'
        ])
    
    def _save_recent_events(self, event_type: str, df: pd.DataFrame):
        """Save recent events to parquet file"""
        
        file_path = self.recent_events_dir / f"recent_{event_type}.parquet"
        
        try:
            if not df.empty:
                df.to_parquet(file_path, index=False)
            elif file_path.exists():
                # Remove empty file
                file_path.unlink()
        except Exception as e:
            print(f"Error saving recent {event_type}: {e}")
    
    def _archive_event(self, event: pd.Series, reason: str):
        """Archive a single event"""
        
        archive_file = self.archives_dir / f"archived_{event['event_type']}s.parquet"
        
        # Load existing archives
        if archive_file.exists():
            try:
                archived_df = pd.read_parquet(archive_file)
            except Exception:
                archived_df = pd.DataFrame()
        else:
            archived_df = pd.DataFrame()
        
        # Add archive metadata
        event_dict = event.to_dict()
        event_dict['archived_at'] = datetime.now(UTC).isoformat()
        event_dict['archive_reason'] = reason
        event_dict['status'] = 'archived'
        
        # Append to archives
        new_archive = pd.DataFrame([event_dict])
        if not archived_df.empty:
            updated_archives = pd.concat([archived_df, new_archive], ignore_index=True)
        else:
            updated_archives = new_archive
        
        try:
            updated_archives.to_parquet(archive_file, index=False)
        except Exception as e:
            print(f"Error archiving event: {e}")
    
    def _get_job_details(self, job_id: str) -> Dict[str, Any]:
        """Get job details from master data"""
        
        jobs_df = self._load_master_data("jobs")
        if jobs_df is None:
            return {}
        
        job_row = jobs_df[jobs_df['Job ID'].astype(str) == str(job_id)]
        if job_row.empty:
            return {}
        
        job = job_row.iloc[0]
        return {
            'job_date': str(job.get('Job Date', '')),
            'job_type': str(job.get('Job Type', '')),
            'job_total': float(job.get('Job Total', 0) or 0),
            'job_status': str(job.get('Status', ''))
        }
    
    def _get_estimate_details(self, estimate_id: str) -> Dict[str, Any]:
        """Get estimate details from master data"""
        
        estimates_df = self._load_master_data("estimates")
        if estimates_df is None:
            return {}
        
        estimate_row = estimates_df[estimates_df['Estimate ID'].astype(str) == str(estimate_id)]
        if estimate_row.empty:
            return {}
        
        estimate = estimate_row.iloc[0]
        return {
            'created_date': str(estimate.get('Created Date', '')),
            'estimate_type': str(estimate.get('Estimate Type', '')),
            'estimate_total': float(estimate.get('Estimate Total', 0) or 0),
            'status': str(estimate.get('Estimate Status', ''))
        }
    
    def _load_master_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load master data with caching"""
        
        if table_name in self._master_data_cache:
            return self._master_data_cache[table_name]
        
        file_path = self.parquet_dir / f"{table_name.capitalize()}.parquet"
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path)
            self._master_data_cache[table_name] = df
            return df
        except Exception:
            return None
    
    def _generate_event_id(self, change: Dict[str, Any]) -> str:
        """Generate unique event ID"""
        
        key_string = f"{change['entity_type']}_{change['entity_id']}_{change['customer_id']}_{change['timestamp']}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for recent events"""
        
        stats = {}
        
        for event_type in ["cancellations", "overdue_maintenance", "unsold_estimates", "lost_customers"]:
            recent_events = self._load_recent_events(event_type)
            stats[event_type] = {
                'count': len(recent_events),
                'active': len(recent_events[recent_events['status'] == 'active']) if not recent_events.empty else 0
            }
        
        return stats
    
    def _move_to_archive_df(self, event_type: str, events_df: pd.DataFrame):
        """Move events dataframe to archive file"""
        
        archive_file = self.archives_dir / f"archived_{event_type}.parquet"
        
        # Load existing archives
        if archive_file.exists():
            try:
                archived_df = pd.read_parquet(archive_file)
                updated_archives = pd.concat([archived_df, events_df], ignore_index=True)
            except Exception:
                updated_archives = events_df
        else:
            updated_archives = events_df
        
        try:
            updated_archives.to_parquet(archive_file, index=False)
        except Exception as e:
            print(f"Error archiving events: {e}")
    
    def _has_recent_customer_contact(self, customer_id: str, reference_date: str) -> bool:
        """Check if customer has had recent contact after reference date"""
        
        try:
            ref_date = pd.to_datetime(reference_date)
            
            # Check calls/contact data
            calls_df = self._load_master_data("calls")
            if calls_df is not None:
                customer_col = self._find_customer_column(calls_df)
                date_col = self._find_date_column(calls_df)
                
                if customer_col and date_col:
                    customer_calls = calls_df[
                        calls_df[customer_col].astype(str) == str(customer_id)
                    ]
                    
                    if not customer_calls.empty:
                        customer_calls[date_col] = pd.to_datetime(customer_calls[date_col], errors='coerce')
                        recent_calls = customer_calls[customer_calls[date_col] > ref_date]
                        
                        return not recent_calls.empty
            
        except Exception:
            pass
        
        return False
    
    def _customer_completed_maintenance_recently(self, customer_id: str) -> bool:
        """Check if customer completed maintenance recently"""
        
        try:
            jobs_df = self._load_master_data("jobs")
            if jobs_df is None:
                return False
            
            # Find customer jobs
            customer_jobs = jobs_df[jobs_df['Customer ID'].astype(str) == str(customer_id)]
            if customer_jobs.empty:
                return False
            
            # Check for completed maintenance jobs in last 30 days
            cutoff_date = datetime.now(UTC) - timedelta(days=30)
            
            for _, job in customer_jobs.iterrows():
                # Check if it's a maintenance job
                job_type = str(job.get('Job Type', '')).lower()
                job_class = str(job.get('Job Class', '')).lower()
                status = str(job.get('Status', '')).lower()
                
                maintenance_keywords = ['maintenance', 'service', 'repair', 'tune', 'clean']
                is_maintenance = any(keyword in job_type or keyword in job_class for keyword in maintenance_keywords)
                
                if is_maintenance and status in ['completed', 'finished', 'complete']:
                    # Check if completion date is recent
                    completion_date = job.get('Completion Date')
                    if completion_date:
                        try:
                            comp_date = pd.to_datetime(completion_date)
                            if comp_date >= cutoff_date:
                                return True
                        except:
                            pass
            
            return False
            
        except Exception:
            return False
    
    def _has_subsequent_job_activity(self, customer_id: str, reference_date: str) -> bool:
        """Check if customer has job activity after reference date"""
        
        try:
            ref_date = pd.to_datetime(reference_date)
            
            jobs_df = self._load_master_data("jobs")
            if jobs_df is None:
                return False
            
            customer_jobs = jobs_df[jobs_df['Customer ID'].astype(str) == str(customer_id)]
            
            if not customer_jobs.empty:
                # Check completion dates
                if 'Completion Date' in customer_jobs.columns:
                    completion_dates = pd.to_datetime(customer_jobs['Completion Date'], errors='coerce')
                    subsequent_jobs = completion_dates[completion_dates > ref_date]
                    if not subsequent_jobs.empty:
                        return True
                
                # Check created dates as fallback
                if 'Created Date' in customer_jobs.columns:
                    created_dates = pd.to_datetime(customer_jobs['Created Date'], errors='coerce')
                    subsequent_jobs = created_dates[created_dates > ref_date]
                    if not subsequent_jobs.empty:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _find_customer_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find customer ID column in DataFrame"""
        
        for col in df.columns:
            if 'customer' in col.lower() and 'id' in col.lower():
                return col
        return None
    
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find date column in DataFrame"""
        
        date_preferences = ['completion date', 'completed date', 'created date', 'date', 'job date']
        
        for pref in date_preferences:
            for col in df.columns:
                if pref in col.lower():
                    return col
        return None