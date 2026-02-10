"""Event persistence and lifecycle management system"""

import json
import hashlib
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .types import EventResult, EventSeverity


class EventStore:
    """Manages persistent storage and lifecycle of detected events"""
    
    def __init__(self, company: str, data_dir: Path):
        self.company = company
        self.data_dir = data_dir
        self.events_dir = data_dir.parent / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        
        # Type-specific event storage
        self.events_by_type_dir = self.events_dir / "by_type"
        self.events_by_type_dir.mkdir(parents=True, exist_ok=True)
        
        # Legacy support - keep for migration
        self.events_parquet = self.events_dir / "events_master.parquet"
        self.processed_keys_file = self.events_dir / "processed_event_keys.txt"
        self.event_changes_log = self.events_dir / "event_changes_log.jsonl"
    
    def save_events(self, events: List[EventResult], scan_timestamp: datetime) -> Dict[str, Any]:
        """Save events to persistent storage with deduplication and change tracking"""
        
        if not events:
            return {"new_events": 0, "updated_events": 0, "removed_events": 0}
        
        # Group events by type for type-specific storage
        events_by_type = {}
        for event in events:
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = []
            events_by_type[event.event_type].append(event)
        
        # Process each event type separately
        total_stats = {"new_events": 0, "updated_events": 0, "removed_events": 0, "by_type": {}}
        
        for event_type, type_events in events_by_type.items():
            type_stats = self._save_events_by_type(event_type, type_events, scan_timestamp)
            total_stats["new_events"] += type_stats["new_events"]
            total_stats["updated_events"] += type_stats["updated_events"] 
            total_stats["removed_events"] += type_stats["removed_events"]
            total_stats["by_type"][event_type] = type_stats
        
        # Also maintain legacy unified storage for backward compatibility
        self._save_events_legacy(events, scan_timestamp)
        
        # Log the operation
        self._log_persistence_operation(total_stats, scan_timestamp)
        
        return total_stats
    
    def _save_events_by_type(self, event_type: str, events: List[EventResult], scan_timestamp: datetime) -> Dict[str, Any]:
        """Save events of a specific type to type-specific storage"""
        
        # Load existing events for this type
        existing_df = self._load_existing_events_by_type(event_type)
        
        # Convert new events to DataFrame
        new_events_data = []
        for event in events:
            event_data = {
                "event_id": self._generate_event_id(event),
                "event_type": event.event_type,
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "severity": event.severity.value,
                "detected_at": event.detected_at.isoformat(),
                "scan_timestamp": scan_timestamp.isoformat(),
                "rule_name": event.rule_name,
                "status": "active",
                "last_updated": datetime.now(UTC).isoformat(),
                **event.details
            }
            new_events_data.append(event_data)
        
        new_df = pd.DataFrame(new_events_data)
        
        # Merge with existing events
        if existing_df.empty:
            updated_df = new_df
            stats = {"new_events": len(new_df), "updated_events": 0, "removed_events": 0}
        else:
            stats = self._merge_events(existing_df, new_df, scan_timestamp)
            updated_df = self._apply_event_changes(existing_df, new_df, stats)
        
        # Save to type-specific file
        self._save_events_parquet_by_type(event_type, updated_df)
        
        return stats
    
    def _generate_event_id(self, event: EventResult) -> str:
        """Generate unique ID for an event"""
        # Create hash from event type, entity type, entity ID, and key details
        key_parts = [
            event.event_type,
            event.entity_type, 
            event.entity_id,
            str(event.details.get("months_overdue", "")),
            str(event.details.get("system_age", "")),
            str(event.details.get("status", ""))
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:12]
    
    def _load_existing_events(self) -> pd.DataFrame:
        """Load existing events from parquet storage"""
        if not self.events_parquet.exists():
            return pd.DataFrame()
        
        try:
            return pd.read_parquet(self.events_parquet)
        except Exception:
            return pd.DataFrame()
    
    def _merge_events(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, scan_timestamp: datetime) -> Dict[str, Any]:
        """Detect changes between existing and new events"""
        
        stats = {"new_events": 0, "updated_events": 0, "removed_events": 0, "changes": []}
        
        # Find new events (not in existing)
        if "event_id" in existing_df.columns:
            existing_ids = set(existing_df["event_id"])
            new_ids = set(new_df["event_id"])
            
            # New events
            truly_new = new_ids - existing_ids
            stats["new_events"] = len(truly_new)
            
            # Updated events (same ID but different details)
            common_ids = existing_ids & new_ids
            for event_id in common_ids:
                existing_row = existing_df[existing_df["event_id"] == event_id].iloc[0]
                new_row = new_df[new_df["event_id"] == event_id].iloc[0]
                
                # Check if key fields changed
                if self._has_event_changed(existing_row, new_row):
                    stats["updated_events"] += 1
                    stats["changes"].append({
                        "event_id": event_id,
                        "change_type": "updated",
                        "old_severity": existing_row.get("severity"),
                        "new_severity": new_row.get("severity"),
                        "timestamp": scan_timestamp.isoformat()
                    })
            
            # Removed events (events that no longer qualify)
            # For now, we'll mark them as "resolved" rather than delete
            removed_ids = existing_ids - new_ids
            stats["removed_events"] = len(removed_ids)
            
            for event_id in removed_ids:
                stats["changes"].append({
                    "event_id": event_id,
                    "change_type": "resolved",
                    "timestamp": scan_timestamp.isoformat()
                })
        else:
            # First run - all events are new
            stats["new_events"] = len(new_df)
        
        return stats
    
    def _has_event_changed(self, existing_row: pd.Series, new_row: pd.Series) -> bool:
        """Check if an event has meaningfully changed"""
        
        # Check key fields that indicate real changes
        key_fields = ["severity", "status", "months_overdue", "system_age"]
        
        for field in key_fields:
            if field in existing_row and field in new_row:
                if str(existing_row[field]) != str(new_row[field]):
                    return True
        
        return False
    
    def _apply_event_changes(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, stats: Dict) -> pd.DataFrame:
        """Apply changes to create updated event DataFrame"""
        
        if existing_df.empty:
            return new_df
        
        # Start with existing events
        updated_df = existing_df.copy()
        
        # Mark removed events as resolved
        removed_ids = [c["event_id"] for c in stats["changes"] if c["change_type"] == "resolved"]
        if removed_ids:
            mask = updated_df["event_id"].isin(removed_ids)
            updated_df.loc[mask, "status"] = "resolved"
            updated_df.loc[mask, "last_updated"] = datetime.now(UTC).isoformat()
        
        # Update changed events
        updated_ids = [c["event_id"] for c in stats["changes"] if c["change_type"] == "updated"]
        for event_id in updated_ids:
            # Replace existing row with new data
            mask = updated_df["event_id"] == event_id
            new_row = new_df[new_df["event_id"] == event_id].iloc[0]
            new_row["last_updated"] = datetime.now(UTC).isoformat()
            
            # Update the row
            for col in new_row.index:
                if col in updated_df.columns:
                    updated_df.loc[mask, col] = new_row[col]
        
        # Add truly new events
        new_ids = set(new_df["event_id"]) - set(existing_df["event_id"])
        if new_ids:
            new_rows = new_df[new_df["event_id"].isin(new_ids)]
            updated_df = pd.concat([updated_df, new_rows], ignore_index=True)
        
        return updated_df
    
    def _save_events_parquet_by_type(self, event_type: str, events_df: pd.DataFrame) -> None:
        """Save events DataFrame to type-specific parquet with backup"""
        
        type_parquet_path = self.events_by_type_dir / f"{event_type}.parquet"
        
        # Create backup if file exists
        if type_parquet_path.exists():
            backup_path = type_parquet_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
            try:
                type_parquet_path.rename(backup_path)
            except Exception:
                pass
        
        # Convert all columns to string for consistency
        events_df_str = events_df.copy()
        for col in events_df_str.columns:
            events_df_str[col] = events_df_str[col].astype(str).fillna("")
        
        # Save to parquet
        table = pa.Table.from_pandas(events_df_str, preserve_index=False)
        pq.write_table(table, type_parquet_path)
    
    def _save_events_parquet(self, events_df: pd.DataFrame) -> None:
        """Save events DataFrame to parquet with backup (legacy method)"""
        
        # Create backup if file exists
        if self.events_parquet.exists():
            backup_path = self.events_parquet.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
            try:
                self.events_parquet.rename(backup_path)
            except Exception:
                pass
        
        # Convert all columns to string for consistency
        events_df_str = events_df.copy()
        for col in events_df_str.columns:
            events_df_str[col] = events_df_str[col].astype(str).fillna("")
        
        # Save to parquet
        table = pa.Table.from_pandas(events_df_str, preserve_index=False)
        pq.write_table(table, self.events_parquet)
    
    def _save_events_legacy(self, events: List[EventResult], scan_timestamp: datetime) -> None:
        """Maintain legacy unified storage for backward compatibility"""
        
        # Load existing legacy events
        existing_df = self._load_existing_events()
        
        # Convert new events to DataFrame
        new_events_data = []
        for event in events:
            event_data = {
                "event_id": self._generate_event_id(event),
                "event_type": event.event_type,
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "severity": event.severity.value,
                "detected_at": event.detected_at.isoformat(),
                "scan_timestamp": scan_timestamp.isoformat(),
                "rule_name": event.rule_name,
                "status": "active",
                "last_updated": datetime.now(UTC).isoformat(),
                **event.details
            }
            new_events_data.append(event_data)
        
        new_df = pd.DataFrame(new_events_data)
        
        # Merge with existing events
        if existing_df.empty:
            updated_df = new_df
        else:
            stats = self._merge_events(existing_df, new_df, scan_timestamp)
            updated_df = self._apply_event_changes(existing_df, new_df, stats)
        
        # Save to legacy file
        self._save_events_parquet(updated_df)
    
    def _log_persistence_operation(self, stats: Dict[str, Any], scan_timestamp: datetime) -> None:
        """Log event persistence operations"""
        
        log_entry = {
            "ts": datetime.now(UTC).isoformat(),
            "company": self.company,
            "action": "event_persistence",
            "scan_timestamp": scan_timestamp.isoformat(),
            **stats
        }
        
        with open(self.event_changes_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Log individual changes
        for change in stats.get("changes", []):
            change_entry = {
                "ts": datetime.now(UTC).isoformat(),
                "company": self.company,
                "action": "event_change",
                **change
            }
            with open(self.event_changes_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(change_entry) + "\n")
    
    def get_recent_events(self, event_type: Optional[str] = None, days_back: int = 30) -> pd.DataFrame:
        """Get recent events from storage"""
        
        events_df = self._load_existing_events()
        if events_df.empty:
            return events_df
        
        # Filter by event type if specified
        if event_type:
            events_df = events_df[events_df["event_type"] == event_type]
        
        # Filter by date
        cutoff = datetime.now(UTC) - timedelta(days=days_back)
        events_df["detected_at_dt"] = pd.to_datetime(events_df["detected_at"], errors="coerce")
        recent_mask = events_df["detected_at_dt"] >= cutoff
        
        return events_df[recent_mask]
    
    def get_active_events(self, event_type: Optional[str] = None) -> pd.DataFrame:
        """Get currently active (unresolved) events"""
        
        events_df = self._load_existing_events()
        if events_df.empty:
            return events_df
        
        # Filter to active events
        active_mask = events_df["status"] == "active"
        active_events = events_df[active_mask]
        
        # Filter by event type if specified
        if event_type:
            active_events = active_events[active_events["event_type"] == event_type]
        
        return active_events
    
    def _load_existing_events_by_type(self, event_type: str) -> pd.DataFrame:
        """Load existing events for a specific type"""
        
        type_parquet_path = self.events_by_type_dir / f"{event_type}.parquet"
        
        if not type_parquet_path.exists():
            return pd.DataFrame()
        
        try:
            return pd.read_parquet(type_parquet_path)
        except Exception:
            return pd.DataFrame()
    
    def load_events_by_type(self, event_type: str, days_back: Optional[int] = None) -> pd.DataFrame:
        """Load events for a specific type, optionally filtered by date"""
        
        events_df = self._load_existing_events_by_type(event_type)
        
        if events_df.empty or days_back is None:
            return events_df
        
        # Filter by date
        cutoff_date = datetime.now(UTC) - timedelta(days=days_back)
        
        try:
            events_df["detected_at_dt"] = pd.to_datetime(events_df["detected_at"])
            recent_df = events_df[events_df["detected_at_dt"] >= cutoff_date]
            recent_df = recent_df.drop("detected_at_dt", axis=1)
            return recent_df
        except Exception:
            cutoff_str = cutoff_date.isoformat()
            return events_df[events_df["detected_at"] >= cutoff_str]
    
    def get_all_event_types(self) -> List[str]:
        """Get list of all available event types"""
        
        event_types = []
        
        # Get types from type-specific storage
        if self.events_by_type_dir.exists():
            for parquet_file in self.events_by_type_dir.glob("*.parquet"):
                if not parquet_file.name.startswith("backup_"):
                    event_types.append(parquet_file.stem)
        
        # Also check legacy storage for additional types
        legacy_df = self._load_existing_events()
        if not legacy_df.empty and "event_type" in legacy_df.columns:
            legacy_types = legacy_df["event_type"].unique().tolist()
            for event_type in legacy_types:
                if event_type not in event_types:
                    event_types.append(event_type)
        
        return sorted(event_types)


class ChangeLogEventDetector:
    """Detects events from change logs instead of master data scans"""
    
    def __init__(self, company: str, data_dir: Path):
        self.company = company
        self.data_dir = data_dir
        primary = data_dir / "logs"
        fallback = data_dir.parent / "logs"
        self.logs_dir = primary if primary.exists() or not fallback.exists() else fallback
    
    def detect_recent_cancellations(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Detect job cancellations from change logs within time window"""
        
        cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
        
        # Load job change logs
        job_changes_file = self.logs_dir / "job_changes_log.jsonl"
        if not job_changes_file.exists():
            return []
        
        cancellations = []
        
        try:
            with open(job_changes_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Check if this is a recent cancellation
                        if (entry.get("change_type") == "update_cell" and
                            entry.get("column", "").lower() in ["status", "job status"] and
                            str(entry.get("new", "")).lower() in ["canceled", "cancelled"] and
                            str(entry.get("old", "")).lower() not in ["canceled", "cancelled"]):
                            
                            # Check if within time window
                            try:
                                entry_time = datetime.fromisoformat(entry.get("ts", ""))
                                if entry_time >= cutoff:
                                    cancellations.append({
                                        "job_id": entry.get("id"),
                                        "timestamp": entry.get("ts"),
                                        "old_status": entry.get("old"),
                                        "new_status": entry.get("new"),
                                        "change_source": "log_analysis"
                                    })
                            except ValueError:
                                continue
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        
        return cancellations
    
    def detect_recent_unsold_estimates(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Detect unsold estimates from change logs within time window"""
        
        cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
        
        # Load estimate change logs
        estimate_changes_file = self.logs_dir / "estimate_changes_log.jsonl"
        if not estimate_changes_file.exists():
            return []
        
        unsold_estimates = []
        target_statuses = {"dismissed", "open"}
        
        try:
            with open(estimate_changes_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Check if this is a recent status change to unsold
                        if (entry.get("change_type") == "update_cell" and
                            entry.get("column", "").lower() in ["estimate status", "status"] and
                            str(entry.get("new", "")).lower() in target_statuses):
                            
                            # Check if within time window
                            try:
                                entry_time = datetime.fromisoformat(entry.get("ts", ""))
                                if entry_time >= cutoff:
                                    unsold_estimates.append({
                                        "estimate_id": entry.get("id"),
                                        "timestamp": entry.get("ts"),
                                        "old_status": entry.get("old"),
                                        "new_status": entry.get("new"),
                                        "change_source": "log_analysis"
                                    })
                            except ValueError:
                                continue
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        
        return unsold_estimates


class EventEnricher:
    """Enriches event payloads with RFM scores, demographic data, and marketable flags"""
    
    def __init__(self, company: str, data_dir: Path):
        self.company = company
        self.data_dir = data_dir
        self.parquet_dir = Path(f"companies/{company}/parquet")
        
        # Customer profile data sources (NEW - faster enrichment) - now unified in single file
        self.unified_profiles_file = self.parquet_dir / "customer_profiles_core_data.parquet"
        
        # Profile data cache
        self._unified_profiles_cache: Optional[pd.DataFrame] = None
        
        # Legacy enrichment data cache (fallback when profiles not available)
        self._jobs_df = None
        self._customers_df = None
        self._locations_df = None
        self._estimates_df = None
        self._exemptions_df = None
        self._income_df = None
    
    def enrich_event(self, event: EventResult, enrichment_config: Dict[str, bool]) -> Dict[str, Any]:
        """Enrich a single event with configured additional data"""
        
        enriched_payload = event.details.copy()
        
        # Get customer ID for the event
        customer_id = self._get_customer_id_for_event(event)
        if not customer_id:
            return enriched_payload
        
        # Try to get enrichment data from customer profiles first (NEW - much faster)
        profile_data = self._get_profile_enrichment_data(customer_id)
        
        if profile_data:
            # Use pre-calculated profile data (fast path)
            if enrichment_config.get("include_rfm", False):
                rfm_keys = ["rfm_recency", "rfm_frequency", "rfm_monetary", "rfm_score", "rfm_segment"]
                for key in rfm_keys:
                    if key in profile_data and pd.notna(profile_data[key]):
                        enriched_payload[key] = profile_data[key]
            
            if enrichment_config.get("include_demographics", False):
                demo_keys = ["household_income", "property_value"]
                for key in demo_keys:
                    if key in profile_data and pd.notna(profile_data[key]):
                        enriched_payload[key] = profile_data[key]
            
            if enrichment_config.get("include_marketable", False):
                marketable_keys = ["is_marketable", "do_not_call", "do_not_service"]
                for key in marketable_keys:
                    if key in profile_data and pd.notna(profile_data[key]):
                        enriched_payload[key] = profile_data[key]
            
            if enrichment_config.get("include_segmentation", False):
                segment_keys = ["customer_tier", "customer_segment"]
                for key in segment_keys:
                    if key in profile_data and pd.notna(profile_data[key]):
                        enriched_payload[key] = profile_data[key]
            
            if enrichment_config.get("include_permit_data", False):
                permit_keys = ["permit_matches", "permit_count", "competitor_permits", "competitor_permit_count"]
                for key in permit_keys:
                    if key in profile_data and pd.notna(profile_data[key]):
                        enriched_payload[key] = profile_data[key]
        else:
            # Fallback to legacy real-time calculation (slower but comprehensive)
            if enrichment_config.get("include_rfm", False):
                rfm_data = self._calculate_rfm_scores(customer_id, event.event_type)
                enriched_payload.update(rfm_data)
            
            if enrichment_config.get("include_demographics", False):
                demo_data = self._get_demographic_data(customer_id)
                enriched_payload.update(demo_data)
            
            if enrichment_config.get("include_marketable", False):
                marketable_data = self._calculate_marketable_flag(customer_id, event.event_type)
                enriched_payload.update(marketable_data)
            
            if enrichment_config.get("include_segmentation", False):
                segment_data = self._get_customer_segmentation(customer_id)
                enriched_payload.update(segment_data)
        
        return enriched_payload
    
    def _get_customer_id_for_event(self, event: EventResult) -> Optional[str]:
        """Extract customer ID from event"""
        
        # Direct customer ID in details
        if "customer_id" in event.details:
            return str(event.details["customer_id"])
        
        # For location-based events, look up customer via location
        if event.entity_type == "location":
            locations_df = self._get_locations_df()
            if locations_df is not None:
                location_mask = locations_df["Location ID"].astype(str) == event.entity_id
                if location_mask.any():
                    customer_id = locations_df[location_mask]["Customer ID"].iloc[0]
                    return str(customer_id) if pd.notna(customer_id) else None
        
        return None
    
    def _get_profile_enrichment_data(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get enrichment data from unified customer profiles (NEW - fast path)"""
        
        # Load unified profile data
        unified_df = self._get_unified_profiles_df()
        if unified_df is not None and not unified_df.empty:
            # Find profile by customer_id
            customer_mask = unified_df["customer_id"].astype(str) == str(customer_id)
            if customer_mask.any():
                profile_row = unified_df[customer_mask].iloc[0]
                return profile_row.to_dict()
        
        return None
    
    def _get_unified_profiles_df(self) -> Optional[pd.DataFrame]:
        """Load unified customer profiles data (core + enrichment in single file)"""
        
        if self._unified_profiles_cache is not None:
            return self._unified_profiles_cache
        
        if not self.unified_profiles_file.exists():
            return None
        
        try:
            self._unified_profiles_cache = pd.read_parquet(self.unified_profiles_file)
            return self._unified_profiles_cache
        except Exception:
            return None
    
    def _calculate_rfm_scores(self, customer_id: str, event_type: str) -> Dict[str, Any]:
        """Calculate RFM scores using legacy logic"""
        
        jobs_df = self._get_jobs_df()
        if jobs_df is None:
            return {}
        
        # Get all jobs for this customer and their locations
        customer_jobs = self._get_customer_jobs(customer_id, jobs_df)
        
        if customer_jobs.empty:
            return {
                "recent_activity": False,
                "recent_activity_dates": "",
                "completed_jobs_found": 0,
                "rfm_recency_score": 1,
                "rfm_frequency_score": 1,
                "rfm_monetary_score": 1,
                "rfm_score": 3,
                "rfm_recency_value": None,
                "rfm_frequency_value": 0,
                "rfm_monetary_value": 0.0,
                "customer_segment": "T5 - Lost Customer"
            }
        
        # Calculate recent activity (legacy logic)
        window_days = 25  # Configurable constant from legacy
        today = datetime.now().date()
        window_start = today - timedelta(days=window_days)
        window_end = today + timedelta(days=window_days)
        
        recent_activity = False
        recent_dates = set()
        
        for _, job in customer_jobs.iterrows():
            # Check Created Date and Scheduled Date
            for date_col in ["Created Date", "Scheduled Date"]:
                if date_col in job and pd.notna(job[date_col]):
                    try:
                        job_date = pd.to_datetime(job[date_col]).date()
                        if window_start <= job_date <= window_end:
                            if str(job.get("Status", "")).lower() != "canceled":
                                recent_activity = True
                                recent_dates.add(job_date.isoformat())
                    except Exception:
                        continue
        
        # Calculate RFM components
        completed_jobs = customer_jobs[customer_jobs["Status"] == "Completed"]
        
        # Recency: days since most recent completed job
        rfm_recency_value = None
        rfm_recency_score = 1
        
        if not completed_jobs.empty:
            try:
                latest_completion = completed_jobs["Created Date"].max()
                if pd.notna(latest_completion):
                    latest_date = pd.to_datetime(latest_completion).date()
                    days_since = (today - latest_date).days
                    rfm_recency_value = days_since
                    
                    # Score based on days
                    if days_since <= 14:
                        rfm_recency_score = 5
                    elif days_since <= 30:
                        rfm_recency_score = 4
                    elif days_since <= 90:
                        rfm_recency_score = 3
                    elif days_since <= 365:
                        rfm_recency_score = 2
                    else:
                        rfm_recency_score = 1
            except Exception:
                pass
        
        # Frequency: number of completed jobs
        rfm_frequency_value = len(completed_jobs)
        if rfm_frequency_value <= 1:
            rfm_frequency_score = 1
        elif rfm_frequency_value == 2:
            rfm_frequency_score = 2
        elif rfm_frequency_value == 3:
            rfm_frequency_score = 3
        elif 4 <= rfm_frequency_value <= 10:
            rfm_frequency_score = 4
        else:
            rfm_frequency_score = 5
        
        # Monetary: sum of Jobs Total
        rfm_monetary_value = 0.0
        try:
            if "Jobs Total" in completed_jobs.columns:
                monetary_values = pd.to_numeric(completed_jobs["Jobs Total"], errors="coerce")
                rfm_monetary_value = float(monetary_values.sum())
        except Exception:
            pass
        
        # Monetary score
        if rfm_monetary_value <= 750:
            rfm_monetary_score = 1
        elif rfm_monetary_value <= 1500:
            rfm_monetary_score = 2
        elif rfm_monetary_value <= 3750:
            rfm_monetary_score = 3
        elif rfm_monetary_value <= 10000:
            rfm_monetary_score = 4
        else:
            rfm_monetary_score = 5
        
        # Total RFM score and segmentation
        rfm_score = rfm_recency_score + rfm_frequency_score + rfm_monetary_score
        
        if rfm_score <= 3:
            customer_segment = "T5 - Lost Customer"
        elif rfm_score <= 6:
            customer_segment = "T4 - Occasional Shopper"
        elif rfm_score <= 9:
            customer_segment = "T3 - Emerging Customer"
        elif rfm_score <= 12:
            customer_segment = "T2 - Steady Customer"
        elif rfm_score <= 15:
            customer_segment = "T1 - Premium Customer"
        else:
            customer_segment = "Unknown"
        
        return {
            "recent_activity": recent_activity,
            "recent_activity_dates": ";".join(sorted(recent_dates)),
            "completed_jobs_found": len(completed_jobs),
            "rfm_recency_score": rfm_recency_score,
            "rfm_frequency_score": rfm_frequency_score,
            "rfm_monetary_score": rfm_monetary_score,
            "rfm_score": rfm_score,
            "rfm_recency_value": rfm_recency_value,
            "rfm_frequency_value": rfm_frequency_value,
            "rfm_monetary_value": rfm_monetary_value,
            "customer_segment": customer_segment
        }
    
    def _get_customer_jobs(self, customer_id: str, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Get all jobs for a customer and their locations"""
        
        # Get customer's locations
        locations_df = self._get_locations_df()
        customer_locations = set()
        
        if locations_df is not None:
            customer_mask = locations_df["Customer ID"].astype(str) == customer_id
            customer_locations = set(locations_df[customer_mask]["Location ID"].astype(str))
        
        # Get jobs for customer and all their locations
        customer_job_mask = jobs_df["Customer ID"].astype(str) == customer_id
        location_job_mask = jobs_df["Location ID"].astype(str).isin(customer_locations)
        
        all_customer_jobs = jobs_df[customer_job_mask | location_job_mask]
        
        # Deduplicate by job content (excluding Job ID as per legacy)
        if not all_customer_jobs.empty:
            dedup_columns = [col for col in all_customer_jobs.columns if col.lower() != "job id"]
            all_customer_jobs = all_customer_jobs.drop_duplicates(subset=dedup_columns)
        
        return all_customer_jobs
    
    def _calculate_marketable_flag(self, customer_id: str, event_type: str) -> Dict[str, Any]:
        """Calculate marketable flag using legacy logic"""
        
        # Get customer data
        customers_df = self._get_customers_df()
        if customers_df is None:
            return {"marketable": True}
        
        customer_mask = customers_df["Customer ID"].astype(str) == customer_id
        if not customer_mask.any():
            return {"marketable": True}
        
        customer_row = customers_df[customer_mask].iloc[0]
        
        # Check Do Not flags
        do_not_service = str(customer_row.get("Do Not Service", "")).lower() == "true"
        do_not_call = str(customer_row.get("Do Not Call", "")).lower() == "true"
        do_not_mail = str(customer_row.get("Do Not Mail", "")).lower() == "true"
        
        # Check recent activity
        rfm_data = self._calculate_rfm_scores(customer_id, event_type)
        recent_activity = rfm_data.get("recent_activity", False)
        
        # Marketable logic from legacy
        marketable = not (recent_activity or do_not_service or do_not_call or do_not_mail)
        
        return {
            "marketable": marketable,
            "do_not_service": do_not_service,
            "do_not_call": do_not_call,
            "do_not_mail": do_not_mail
        }
    
    def _get_demographic_data(self, customer_id: str) -> Dict[str, Any]:
        """Get demographic data by ZIP code"""
        
        # Get customer ZIP
        customers_df = self._get_customers_df()
        if customers_df is None:
            return {}
        
        customer_mask = customers_df["Customer ID"].astype(str) == customer_id
        if not customer_mask.any():
            return {}
        
        customer_row = customers_df[customer_mask].iloc[0]
        zip_code = str(customer_row.get("Zip", "")).strip()
        
        if not zip_code:
            return {}
        
        # Load income/property data
        income_df = self._get_income_df()
        if income_df is None:
            return {}
        
        # Match by ZIP
        zip_mask = income_df["Zip Code"].astype(str) == zip_code
        if zip_mask.any():
            income_row = income_df[zip_mask].iloc[0]
            household_income = income_row.get("Household Income", "")
            property_value = income_row.get("Property Value", "")
            
            # Convert to numeric values if possible
            try:
                household_income = float(household_income) if household_income and str(household_income).strip() else None
            except (ValueError, TypeError):
                household_income = None
            
            try:
                property_value = float(property_value) if property_value and str(property_value).strip() else None
            except (ValueError, TypeError):
                property_value = None
            
            return {
                "household_income": household_income,
                "property_value": property_value
            }
        
        return {}
    
    def _get_customer_segmentation(self, customer_id: str) -> Dict[str, Any]:
        """Get customer segmentation data"""
        
        # This would include additional segmentation logic
        # For now, return RFM-based segment (already in RFM calculation)
        return {}
    
    # Data loading methods with caching
    def _get_jobs_df(self) -> Optional[pd.DataFrame]:
        if self._jobs_df is None:
            jobs_path = self.parquet_dir / "Jobs.parquet"
            if jobs_path.exists():
                self._jobs_df = pd.read_parquet(jobs_path)
        return self._jobs_df
    
    def _get_customers_df(self) -> Optional[pd.DataFrame]:
        if self._customers_df is None:
            customers_path = self.parquet_dir / "Customers.parquet"
            if customers_path.exists():
                self._customers_df = pd.read_parquet(customers_path)
        return self._customers_df
    
    def _get_locations_df(self) -> Optional[pd.DataFrame]:
        if self._locations_df is None:
            locations_path = self.parquet_dir / "Locations.parquet"
            if locations_path.exists():
                self._locations_df = pd.read_parquet(locations_path)
        return self._locations_df
    
    def _get_estimates_df(self) -> Optional[pd.DataFrame]:
        if self._estimates_df is None:
            estimates_path = self.parquet_dir / "Estimates.parquet"
            if estimates_path.exists():
                self._estimates_df = pd.read_parquet(estimates_path)
        return self._estimates_df
    
    def _get_income_df(self) -> Optional[pd.DataFrame]:
        if self._income_df is None:
            income_path = Path("global_data/demographics/demographics.csv")
            if income_path.exists():
                self._income_df = pd.read_csv(income_path, dtype=str)
        return self._income_df
