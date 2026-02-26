"""Event-specific master parquet file storage system"""

import time
import hashlib
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datahound.storage.bootstrap import get_storage_dal_from_env
from datahound.storage.db.models import EventIndexRecord
from .types import EventResult, EventSeverity


class EventMasterStorage:
    """Manages master parquet files for individual event types"""
    
    def __init__(self, company: str, data_dir: Path):
        self.company = company
        self.data_dir = data_dir
        self.events_dir = data_dir.parent / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        
        # Master event files directory
        self.master_files_dir = self.events_dir / "master_files"
        self.master_files_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_file = data_dir / "logs" / "event_storage_log.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Optional control-plane DAL (enabled when DATAHOUND_STORAGE_URL is configured)
        self._storage_dal = get_storage_dal_from_env()
    
    def get_master_file_path(self, event_type: str) -> Path:
        """Get the master parquet file path for an event type"""
        return self.master_files_dir / f"{event_type}_master.parquet"
    
    def save_event_results(self, event_type: str, events: List[EventResult], 
                          scan_timestamp: datetime, update_mode: str = "upsert") -> Dict[str, Any]:
        """Save event results to event-specific master parquet file
        
        Args:
            event_type: Type of event (e.g., 'overdue_maintenance')
            events: List of event results to save
            scan_timestamp: When the scan was performed
            update_mode: 'upsert' (update/insert), 'append_only' (for aging systems)
        """
        
        if not events:
            return {"new_events": 0, "updated_events": 0, "total_events": 0}
        
        master_file = self.get_master_file_path(event_type)
        
        # Convert events to DataFrame
        event_data = []
        for event in events:
            event_record = {
                "event_id": self._generate_event_id(event),
                "event_type": event.event_type,
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "severity": event.severity.value,
                "detected_at": event.detected_at.isoformat(),
                "scan_timestamp": scan_timestamp.isoformat(),
                "rule_name": event.rule_name,
                "last_updated": datetime.now(UTC).isoformat(),
                **event.details  # Flatten all event details
            }
            event_data.append(event_record)
        
        new_df = pd.DataFrame(event_data)
        
        # Handle existing data
        stats = {"new_events": 0, "updated_events": 0, "total_events": 0}
        
        if master_file.exists():
            try:
                existing_df = pd.read_parquet(master_file)
                
                if update_mode == "append_only":
                    # For aging systems - only append new data, never update
                    if "event_id" in existing_df.columns:
                        existing_ids = set(existing_df["event_id"])
                        new_events_only = new_df[~new_df["event_id"].isin(existing_ids)]
                        
                        if len(new_events_only) > 0:
                            combined_df = pd.concat([existing_df, new_events_only], ignore_index=True)
                            stats["new_events"] = len(new_events_only)
                            stats["total_events"] = len(combined_df)
                            
                            self._log_event("info", f"Appended {len(new_events_only)} new {event_type} events", {
                                "event_type": event_type,
                                "new_events": len(new_events_only),
                                "total_events": len(combined_df)
                            })
                        else:
                            combined_df = existing_df
                            stats["total_events"] = len(combined_df)
                            
                            self._log_event("info", f"No new {event_type} events to append", {
                                "event_type": event_type,
                                "total_events": len(combined_df)
                            })
                    else:
                        # No event_id column, just append
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        stats["new_events"] = len(new_df)
                        stats["total_events"] = len(combined_df)
                
                else:
                    # Upsert mode - update existing, insert new
                    if "event_id" in existing_df.columns:
                        # Merge based on event_id
                        existing_ids = set(existing_df["event_id"])
                        new_ids = set(new_df["event_id"])
                        
                        # Find new vs updated events
                        truly_new_ids = new_ids - existing_ids
                        update_ids = new_ids & existing_ids
                        
                        stats["new_events"] = len(truly_new_ids)
                        stats["updated_events"] = len(update_ids)
                        
                        # Remove old versions of updated events
                        existing_without_updates = existing_df[~existing_df["event_id"].isin(update_ids)]
                        
                        # Combine: existing (minus updates) + all new data
                        combined_df = pd.concat([existing_without_updates, new_df], ignore_index=True)
                        stats["total_events"] = len(combined_df)
                        
                        self._log_event("info", f"Upserted {event_type} events", {
                            "event_type": event_type,
                            "new_events": stats["new_events"],
                            "updated_events": stats["updated_events"],
                            "total_events": stats["total_events"]
                        })
                    else:
                        # No event_id column, just replace all data
                        combined_df = new_df
                        stats["new_events"] = len(new_df)
                        stats["total_events"] = len(combined_df)
                
            except Exception as e:
                self._log_event("error", f"Error reading existing {event_type} master file", {
                    "event_type": event_type,
                    "error": str(e)
                })
                # Fall back to overwriting
                combined_df = new_df
                stats["new_events"] = len(new_df)
                stats["total_events"] = len(combined_df)
        else:
            # No existing file - all events are new
            combined_df = new_df
            stats["new_events"] = len(new_df)
            stats["total_events"] = len(combined_df)
        
        # Save to parquet
        try:
            combined_df.to_parquet(master_file, index=False)
            
            self._log_event("info", f"Saved {event_type} master file", {
                "event_type": event_type,
                "file": str(master_file),
                "total_events": len(combined_df),
                "file_size_mb": round(master_file.stat().st_size / (1024*1024), 2)
            })

            self._sync_events_index(event_type=event_type, frame=combined_df, source_file=master_file)
            
        except Exception as e:
            self._log_event("error", f"Failed to save {event_type} master file", {
                "event_type": event_type,
                "error": str(e)
            })
            raise
        
        return stats
    
    def load_event_master_data(self, event_type: str) -> Optional[pd.DataFrame]:
        """Load master data for an event type"""
        
        master_file = self.get_master_file_path(event_type)
        
        if not master_file.exists():
            return None
        
        try:
            df = pd.read_parquet(master_file)
            
            self._log_event("info", f"Loaded {event_type} master data", {
                "event_type": event_type,
                "records": len(df),
                "columns": len(df.columns)
            })
            
            return df
            
        except Exception as e:
            self._log_event("error", f"Failed to load {event_type} master data", {
                "event_type": event_type,
                "error": str(e)
            })
            return None
    
    def get_event_master_stats(self, event_type: str) -> Dict[str, Any]:
        """Get statistics about an event master file"""
        
        master_file = self.get_master_file_path(event_type)
        
        if not master_file.exists():
            return {"exists": False}
        
        try:
            df = pd.read_parquet(master_file)
            
            stats = {
                "exists": True,
                "total_records": len(df),
                "file_size_mb": round(master_file.stat().st_size / (1024*1024), 2),
                "last_modified": datetime.fromtimestamp(master_file.stat().st_mtime).isoformat(),
                "columns": list(df.columns)
            }
            
            # Add severity breakdown if available
            if "severity" in df.columns:
                severity_counts = df["severity"].value_counts().to_dict()
                stats["severity_breakdown"] = severity_counts
            
            # Add recent activity if available
            if "scan_timestamp" in df.columns:
                df["scan_timestamp"] = pd.to_datetime(df["scan_timestamp"])
                latest_scan = df["scan_timestamp"].max()
                stats["latest_scan"] = latest_scan.isoformat()
                
                # Scans in last 30 days
                recent_cutoff = datetime.now() - pd.Timedelta(days=30)
                recent_scans = len(df[df["scan_timestamp"] > recent_cutoff])
                stats["recent_scans_30d"] = recent_scans
            
            return stats
            
        except Exception as e:
            return {
                "exists": True,
                "error": str(e)
            }
    
    def list_all_event_master_files(self) -> Dict[str, Dict[str, Any]]:
        """List all event master files with their statistics"""
        
        master_files = {}
        
        for master_file in self.master_files_dir.glob("*_master.parquet"):
            event_type = master_file.stem.replace("_master", "")
            master_files[event_type] = self.get_event_master_stats(event_type)
        
        return master_files
    
    def cleanup_old_events(self, event_type: str, days_to_keep: int = 90) -> int:
        """Clean up old events from master file"""
        
        master_file = self.get_master_file_path(event_type)
        
        if not master_file.exists():
            return 0
        
        try:
            df = pd.read_parquet(master_file)
            
            if "scan_timestamp" not in df.columns:
                return 0
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            
            df["scan_timestamp"] = pd.to_datetime(df["scan_timestamp"])
            
            # Keep only recent events
            recent_df = df[df["scan_timestamp"] > cutoff_date]
            
            removed_count = len(df) - len(recent_df)
            
            if removed_count > 0:
                # Save cleaned data
                recent_df.to_parquet(master_file, index=False)
                
                self._log_event("info", f"Cleaned up old {event_type} events", {
                    "event_type": event_type,
                    "removed_events": removed_count,
                    "remaining_events": len(recent_df),
                    "days_to_keep": days_to_keep
                })
            
            return removed_count
            
        except Exception as e:
            self._log_event("error", f"Failed to cleanup {event_type} events", {
                "event_type": event_type,
                "error": str(e)
            })
            return 0
    
    def _sync_events_index(self, *, event_type: str, frame: pd.DataFrame, source_file: Path) -> None:
        """Best-effort sync of parquet master records into events_index control-plane table."""
        if self._storage_dal is None:
            return
        if frame.empty:
            return
        required_cols = {"event_id", "entity_type", "entity_id", "severity"}
        if not required_cols.issubset(set(frame.columns)):
            return

        records: List[EventIndexRecord] = []
        known_event_ids: Set[str] = set()
        for _, row in frame.iterrows():
            event_id = str(row.get("event_id", "")).strip()
            if not event_id:
                continue
            known_event_ids.add(event_id)
            detected_at = self._parse_timestamp(row.get("detected_at")) or datetime.now(UTC)
            status_value = str(row.get("status", "active") or "active").strip().lower()
            if status_value not in {"active", "resolved", "archived"}:
                status_value = "active"

            details = {
                key: self._coerce_json_value(value)
                for key, value in row.items()
                if key
                not in {
                    "event_id",
                    "event_type",
                    "entity_type",
                    "entity_id",
                    "severity",
                    "status",
                    "detected_at",
                    "last_updated",
                    "scan_timestamp",
                }
            }

            records.append(
                EventIndexRecord(
                    id=None,
                    event_id=event_id,
                    company=self.company,
                    event_type=str(row.get("event_type", event_type) or event_type),
                    entity_type=str(row.get("entity_type", "unknown") or "unknown"),
                    entity_id=str(row.get("entity_id", "unknown") or "unknown"),
                    severity=str(row.get("severity", "medium") or "medium"),
                    status=status_value,
                    first_seen_at=detected_at,
                    last_seen_at=datetime.now(UTC),
                    source_file=str(source_file),
                    details_json=details,
                )
            )

        if not records:
            return

        try:
            self._storage_dal.upsert_events(records)
            self._storage_dal.resolve_missing_events(
                known_event_ids,
                event_type=event_type,
                company=self.company,
            )
        except Exception as exc:
            self._log_event("warning", "Failed syncing events_index", {
                "event_type": event_type,
                "error": str(exc),
            })

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except Exception:
            return None

    @staticmethod
    def _coerce_json_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return str(value)

    def _generate_event_id(self, event: EventResult) -> str:
        """Generate unique ID for an event"""
        
        # For aging systems, include more specificity to allow multiple analyses
        if event.event_type == "aging_systems":
            key_parts = [
                event.event_type,
                event.entity_type,
                event.entity_id,
                str(event.details.get("system_age", "")),
                event.detected_at.strftime("%Y%m%d"),  # Include date for uniqueness
                str(event.details.get("reasoning", ""))[:50]  # Include part of reasoning
            ]
        else:
            # Standard ID generation for other events
            key_parts = [
                event.event_type,
                event.entity_type,
                event.entity_id,
                str(event.details.get("months_overdue", "")),
                str(event.details.get("status", "")),
                str(event.details.get("system_age", ""))
            ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _log_event(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log event to JSONL file"""
        
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
            "company": self.company,
            "component": "event_master_storage"
        }
        
        if details:
            log_entry.update(details)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                import json
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass  # Don't let logging errors break the main process
