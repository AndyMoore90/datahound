"""
Core Business Events System - Implements the essential business events
"""

import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from .types import EventResult, EventSeverity, EventRule, EventScanResult, EventScanConfig, RuleType
from .engine import EventScanner
from .scan_methods import scan_unsold_estimates, scan_lost_customers
from .event_storage import EventMasterStorage
from .logging import EventLogger
from ..upsert.types import AuditChange


class CoreBusinessEventProcessor:
    """Processes the core business events required for HVAC business intelligence"""
    
    def __init__(self, company: str, data_dir: Path, parquet_dir: Path):
        self.company = company
        self.data_dir = data_dir
        self.parquet_dir = parquet_dir
        
        # Initialize components
        self.event_scanner = EventScanner(company, parquet_dir, data_dir=data_dir)
        self.master_storage = EventMasterStorage(company, data_dir)
        self.logger = EventLogger(company, data_dir)
    
    def process_realtime_events(self, audit_changes: List[AuditChange]) -> Dict[str, Any]:
        """Process real-time events during upsert (cancellations, unsold estimates)"""
        
        start_time = time.perf_counter()
        events_detected = []
        
        try:
            # Group changes by file type
            changes_by_type = {}
            for change in audit_changes:
                file_type = getattr(change, 'file_type', 'unknown')
                if file_type not in changes_by_type:
                    changes_by_type[file_type] = []
                changes_by_type[file_type].append(change)
            
            # Process job cancellations (real-time)
            if 'jobs' in changes_by_type:
                job_events = self._detect_job_cancellations(changes_by_type['jobs'])
                events_detected.extend(job_events)
            
            # Process estimate status changes (real-time) 
            if 'estimates' in changes_by_type:
                estimate_events = self._detect_unsold_estimates_from_changes(changes_by_type['estimates'])
                events_detected.extend(estimate_events)
            
            # Save real-time events
            if events_detected:
                self._save_realtime_events(events_detected)
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            
            return {
                "realtime_events_detected": len(events_detected),
                "processing_time_ms": duration_ms,
                "event_types": list(set(e.event_type for e in events_detected))
            }
            
        except Exception as e:
            self.logger.log_event("error", f"Real-time event processing failed: {e}")
            return {"realtime_events_detected": 0, "error": str(e)}
    
    def process_full_scan_events(self, exclude_aging_systems: bool = True) -> Dict[str, Any]:
        """Process full scan events after upsert (lost customers, overdue maintenance)"""
        
        start_time = time.perf_counter()
        scan_results = {}
        
        try:
            # Configure full scan
            scan_config = EventScanConfig(
                use_change_log_detection=False,  # Full master data scan
                persist_events=True,
                include_enriched_data=True,
                show_progress=False,
                processing_limit=None
            )
            
            # 1. Lost Customers Scan (full master data)
            lost_customers_rule = self._create_lost_customers_rule()
            lost_customers_results = self.event_scanner.scan_for_events([lost_customers_rule], scan_config)
            
            if lost_customers_results:
                scan_results['lost_customers'] = lost_customers_results[0]
                self._update_master_event_file('lost_customers', lost_customers_results[0])
            
            # 2. Unsold Estimates Scan (full master data - to update existing master event file)
            unsold_estimates_rule = self._create_unsold_estimates_rule()
            unsold_estimates_results = self.event_scanner.scan_for_events([unsold_estimates_rule], scan_config)
            
            if unsold_estimates_results:
                scan_results['unsold_estimates'] = unsold_estimates_results[0]
                self._update_master_event_file('unsold_estimates', unsold_estimates_results[0])
            
            # 3. Overdue Maintenance Scan (if not excluded)
            if not exclude_aging_systems:
                overdue_maintenance_rule = self._create_overdue_maintenance_rule()
                maintenance_results = self.event_scanner.scan_for_events([overdue_maintenance_rule], scan_config)
                
                if maintenance_results:
                    scan_results['overdue_maintenance'] = maintenance_results[0]
                    self._update_master_event_file('overdue_maintenance', maintenance_results[0])
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            
            # Calculate totals
            total_events = sum(result.total_events for result in scan_results.values())
            
            return {
                "full_scan_events_detected": total_events,
                "processing_time_ms": duration_ms,
                "scan_results": {
                    event_type: {
                        "events_found": result.total_events,
                        "entities_examined": result.total_entities_examined,
                        "master_file_updated": True
                    } for event_type, result in scan_results.items()
                },
                "master_files_updated": list(scan_results.keys())
            }
            
        except Exception as e:
            self.logger.log_event("error", f"Full scan event processing failed: {e}")
            return {"full_scan_events_detected": 0, "error": str(e)}
    
    def _detect_job_cancellations(self, job_changes: List[AuditChange]) -> List[EventResult]:
        """Detect job cancellations from audit changes"""
        
        events = []
        
        for change in job_changes:
            # Check for status changes to canceled
            if 'status' in change.column.lower():
                old_status = change.old_value.lower()
                new_status = change.new_value.lower()
                
                if 'cancel' in new_status and 'cancel' not in old_status:
                    # Create enhanced cancellation event
                    event = self._create_enhanced_cancellation_event(change)
                    events.append(event)
        
        return events
    
    def _detect_unsold_estimates_from_changes(self, estimate_changes: List[AuditChange]) -> List[EventResult]:
        """Detect unsold estimates from audit changes"""
        
        events = []
        
        for change in estimate_changes:
            # Check for status changes that indicate unsold
            if 'status' in change.column.lower():
                old_status = change.old_value.lower()
                new_status = change.new_value.lower()
                
                # Detect unsold statuses
                unsold_statuses = ['dismissed', 'expired', 'rejected', 'declined']
                
                if any(status in new_status for status in unsold_statuses) and not any(status in old_status for status in unsold_statuses):
                    event = self._create_unsold_estimate_event(change)
                    events.append(event)
        
        return events
    
    def _create_enhanced_cancellation_event(self, change: AuditChange) -> EventResult:
        """Create enhanced job cancellation event with business context"""
        
        job_id = change.id_value
        
        # Get additional context
        context = self._get_job_context(job_id)
        
        return EventResult(
            event_type="job_canceled",
            entity_type="job",
            entity_id=job_id,
            severity=self._determine_cancellation_severity(context),
            detected_at=datetime.now(UTC),
            details={
                "old_status": change.old_value,
                "new_status": change.new_value,
                "status_field": change.column,
                "source": "realtime_upsert_detection",
                "is_new_job": change.old_value == "",
                **context
            },
            rule_name="Real-time Job Cancellation Detection"
        )
    
    def _create_unsold_estimate_event(self, change: AuditChange) -> EventResult:
        """Create unsold estimate event"""
        
        estimate_id = change.id_value
        
        # Get estimate context
        context = self._get_estimate_context(estimate_id)
        
        return EventResult(
            event_type="unsold_estimate",
            entity_type="estimate",
            entity_id=estimate_id,
            severity=EventSeverity.MEDIUM,
            detected_at=datetime.now(UTC),
            details={
                "old_status": change.old_value,
                "new_status": change.new_value,
                "status_field": change.column,
                "source": "realtime_upsert_detection",
                **context
            },
            rule_name="Real-time Unsold Estimate Detection"
        )
    
    def _get_job_context(self, job_id: str) -> Dict[str, Any]:
        """Get business context for a job"""
        
        try:
            jobs_df = self.event_scanner.load_master_table("jobs")
            if jobs_df is None:
                return {"context": "jobs_table_not_available"}
            
            # Find the job
            job_matches = jobs_df[jobs_df['Job ID'].astype(str) == str(job_id)]
            if job_matches.empty:
                return {"context": "job_not_found"}
            
            job_row = job_matches.iloc[0]
            
            # Extract context
            customer_id = str(job_row.get('Customer ID', ''))
            job_date = str(job_row.get('Job Date', ''))
            
            # Find related jobs for this customer
            if customer_id:
                related_jobs = self._find_customer_jobs_in_timeframe(jobs_df, customer_id, job_date, 30)
                
                return {
                    "customer_id": customer_id,
                    "job_date": job_date,
                    "related_jobs_count": len(related_jobs),
                    "has_active_jobs_nearby": any(
                        'cancel' not in job.get('status', '').lower() and 
                        'complete' not in job.get('status', '').lower()
                        for job in related_jobs
                    )
                }
            
            return {"customer_id": customer_id, "job_date": job_date}
            
        except Exception as e:
            return {"context": f"error: {str(e)}"}
    
    def _get_estimate_context(self, estimate_id: str) -> Dict[str, Any]:
        """Get business context for an estimate"""
        
        try:
            estimates_df = self.event_scanner.load_master_table("estimates")
            if estimates_df is None:
                return {"context": "estimates_table_not_available"}
            
            # Find the estimate
            estimate_matches = estimates_df[estimates_df['Estimate ID'].astype(str) == str(estimate_id)]
            if estimate_matches.empty:
                return {"context": "estimate_not_found"}
            
            estimate_row = estimate_matches.iloc[0]
            
            # Extract context
            customer_id = str(estimate_row.get('Customer ID', ''))
            estimate_total = str(estimate_row.get('Estimates Subtotal', ''))
            created_date = str(estimate_row.get('Created Date', ''))
            
            return {
                "customer_id": customer_id,
                "estimate_total": estimate_total,
                "created_date": created_date
            }
            
        except Exception as e:
            return {"context": f"error: {str(e)}"}
    
    def _find_customer_jobs_in_timeframe(self, jobs_df: pd.DataFrame, customer_id: str, 
                                        reference_date: str, days: int) -> List[Dict[str, Any]]:
        """Find customer jobs within a timeframe"""
        
        try:
            # Parse reference date
            ref_datetime = pd.to_datetime(reference_date, errors='coerce')
            if pd.isna(ref_datetime):
                return []
            
            # Filter to customer jobs
            customer_jobs = jobs_df[jobs_df['Customer ID'].astype(str) == str(customer_id)]
            
            if customer_jobs.empty:
                return []
            
            # Parse job dates and filter to timeframe
            customer_jobs = customer_jobs.copy()
            customer_jobs['parsed_date'] = pd.to_datetime(customer_jobs['Job Date'], errors='coerce')
            
            from datetime import timedelta
            start_date = ref_datetime - timedelta(days=days)
            end_date = ref_datetime + timedelta(days=days)
            
            nearby_jobs = customer_jobs[
                (customer_jobs['parsed_date'] >= start_date) &
                (customer_jobs['parsed_date'] <= end_date) &
                customer_jobs['parsed_date'].notna()
            ]
            
            # Convert to list
            jobs_list = []
            for _, row in nearby_jobs.iterrows():
                jobs_list.append({
                    'job_id': str(row.get('Job ID', '')),
                    'status': str(row.get('Status', '')),
                    'date': str(row.get('Job Date', ''))
                })
            
            return jobs_list
            
        except Exception:
            return []
    
    def _determine_cancellation_severity(self, context: Dict[str, Any]) -> EventSeverity:
        """Determine cancellation severity based on business context"""
        
        if context.get("has_active_jobs_nearby", False):
            return EventSeverity.HIGH
        elif context.get("related_jobs_count", 0) > 0:
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.LOW
    
    def _save_realtime_events(self, events: List[EventResult]):
        """Save real-time events to master event files"""
        
        # Group events by type
        events_by_type = {}
        for event in events:
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = []
            events_by_type[event.event_type].append(event)
        
        # Save each event type
        for event_type, event_list in events_by_type.items():
            self.master_storage.save_event_results(
                event_type=event_type,
                events=event_list,
                scan_timestamp=datetime.now(UTC),
                update_mode="upsert"
            )
    
    def _create_lost_customers_rule(self) -> EventRule:
        """Create rule for lost customers detection"""
        
        return EventRule(
            name="Lost Customers Detection",
            description="Detect customers using competitors through permit data analysis",
            event_type="lost_customers",
            rule_type=RuleType.CROSS_TABLE,
            target_tables=["customers", "locations", "calls"],
            detection_logic={
                "permit_analysis": True,
                "call_history_analysis": True,
                "contractor_timeline": True
            },
            output_fields=["entity_id", "entity_type", "detected_at", "details"],
            severity=EventSeverity.HIGH,
            enabled=True
        )
    
    def _create_unsold_estimates_rule(self) -> EventRule:
        """Create rule for unsold estimates detection"""
        
        return EventRule(
            name="Unsold Estimates Detection", 
            description="Detect estimates that were dismissed or expired",
            event_type="unsold_estimates",
            rule_type=RuleType.SINGLE_TABLE,
            target_tables=["estimates"],
            detection_logic={
                "include_statuses": ["Dismissed", "Open", "Expired"],
                "exclude_substrings": ["This is an empty"]
            },
            output_fields=["entity_id", "entity_type", "detected_at", "details"],
            severity=EventSeverity.MEDIUM,
            enabled=True
        )
    
    def _create_overdue_maintenance_rule(self) -> EventRule:
        """Create rule for overdue maintenance detection"""
        
        return EventRule(
            name="Overdue Maintenance Detection",
            description="Detect locations and customers with overdue maintenance",
            event_type="overdue_maintenance",
            rule_type=RuleType.CROSS_TABLE,
            target_tables=["jobs", "locations", "customers"],
            detection_logic={
                "maintenance_criteria": {
                    "job_class_values": ["maintenance"],
                    "job_type_values": ["maintenance"]
                },
                "threshold": {"default_months": 12}
            },
            output_fields=["entity_id", "entity_type", "detected_at", "details"],
            severity=EventSeverity.MEDIUM,
            enabled=True,
            threshold_months=12
        )
    
    def _update_master_event_file(self, event_type: str, scan_result: EventScanResult):
        """Update master event file and track changes"""
        
        try:
            # Get the master event file path
            event_file = self.data_dir / "events" / "master_files" / f"{event_type}_master.parquet"
            
            # Load existing events if file exists
            existing_events = []
            if event_file.exists():
                try:
                    existing_df = pd.read_parquet(event_file)
                    existing_events = existing_df.to_dict('records')
                except Exception:
                    existing_events = []
            
            # Convert new events to records
            new_events = []
            for event in scan_result.events:
                new_events.append({
                    "event_type": event.event_type,
                    "entity_type": event.entity_type,
                    "entity_id": event.entity_id,
                    "severity": event.severity.value,
                    "detected_at": event.detected_at.isoformat(),
                    "details": str(event.details),
                    "rule_name": event.rule_name
                })
            
            # Track changes
            changes_tracked = self._track_event_file_changes(event_type, existing_events, new_events)
            
            # Save updated events
            if new_events:
                updated_df = pd.DataFrame(new_events)
                event_file.parent.mkdir(parents=True, exist_ok=True)
                updated_df.to_parquet(event_file, index=False)
            
            self.logger.log_event("info", f"Updated master event file: {event_type}", {
                "existing_events": len(existing_events),
                "new_events": len(new_events),
                "changes_tracked": changes_tracked,
                "file_path": str(event_file)
            })
            
        except Exception as e:
            self.logger.log_event("error", f"Failed to update master event file {event_type}: {e}")
    
    def _track_event_file_changes(self, event_type: str, existing_events: List[Dict], 
                                 new_events: List[Dict]) -> Dict[str, Any]:
        """Track changes to master event files"""
        
        try:
            # Create tracking log
            changes_log_file = self.data_dir / "logs" / f"{event_type}_master_changes.jsonl"
            changes_log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Calculate changes
            existing_ids = set(event.get('entity_id', '') for event in existing_events)
            new_ids = set(event.get('entity_id', '') for event in new_events)
            
            added_ids = new_ids - existing_ids
            removed_ids = existing_ids - new_ids
            updated_ids = existing_ids & new_ids
            
            # Log the changes
            change_summary = {
                "timestamp": datetime.now(UTC).isoformat(),
                "event_type": event_type,
                "existing_count": len(existing_events),
                "new_count": len(new_events),
                "added_count": len(added_ids),
                "removed_count": len(removed_ids),
                "updated_count": len(updated_ids),
                "added_ids": list(added_ids)[:10],  # Sample
                "removed_ids": list(removed_ids)[:10]  # Sample
            }
            
            # Save to log
            with open(changes_log_file, 'a', encoding='utf-8') as f:
                import json
                f.write(json.dumps(change_summary) + '\n')
            
            return {
                "added": len(added_ids),
                "removed": len(removed_ids),
                "updated": len(updated_ids)
            }
            
        except Exception as e:
            return {"error": str(e)}
