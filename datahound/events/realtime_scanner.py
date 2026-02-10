"""
Real-time event scanner based on change logs from upsert operations
"""

import json
import time
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import pandas as pd

from .types import EventResult, EventSeverity, EventRule, EventScanResult, EventScanConfig
from .logging import EventLogger
from .event_storage import EventMasterStorage
from ..upsert.types import AuditChange


class ChangeLogEventScanner:
    """Scanner that detects events from upsert change logs instead of master data"""
    
    def __init__(self, company: str, data_dir: Path):
        self.company = company
        self.data_dir = data_dir
        
        # Initialize components
        self.logger = EventLogger(company, data_dir)
        self.master_storage = EventMasterStorage(company, data_dir)
        
        # Change log files
        self.change_logs_dir = data_dir / "logs"
        self.apply_log = self.change_logs_dir / "apply_log.jsonl"
        
    def scan_from_recent_changes(self, hours_back: int = 24, 
                                processing_limit: Optional[int] = None) -> Dict[str, EventScanResult]:
        """Scan for events based on recent changes in the last N hours"""
        
        since_time = datetime.now(UTC) - timedelta(hours=hours_back)
        
        # Load recent changes from apply log
        recent_changes = self._load_recent_changes(since_time)
        
        if not recent_changes:
            self.logger.log_event("info", "No recent changes found for event detection", {
                "hours_back": hours_back,
                "since_time": since_time.isoformat()
            })
            return {}
        
        # Group changes by type
        changes_by_type = self._group_changes_by_type(recent_changes)
        
        # Detect events for each change type
        scan_results = {}
        
        for file_type, changes in changes_by_type.items():
            if processing_limit and len(changes) > processing_limit:
                changes = changes[:processing_limit]
            
            events = self._detect_events_from_changes(file_type, changes)
            
            if events:
                # Save events to master storage
                self.master_storage.save_event_results(
                    event_type=f"{file_type}_changes",
                    events=events,
                    scan_timestamp=datetime.now(UTC),
                    update_mode="upsert"
                )
                
                # Create scan result
                scan_results[file_type] = EventScanResult(
                    rule_name=f"Change Log {file_type.title()}",
                    total_events=len(events),
                    events_by_severity={EventSeverity.LOW: len(events)},
                    events=events,
                    scan_duration_ms=0,
                    tables_scanned=[file_type],
                    config_used=EventScanConfig(),
                    total_entities_examined=len(changes),
                    entities_processed=len(events),
                    processing_limit_applied=processing_limit is not None and len(recent_changes) > processing_limit
                )
        
        return scan_results
    
    def _load_recent_changes(self, since_time: datetime) -> List[Dict[str, Any]]:
        """Load recent changes from apply log"""
        
        if not self.apply_log.exists():
            return []
        
        recent_changes = []
        
        try:
            with open(self.apply_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Parse timestamp
                        ts_str = entry.get('ts', '')
                        if ts_str:
                            entry_time = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                            
                            if entry_time >= since_time:
                                recent_changes.append(entry)
                    except:
                        continue
        except Exception as e:
            self.logger.log_event("error", f"Failed to load recent changes: {e}")
        
        return recent_changes
    
    def _group_changes_by_type(self, changes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group changes by file type"""
        
        grouped = {}
        
        for change in changes:
            file_type = change.get('file_type', 'unknown')
            if file_type not in grouped:
                grouped[file_type] = []
            grouped[file_type].append(change)
        
        return grouped
    
    def _detect_events_from_changes(self, file_type: str, changes: List[Dict[str, Any]]) -> List[EventResult]:
        """Detect events from changes for a specific file type"""
        
        events = []
        
        for change in changes:
            # Extract change information
            change_info = {
                'file_type': file_type,
                'timestamp': change.get('ts', ''),
                'new_rows': change.get('new_rows', 0),
                'updated_rows': change.get('updated_rows', 0),
                'audit_changes': change.get('audit_changes', 0)
            }
            
            # Focus on new record events for activity data (jobs, calls, estimates)
            if change.get('new_rows', 0) > 0:
                if file_type in ['jobs', 'calls', 'estimates', 'invoices']:
                    # For activity data, analyze the new records for business events
                    events.extend(self._analyze_new_activity_records(file_type, change))
                else:
                    # For other data, create standard new record events
                    events.extend(self._create_new_record_events(file_type, change))
            
            # Update events (mainly for customers)
            if change.get('audit_changes', 0) > 0:
                events.extend(self._create_update_events(file_type, change))
        
        return events
    
    def _analyze_new_activity_records(self, file_type: str, change: Dict[str, Any]) -> List[EventResult]:
        """Analyze new activity records for business events"""
        
        try:
            # Load the prepared file to analyze new records
            prepared_file = change.get('prepared', '')
            if not prepared_file:
                return []
            
            # Find the prepared file
            downloads_dir = Path(f"data/{self.company}/downloads")
            prep_files = list(downloads_dir.glob(f"*{prepared_file}*"))
            
            if not prep_files:
                return []
            
            prep_df = pd.read_parquet(prep_files[0])
            
            # Analyze new records based on file type
            if file_type == 'jobs':
                return self._analyze_new_jobs(prep_df, change)
            elif file_type == 'calls':
                return self._analyze_new_calls(prep_df, change)
            elif file_type == 'estimates':
                return self._analyze_new_estimates(prep_df, change)
            elif file_type == 'invoices':
                return self._analyze_new_invoices(prep_df, change)
            
        except Exception as e:
            # Fallback to basic new record event
            return self._create_new_record_events(file_type, change)
        
        return []
    
    def _analyze_new_jobs(self, jobs_df: pd.DataFrame, change: Dict[str, Any]) -> List[EventResult]:
        """Analyze new jobs for business events"""
        
        events = []
        
        # Check for canceled jobs
        if 'Status' in jobs_df.columns:
            canceled_jobs = jobs_df[jobs_df['Status'].astype(str).str.lower().str.contains('cancel', na=False)]
            
            for _, job in canceled_jobs.iterrows():
                job_id = str(job.get('Job ID', ''))
                if job_id:
                    event = EventResult(
                        event_type="new_job_canceled",
                        entity_type="job",
                        entity_id=job_id,
                        severity=EventSeverity.HIGH,
                        detected_at=datetime.now(UTC),
                        details={
                            "status": str(job.get('Status', '')),
                            "customer_id": str(job.get('Customer ID', '')),
                            "job_date": str(job.get('Job Date', '')),
                            "source": "new_record_analysis",
                            "is_new_canceled_job": True
                        },
                        rule_name="New Canceled Job Detection"
                    )
                    events.append(event)
        
        # Check for completed jobs
        if 'Status' in jobs_df.columns:
            completed_jobs = jobs_df[jobs_df['Status'].astype(str).str.lower().str.contains('complete', na=False)]
            
            for _, job in completed_jobs.iterrows():
                job_id = str(job.get('Job ID', ''))
                if job_id:
                    event = EventResult(
                        event_type="new_job_completed",
                        entity_type="job",
                        entity_id=job_id,
                        severity=EventSeverity.LOW,
                        detected_at=datetime.now(UTC),
                        details={
                            "status": str(job.get('Status', '')),
                            "customer_id": str(job.get('Customer ID', '')),
                            "job_date": str(job.get('Job Date', '')),
                            "source": "new_record_analysis"
                        },
                        rule_name="New Completed Job Detection"
                    )
                    events.append(event)
        
        return events
    
    def _analyze_new_calls(self, calls_df: pd.DataFrame, change: Dict[str, Any]) -> List[EventResult]:
        """Analyze new calls for business events"""
        
        events = []
        
        # All new calls are potential customer contact events
        for _, call in calls_df.head(10).iterrows():  # Limit to first 10 for performance
            call_id = str(call.get('Call ID', ''))
            customer_id = str(call.get('Customer ID', ''))
            
            if call_id and customer_id:
                event = EventResult(
                    event_type="new_customer_contact",
                    entity_type="call",
                    entity_id=call_id,
                    severity=EventSeverity.LOW,
                    detected_at=datetime.now(UTC),
                    details={
                        "customer_id": customer_id,
                        "call_date": str(call.get('Call Date', '')),
                        "call_time": str(call.get('Call Time', '')),
                        "direction": str(call.get('Direction', '')),
                        "source": "new_record_analysis"
                    },
                    rule_name="New Customer Contact Detection"
                )
                events.append(event)
        
        return events
    
    def _analyze_new_estimates(self, estimates_df: pd.DataFrame, change: Dict[str, Any]) -> List[EventResult]:
        """Analyze new estimates for business events"""
        
        events = []
        
        # Check estimate status
        if 'Estimate Status' in estimates_df.columns:
            # Look for dismissed estimates
            dismissed = estimates_df[estimates_df['Estimate Status'].astype(str).str.lower().str.contains('dismiss', na=False)]
            
            for _, est in dismissed.iterrows():
                est_id = str(est.get('Estimate ID', ''))
                if est_id:
                    event = EventResult(
                        event_type="new_estimate_dismissed",
                        entity_type="estimate",
                        entity_id=est_id,
                        severity=EventSeverity.MEDIUM,
                        detected_at=datetime.now(UTC),
                        details={
                            "status": str(est.get('Estimate Status', '')),
                            "customer_id": str(est.get('Customer ID', '')),
                            "source": "new_record_analysis"
                        },
                        rule_name="New Dismissed Estimate Detection"
                    )
                    events.append(event)
        
        return events
    
    def _analyze_new_invoices(self, invoices_df: pd.DataFrame, change: Dict[str, Any]) -> List[EventResult]:
        """Analyze new invoices for business events"""
        
        events = []
        
        # New invoices represent completed work
        for _, invoice in invoices_df.head(5).iterrows():  # Sample first 5
            invoice_id = str(invoice.get('Invoice ID', ''))
            customer_id = str(invoice.get('Customer ID', ''))
            
            if invoice_id and customer_id:
                event = EventResult(
                    event_type="new_invoice_created",
                    entity_type="invoice",
                    entity_id=invoice_id,
                    severity=EventSeverity.LOW,
                    detected_at=datetime.now(UTC),
                    details={
                        "customer_id": customer_id,
                        "total": str(invoice.get('Total', '')),
                        "invoice_date": str(invoice.get('Invoice Date', '')),
                        "source": "new_record_analysis"
                    },
                    rule_name="New Invoice Detection"
                )
                events.append(event)
        
        return events
    
    def _create_new_record_events(self, file_type: str, change: Dict[str, Any]) -> List[EventResult]:
        """Create events for new records"""
        
        events = []
        new_count = change.get('new_rows', 0)
        
        if new_count > 0:
            event = EventResult(
                event_type=f"new_{file_type}",
                entity_type=file_type.rstrip('s'),  # "jobs" -> "job"
                entity_id="bulk_insert",
                severity=EventSeverity.LOW,
                detected_at=datetime.now(UTC),
                details={
                    "new_records_count": new_count,
                    "file_type": file_type,
                    "source": "upsert_operation",
                    "timestamp": change.get('ts', ''),
                    "prepared_file": change.get('prepared', '')
                },
                rule_name=f"New {file_type.title()} Detection"
            )
            events.append(event)
        
        return events
    
    def _create_update_events(self, file_type: str, change: Dict[str, Any]) -> List[EventResult]:
        """Create events for record updates"""
        
        events = []
        update_count = change.get('audit_changes', 0)
        
        if update_count > 0:
            # Determine severity based on number of changes
            if update_count > 1000:
                severity = EventSeverity.HIGH
            elif update_count > 100:
                severity = EventSeverity.MEDIUM
            else:
                severity = EventSeverity.LOW
            
            event = EventResult(
                event_type=f"updated_{file_type}",
                entity_type=file_type.rstrip('s'),
                entity_id="bulk_update",
                severity=severity,
                detected_at=datetime.now(UTC),
                details={
                    "updated_records_count": change.get('updated_rows', 0),
                    "total_changes": update_count,
                    "file_type": file_type,
                    "source": "upsert_operation",
                    "timestamp": change.get('ts', ''),
                    "prepared_file": change.get('prepared', '')
                },
                rule_name=f"Updated {file_type.title()} Detection"
            )
            events.append(event)
        
        return events


class RealtimeEventProcessor:
    """Lightweight real-time event processor for immediate event detection"""
    
    def __init__(self, company: str, data_dir: Path):
        self.company = company
        self.data_dir = data_dir
        self.scanner = ChangeLogEventScanner(company, data_dir)
        self.logger = EventLogger(company, data_dir)
        
    def process_upsert_events(self, upsert_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process events immediately after upsert operations"""
        
        start_time = time.perf_counter()
        
        try:
            # Scan for events from recent changes (last hour)
            scan_results = self.scanner.scan_from_recent_changes(hours_back=1)
            
            # Aggregate results
            total_events = sum(result.total_events for result in scan_results.values())
            total_processing_time = sum(result.scan_duration_ms for result in scan_results.values())
            
            # Create summary
            processing_summary = {
                "total_events_detected": total_events,
                "event_types_processed": list(scan_results.keys()),
                "total_processing_time_ms": total_processing_time,
                "scan_results": {
                    event_type: {
                        "events_found": result.total_events,
                        "entities_examined": result.total_entities_examined,
                        "entities_processed": result.entities_processed
                    } for event_type, result in scan_results.items()
                }
            }
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            
            self.logger.log_event("info", f"Real-time event processing completed", {
                "events_detected": total_events,
                "processing_duration_ms": duration_ms,
                **processing_summary
            })
            
            return processing_summary
            
        except Exception as e:
            self.logger.log_event("error", f"Real-time event processing failed: {e}")
            return {
                "total_events_detected": 0,
                "error": str(e)
            }


class ChangeLogAnalyzer:
    """Analyzes change logs to identify specific business events"""
    
    def __init__(self, company: str, data_dir: Path):
        self.company = company
        self.data_dir = data_dir
        
    def analyze_customer_changes(self, changes: List[AuditChange]) -> List[EventResult]:
        """Analyze customer changes for business events"""
        
        events = []
        
        # Group changes by customer
        customer_changes = {}
        for change in changes:
            customer_id = change.id_value
            if customer_id not in customer_changes:
                customer_changes[customer_id] = []
            customer_changes[customer_id].append(change)
        
        # Analyze each customer's changes
        for customer_id, customer_change_list in customer_changes.items():
            # Detect significant changes
            contact_updates = [c for c in customer_change_list if 'phone' in c.column.lower() or 'email' in c.column.lower()]
            address_updates = [c for c in customer_change_list if 'address' in c.column.lower()]
            status_updates = [c for c in customer_change_list if 'status' in c.column.lower()]
            
            # Create events for significant changes
            if contact_updates:
                events.append(self._create_contact_update_event(customer_id, contact_updates))
            
            if address_updates:
                events.append(self._create_address_update_event(customer_id, address_updates))
            
            if status_updates:
                events.append(self._create_status_update_event(customer_id, status_updates))
        
        return events
    
    def analyze_job_changes(self, changes: List[AuditChange]) -> List[EventResult]:
        """Analyze job changes for business events with enhanced business logic"""
        
        events = []
        
        # Group changes by job
        job_changes = {}
        for change in changes:
            job_id = change.id_value
            if job_id not in job_changes:
                job_changes[job_id] = []
            job_changes[job_id].append(change)
        
        # Analyze each job's changes
        for job_id, job_change_list in job_changes.items():
            # Detect status changes
            status_changes = [c for c in job_change_list if 'status' in c.column.lower()]
            
            for status_change in status_changes:
                old_status = status_change.old_value.lower()
                new_status = status_change.new_value.lower()
                
                # Enhanced status change detection
                if 'cancel' in new_status and 'cancel' not in old_status:
                    # Create enhanced cancellation event with additional business logic
                    enhanced_event = self._create_enhanced_job_canceled_event(job_id, status_change)
                    events.append(enhanced_event)
                elif 'complete' in new_status and 'complete' not in old_status:
                    events.append(self._create_job_completed_event(job_id, status_change))
                elif 'hold' in new_status and 'hold' not in old_status:
                    events.append(self._create_job_on_hold_event(job_id, status_change))
                elif old_status != new_status and old_status != "":
                    # Detect any status change (not just specific ones)
                    events.append(self._create_job_status_change_event(job_id, status_change))
        
        return events
    
    def analyze_estimate_changes(self, changes: List[AuditChange]) -> List[EventResult]:
        """Analyze estimate changes for business events"""
        
        events = []
        
        # Group changes by estimate
        estimate_changes = {}
        for change in changes:
            estimate_id = change.id_value
            if estimate_id not in estimate_changes:
                estimate_changes[estimate_id] = []
            estimate_changes[estimate_id].append(change)
        
        # Analyze each estimate's changes
        for estimate_id, estimate_change_list in estimate_changes.items():
            status_changes = [c for c in estimate_change_list if 'status' in c.column.lower()]
            
            for status_change in status_changes:
                old_status = status_change.old_value.lower()
                new_status = status_change.new_value.lower()
                
                # Detect estimate events
                if 'sold' in new_status and 'sold' not in old_status:
                    events.append(self._create_estimate_sold_event(estimate_id, status_change))
                elif 'dismiss' in new_status and 'dismiss' not in old_status:
                    events.append(self._create_estimate_dismissed_event(estimate_id, status_change))
                elif 'expired' in new_status and 'expired' not in old_status:
                    events.append(self._create_estimate_expired_event(estimate_id, status_change))
        
        return events
    
    def _create_contact_update_event(self, customer_id: str, changes: List[AuditChange]) -> EventResult:
        """Create event for customer contact information updates"""
        
        return EventResult(
            event_type="customer_contact_updated",
            entity_type="customer",
            entity_id=customer_id,
            severity=EventSeverity.LOW,
            detected_at=datetime.now(UTC),
            details={
                "updated_fields": [c.column for c in changes],
                "changes_count": len(changes),
                "source": "change_log_analysis"
            },
            rule_name="Customer Contact Update Detection"
        )
    
    def _create_address_update_event(self, customer_id: str, changes: List[AuditChange]) -> EventResult:
        """Create event for customer address updates"""
        
        return EventResult(
            event_type="customer_address_updated",
            entity_type="customer", 
            entity_id=customer_id,
            severity=EventSeverity.MEDIUM,
            detected_at=datetime.now(UTC),
            details={
                "updated_fields": [c.column for c in changes],
                "old_addresses": [c.old_value for c in changes],
                "new_addresses": [c.new_value for c in changes],
                "source": "change_log_analysis"
            },
            rule_name="Customer Address Update Detection"
        )
    
    def _create_status_update_event(self, customer_id: str, changes: List[AuditChange]) -> EventResult:
        """Create event for customer status updates"""
        
        return EventResult(
            event_type="customer_status_updated",
            entity_type="customer",
            entity_id=customer_id,
            severity=EventSeverity.MEDIUM,
            detected_at=datetime.now(UTC),
            details={
                "status_changes": [{"field": c.column, "old": c.old_value, "new": c.new_value} for c in changes],
                "source": "change_log_analysis"
            },
            rule_name="Customer Status Update Detection"
        )
    
    def _create_enhanced_job_canceled_event(self, job_id: str, change: AuditChange) -> EventResult:
        """Create enhanced event for job cancellation with business logic"""
        
        # Load jobs data to check for related jobs
        additional_context = self._analyze_cancellation_context(job_id)
        
        # Determine severity based on context
        severity = self._determine_cancellation_severity(additional_context)
        
        return EventResult(
            event_type="job_canceled",
            entity_type="job", 
            entity_id=job_id,
            severity=severity,
            detected_at=datetime.now(UTC),
            details={
                "old_status": change.old_value,
                "new_status": change.new_value,
                "status_field": change.column,
                "source": "change_log_analysis",
                "is_new_job": change.old_value == "",
                **additional_context
            },
            rule_name="Enhanced Job Cancellation Detection"
        )
    
    def _analyze_cancellation_context(self, job_id: str) -> Dict[str, Any]:
        """Analyze context around job cancellation for business intelligence"""
        
        try:
            # Load jobs data to find related information
            jobs_file = Path(f"companies/{self.company}/parquet/Jobs.parquet")
            
            if not jobs_file.exists():
                return {"context_analysis": "jobs_file_not_found"}
            
            jobs_df = pd.read_parquet(jobs_file)
            
            # Find the canceled job
            job_row = None
            job_id_cols = ['Job ID', 'ID', 'job_id']
            
            for col in job_id_cols:
                if col in jobs_df.columns:
                    job_matches = jobs_df[jobs_df[col].astype(str) == str(job_id)]
                    if not job_matches.empty:
                        job_row = job_matches.iloc[0]
                        break
            
            if job_row is None:
                return {"context_analysis": "canceled_job_not_found_in_master"}
            
            # Get customer ID and job date
            customer_id = self._extract_customer_id_from_job(job_row)
            job_date = self._extract_job_date(job_row)
            
            context = {
                "customer_id": customer_id,
                "job_date": job_date,
                "context_analysis": "basic_info_extracted"
            }
            
            # Check for other jobs within 30 days for the same customer
            if customer_id and job_date:
                related_jobs = self._find_related_jobs_within_30_days(
                    jobs_df, customer_id, job_date, exclude_job_id=job_id
                )
                
                context.update({
                    "related_jobs_count": len(related_jobs),
                    "related_jobs_statuses": [job.get('status', 'unknown') for job in related_jobs],
                    "has_active_jobs_nearby": any(
                        status.lower() not in ['canceled', 'cancelled', 'complete', 'completed'] 
                        for status in [job.get('status', '') for job in related_jobs]
                    )
                })
            
            return context
            
        except Exception as e:
            return {"context_analysis": f"error: {str(e)}"}
    
    def _extract_customer_id_from_job(self, job_row: pd.Series) -> Optional[str]:
        """Extract customer ID from job row"""
        
        customer_id_cols = ['Customer ID', 'customer_id', 'CustomerID']
        
        for col in customer_id_cols:
            if col in job_row.index and pd.notna(job_row[col]):
                return str(job_row[col])
        
        return None
    
    def _extract_job_date(self, job_row: pd.Series) -> Optional[str]:
        """Extract job date from job row"""
        
        date_cols = ['Job Date', 'Created Date', 'Scheduled Date', 'Start Date']
        
        for col in date_cols:
            if col in job_row.index and pd.notna(job_row[col]):
                return str(job_row[col])
        
        return None
    
    def _find_related_jobs_within_30_days(self, jobs_df: pd.DataFrame, customer_id: str, 
                                         job_date: str, exclude_job_id: str) -> List[Dict[str, Any]]:
        """Find other jobs for the same customer within 30 days"""
        
        try:
            from datetime import datetime, timedelta
            
            # Parse the job date
            job_datetime = pd.to_datetime(job_date, errors='coerce')
            if pd.isna(job_datetime):
                return []
            
            # Define 30-day window
            start_date = job_datetime - timedelta(days=30)
            end_date = job_datetime + timedelta(days=30)
            
            # Find customer ID column
            customer_col = None
            for col in ['Customer ID', 'customer_id', 'CustomerID']:
                if col in jobs_df.columns:
                    customer_col = col
                    break
            
            if not customer_col:
                return []
            
            # Find job ID column
            job_id_col = None
            for col in ['Job ID', 'ID', 'job_id']:
                if col in jobs_df.columns:
                    job_id_col = col
                    break
            
            if not job_id_col:
                return []
            
            # Find date column
            date_col = None
            for col in ['Job Date', 'Created Date', 'Scheduled Date', 'Start Date']:
                if col in jobs_df.columns:
                    date_col = col
                    break
            
            if not date_col:
                return []
            
            # Filter jobs for same customer, excluding the canceled job
            customer_jobs = jobs_df[
                (jobs_df[customer_col].astype(str) == str(customer_id)) &
                (jobs_df[job_id_col].astype(str) != str(exclude_job_id))
            ].copy()
            
            if customer_jobs.empty:
                return []
            
            # Parse job dates and filter to 30-day window
            customer_jobs['parsed_date'] = pd.to_datetime(customer_jobs[date_col], errors='coerce')
            
            nearby_jobs = customer_jobs[
                (customer_jobs['parsed_date'] >= start_date) &
                (customer_jobs['parsed_date'] <= end_date) &
                customer_jobs['parsed_date'].notna()
            ]
            
            # Convert to list of dictionaries
            related_jobs = []
            for _, row in nearby_jobs.iterrows():
                related_jobs.append({
                    'job_id': str(row[job_id_col]),
                    'status': str(row.get('Status', row.get('Job Status', ''))),
                    'date': str(row[date_col]),
                    'days_from_canceled': (job_datetime - row['parsed_date']).days
                })
            
            return related_jobs
            
        except Exception as e:
            return []
    
    def _determine_cancellation_severity(self, context: Dict[str, Any]) -> EventSeverity:
        """Determine cancellation event severity based on business context"""
        
        # High severity if customer has other active jobs nearby (potential pattern)
        if context.get("has_active_jobs_nearby", False):
            return EventSeverity.HIGH
        
        # Medium severity if customer has other jobs in the timeframe
        if context.get("related_jobs_count", 0) > 0:
            return EventSeverity.MEDIUM
        
        # Low severity for isolated cancellations
        return EventSeverity.LOW
    
    def _create_job_canceled_event(self, job_id: str, change: AuditChange) -> EventResult:
        """Create basic event for job cancellation (legacy method)"""
        
        return EventResult(
            event_type="job_canceled",
            entity_type="job",
            entity_id=job_id,
            severity=EventSeverity.HIGH,
            detected_at=datetime.now(UTC),
            details={
                "old_status": change.old_value,
                "new_status": change.new_value,
                "status_field": change.column,
                "source": "change_log_analysis"
            },
            rule_name="Job Cancellation Detection"
        )
    
    def _create_job_completed_event(self, job_id: str, change: AuditChange) -> EventResult:
        """Create event for job completion"""
        
        return EventResult(
            event_type="job_completed",
            entity_type="job",
            entity_id=job_id,
            severity=EventSeverity.LOW,
            detected_at=datetime.now(UTC),
            details={
                "old_status": change.old_value,
                "new_status": change.new_value,
                "status_field": change.column,
                "source": "change_log_analysis"
            },
            rule_name="Job Completion Detection"
        )
    
    def _create_job_on_hold_event(self, job_id: str, change: AuditChange) -> EventResult:
        """Create event for job put on hold"""
        
        return EventResult(
            event_type="job_on_hold",
            entity_type="job",
            entity_id=job_id,
            severity=EventSeverity.MEDIUM,
            detected_at=datetime.now(UTC),
            details={
                "old_status": change.old_value,
                "new_status": change.new_value,
                "status_field": change.column,
                "source": "change_log_analysis"
            },
            rule_name="Job On Hold Detection"
        )
    
    def _create_job_status_change_event(self, job_id: str, change: AuditChange) -> EventResult:
        """Create event for any job status change"""
        
        return EventResult(
            event_type="job_status_changed",
            entity_type="job",
            entity_id=job_id,
            severity=EventSeverity.LOW,
            detected_at=datetime.now(UTC),
            details={
                "old_status": change.old_value,
                "new_status": change.new_value,
                "status_field": change.column,
                "source": "change_log_analysis"
            },
            rule_name="Job Status Change Detection"
        )
    
    def _create_estimate_sold_event(self, estimate_id: str, change: AuditChange) -> EventResult:
        """Create event for estimate sold"""
        
        return EventResult(
            event_type="estimate_sold",
            entity_type="estimate",
            entity_id=estimate_id,
            severity=EventSeverity.LOW,
            detected_at=datetime.now(UTC),
            details={
                "old_status": change.old_value,
                "new_status": change.new_value,
                "status_field": change.column,
                "source": "change_log_analysis"
            },
            rule_name="Estimate Sold Detection"
        )
    
    def _create_estimate_dismissed_event(self, estimate_id: str, change: AuditChange) -> EventResult:
        """Create event for estimate dismissed"""
        
        return EventResult(
            event_type="estimate_dismissed",
            entity_type="estimate",
            entity_id=estimate_id,
            severity=EventSeverity.MEDIUM,
            detected_at=datetime.now(UTC),
            details={
                "old_status": change.old_value,
                "new_status": change.new_value,
                "status_field": change.column,
                "source": "change_log_analysis"
            },
            rule_name="Estimate Dismissed Detection"
        )
    
    def _create_estimate_expired_event(self, estimate_id: str, change: AuditChange) -> EventResult:
        """Create event for estimate expired"""
        
        return EventResult(
            event_type="estimate_expired",
            entity_type="estimate",
            entity_id=estimate_id,
            severity=EventSeverity.MEDIUM,
            detected_at=datetime.now(UTC),
            details={
                "old_status": change.old_value,
                "new_status": change.new_value,
                "status_field": change.column,
                "source": "change_log_analysis"
            },
            rule_name="Estimate Expired Detection"
        )

