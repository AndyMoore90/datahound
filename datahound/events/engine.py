from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, date, UTC
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pyarrow.parquet as pq

from .types import (
    EventRule, 
    EventResult, 
    EventScanConfig, 
    EventScanResult, 
    MaintenanceResult,
    SystemAgeResult,
    PermitMatchResult,
    PermitReplacementResult,
    LostCustomerResult,
    EventSeverity, 
    RuleType
)
from .llm_utils import LLMConfig, SystemAgeAnalyzer, PermitReplacementAnalyzer
from .address_utils import normalize_address_street, extract_house_number_token, get_street_core_without_number, is_mccullough
from .scan_methods import scan_aging_systems, scan_canceled_jobs, scan_unsold_estimates, scan_permit_matching, scan_system_age_audit, scan_permit_replacements, scan_lost_customers
from .logging import EventLogger
from .persistence import EventStore, ChangeLogEventDetector, EventEnricher
from .event_storage import EventMasterStorage
from .change_processor import RecentEventsManager


class EventScanner:
    """Core event detection engine for scanning master parquet files"""
    
    def __init__(self, company: str, parquet_dir: Path, llm_config: Optional[LLMConfig] = None, data_dir: Optional[Path] = None):
        self.company = company
        self.parquet_dir = parquet_dir
        self.master_data: Dict[str, pd.DataFrame] = {}
        self.table_schemas: Dict[str, List[str]] = {}
        
        # Initialize logging
        self.data_dir = data_dir or Path(f"data/{company}/downloads")
        self.logger = EventLogger(company, self.data_dir)
        
        # Initialize event persistence and enrichment
        self.event_store = EventStore(company, self.data_dir)
        self.change_detector = ChangeLogEventDetector(company, self.data_dir)
        self.enricher = EventEnricher(company, self.data_dir)
        
        # Initialize new master storage system
        self.master_storage = EventMasterStorage(company, self.data_dir)
        
        # Initialize recent events manager
        self.recent_events_manager = RecentEventsManager(company, self.data_dir)
        
        # Initialize LLM analyzers
        self.llm_config = llm_config or LLMConfig()
        self.system_age_analyzer = SystemAgeAnalyzer(self.llm_config, self.logger)
        self.permit_replacement_analyzer = PermitReplacementAnalyzer(self.llm_config, self.logger)
    
    def load_master_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load a master parquet file and cache it"""
        start_time = time.perf_counter()
        
        if table_name in self.master_data:
            return self.master_data[table_name]
        
        parquet_path = self.parquet_dir / f"{table_name.capitalize()}.parquet"
        if not parquet_path.exists():
            self.logger.log_table_load(table_name, 0, [], 0, False)
            return None
        
        try:
            df = pd.read_parquet(parquet_path)
            self.master_data[table_name] = df
            self.table_schemas[table_name] = list(df.columns)
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            self.logger.log_table_load(table_name, len(df), list(df.columns), duration_ms, True)
            
            return df
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            self.logger.log_table_load(table_name, 0, [], duration_ms, False)
            return None
    
    def clear_cache(self) -> None:
        """Clear cached master data to free memory"""
        self.master_data.clear()
        self.table_schemas.clear()
    
    def get_available_tables(self) -> List[str]:
        """Get list of available master parquet files"""
        tables = []
        for parquet_file in self.parquet_dir.glob("*.parquet"):
            table_name = parquet_file.stem.lower()
            tables.append(table_name)
        return sorted(tables)
    
    def load_permit_data(self) -> Optional[pd.DataFrame]:
        """Load permit data from global data directory"""
        # Check for permit data in global_data/permits/
        permit_files = [
            Path("global_data/permits/permit_data.csv"),
            Path("global_data/permits").glob("permits_austin_*.csv")
        ]
        
        # Try to find the most recent permit file
        permit_path = None
        for path_or_glob in permit_files:
            if isinstance(path_or_glob, Path) and path_or_glob.exists():
                permit_path = path_or_glob
                break
            elif hasattr(path_or_glob, '__iter__'):  # It's a glob result
                files = list(path_or_glob)
                if files:
                    # Get most recent file
                    permit_path = max(files, key=lambda p: p.stat().st_mtime)
                    break
        
        if not permit_path:
            return None
        
        try:
            # Read with string dtype to avoid mixed type warnings
            df = pd.read_csv(permit_path, dtype=str, low_memory=False)
            self.logger.log_table_load("permits", len(df), list(df.columns), 0, True)
            return df
        except Exception as e:
            self.logger.log_table_load("permits", 0, [], 0, False)
            return None
    
    def scan_for_events(self, rules: List[EventRule], config: EventScanConfig) -> List[EventScanResult]:
        """Execute multiple rules and return all detected events"""
        scan_timestamp = datetime.now(UTC)
        results = []
        
        for rule in rules:
            if rule.enabled:
                # Check if this rule supports change log detection
                if (config.use_change_log_detection and 
                    rule.event_type in ["canceled_jobs", "unsold_estimates"]):
                    result = self._scan_from_change_logs(rule, config, scan_timestamp)
                else:
                    result = self.scan_single_rule(rule, config)
                
                # Enrich events if requested
                if config.include_enriched_data and result.events:
                    enriched_events = []
                    for event in result.events:
                        enriched_payload = self.enricher.enrich_event(event, config.enrichment)
                        # Create new event with enriched payload
                        enriched_event = EventResult(
                            event_type=event.event_type,
                            entity_type=event.entity_type,
                            entity_id=event.entity_id,
                            severity=event.severity,
                            detected_at=event.detected_at,
                            details=enriched_payload,
                            rule_name=event.rule_name,
                            months_overdue=event.months_overdue,
                            last_maintenance_date=event.last_maintenance_date
                        )
                        enriched_events.append(enriched_event)
                    
                    # Update result with enriched events
                    result.events = enriched_events
                
                # Persist events if requested
                if config.persist_events and result.events:
                    # Save to legacy storage for backward compatibility
                    persistence_stats = self.event_store.save_events(result.events, scan_timestamp)
                    self.logger.log_processing_stats(rule.name, "persistence", persistence_stats)
                    
                    # Save to new master storage system
                    update_mode = "append_only" if rule.event_type == "aging_systems" else "upsert"
                    master_stats = self.master_storage.save_event_results(
                        rule.event_type, result.events, scan_timestamp, update_mode
                    )
                    self.logger.log_processing_stats(rule.name, "master_storage", master_stats)
                
                results.append(result)
        
        # Process recent events after all scans complete
        if config.persist_events:
            try:
                self.recent_events_manager.process_change_logs()
                self.logger.log_processing_stats("recent_events", "processing", 
                    self.recent_events_manager.get_summary_stats())
            except Exception as e:
                self.logger.log_scan_error("recent_events_processing", str(e), {
                    "component": "RecentEventsManager"
                })
        
        return results
    
    def _scan_from_change_logs(self, rule: EventRule, config: EventScanConfig, scan_timestamp: datetime) -> EventScanResult:
        """Scan for events using change log analysis instead of master data"""
        
        start_time = time.perf_counter()
        
        # Log scan start
        self.logger.log_scan_start(rule.name, "change_log_analysis", {
            "event_type": rule.event_type,
            "hours_back": config.change_log_hours_back,
            "detection_mode": "change_log"
        })
        
        events = []
        
        try:
            if rule.event_type == "canceled_jobs":
                # Detect recent cancellations from change logs
                recent_cancellations = self.change_detector.detect_recent_cancellations(config.change_log_hours_back)
                
                # Convert to EventResult objects
                for cancellation in recent_cancellations:
                    event = EventResult(
                        event_type="canceled_jobs",
                        entity_type="job",
                        entity_id=cancellation["job_id"],
                        severity=EventSeverity.MEDIUM,
                        detected_at=datetime.fromisoformat(cancellation["timestamp"]),
                        details={
                            "status": cancellation["new_status"],
                            "old_status": cancellation["old_status"],
                            "change_source": "log_analysis",
                            "detection_window_hours": config.change_log_hours_back
                        },
                        rule_name=rule.name
                    )
                    events.append(event)
            
            elif rule.event_type == "unsold_estimates":
                # Detect recent unsold estimates from change logs
                recent_unsold = self.change_detector.detect_recent_unsold_estimates(config.change_log_hours_back)
                
                # Convert to EventResult objects
                for unsold in recent_unsold:
                    event = EventResult(
                        event_type="unsold_estimates",
                        entity_type="estimate",
                        entity_id=unsold["estimate_id"],
                        severity=EventSeverity.MEDIUM,
                        detected_at=datetime.fromisoformat(unsold["timestamp"]),
                        details={
                            "status": unsold["new_status"],
                            "old_status": unsold["old_status"],
                            "change_source": "log_analysis",
                            "detection_window_hours": config.change_log_hours_back
                        },
                        rule_name=rule.name
                    )
                    events.append(event)
        
        except Exception as e:
            self.logger.log_scan_error(rule.name, str(e), {
                "detection_mode": "change_log",
                "hours_back": config.change_log_hours_back
            })
            raise
        
        end_time = time.perf_counter()
        duration_ms = int((end_time - start_time) * 1000)
        
        # Aggregate results by severity
        events_by_severity = defaultdict(int)
        for event in events:
            events_by_severity[event.severity] += 1
        
        # Log completion
        self.logger.log_scan_complete(
            rule.name, len(events), 0, len(events), duration_ms, False
        )
        
        return EventScanResult(
            rule_name=rule.name,
            total_events=len(events),
            events_by_severity=dict(events_by_severity),
            events=events,
            scan_duration_ms=duration_ms,
            tables_scanned=["change_logs"],
            config_used=config,
            total_entities_examined=len(events),
            entities_processed=len(events),
            processing_limit_applied=False
        )
    
    def scan_single_rule(self, rule: EventRule, config: EventScanConfig) -> EventScanResult:
        """Execute a single event detection rule"""
        start_time = time.perf_counter()
        
        # Log scan start
        self.logger.log_scan_start(rule.name, rule.rule_type.value, {
            "event_type": rule.event_type,
            "target_tables": rule.target_tables,
            "processing_limit": config.processing_limit,
            "show_progress": config.show_progress,
            "threshold_min": config.months_threshold_min,
            "threshold_max": config.months_threshold_max
        })
        
        try:
            if rule.event_type == "overdue_maintenance":
                events = self._scan_overdue_maintenance(rule, config)
            elif rule.event_type == "aging_systems":
                events = scan_aging_systems(self, rule, config)
            elif rule.event_type == "canceled_jobs":
                events = scan_canceled_jobs(self, rule, config)
            elif rule.event_type == "unsold_estimates":
                events = scan_unsold_estimates(self, rule, config)
            elif rule.event_type == "permit_matches":
                events = scan_permit_matching(self, rule, config)
            elif rule.event_type == "system_age_audit":
                events = scan_system_age_audit(self, rule, config)
            elif rule.event_type == "permit_replacements":
                events = scan_permit_replacements(self, rule, config)
            elif rule.event_type == "lost_customers":
                events = scan_lost_customers(self, rule, config)
            else:
                events = []
        except Exception as e:
            self.logger.log_scan_error(rule.name, str(e), {
                "rule_type": rule.rule_type.value,
                "event_type": rule.event_type,
                "config": config.__dict__
            })
            raise
        
        end_time = time.perf_counter()
        duration_ms = int((end_time - start_time) * 1000)
        
        # Aggregate results by severity
        events_by_severity = defaultdict(int)
        for event in events:
            events_by_severity[event.severity] += 1
        
        # Calculate processing statistics
        total_examined = 0
        processed = 0
        limit_applied = False
        
        if rule.event_type == "overdue_maintenance":
            # Get stats from the actual scan method
            if hasattr(self, '_last_maintenance_stats'):
                total_examined = self._last_maintenance_stats.get('total_examined', 0)
                processed = self._last_maintenance_stats.get('processed', 0)
                limit_applied = self._last_maintenance_stats.get('limit_applied', False)
            else:
                total_examined = 0
                processed = 0
                limit_applied = False
        elif rule.event_type == "aging_systems":
            # Get stats from the scan method
            if hasattr(self, '_last_aging_stats'):
                total_examined = self._last_aging_stats.get('total_examined', 0)
                processed = self._last_aging_stats.get('processed', 0)
                limit_applied = self._last_aging_stats.get('limit_applied', False)
            else:
                total_examined = 0
                processed = 0
                limit_applied = False
        elif rule.event_type == "canceled_jobs":
            if hasattr(self, '_last_canceled_stats'):
                total_examined = self._last_canceled_stats.get('total_examined', 0)
                processed = self._last_canceled_stats.get('processed', 0)
                limit_applied = self._last_canceled_stats.get('limit_applied', False)
            else:
                total_examined = 0
                processed = 0
                limit_applied = False
        elif rule.event_type == "unsold_estimates":
            if hasattr(self, '_last_estimates_stats'):
                total_examined = self._last_estimates_stats.get('total_examined', 0)
                processed = self._last_estimates_stats.get('processed', 0)
                limit_applied = self._last_estimates_stats.get('limit_applied', False)
            else:
                total_examined = 0
                processed = 0
                limit_applied = False
        elif rule.event_type in ["permit_matches", "lost_customers"]:
            # Handle permit-based scans
            stat_name = f"_last_{rule.event_type.replace('_', '_')}_stats"
            if hasattr(self, stat_name):
                stats = getattr(self, stat_name)
                total_examined = stats.get('total_examined', 0)
                processed = stats.get('processed', 0)
                limit_applied = stats.get('limit_applied', False)
            else:
                total_examined = 0
                processed = 0
                limit_applied = False
        elif rule.event_type == "canceled_jobs":
            if hasattr(self, '_last_canceled_stats'):
                total_examined = self._last_canceled_stats.get('total_examined', 0)
                processed = self._last_canceled_stats.get('processed', 0)
                limit_applied = self._last_canceled_stats.get('limit_applied', False)
            else:
                total_examined = 0
                processed = 0
                limit_applied = False
        elif rule.event_type == "unsold_estimates":
            if hasattr(self, '_last_estimates_stats'):
                total_examined = self._last_estimates_stats.get('total_examined', 0)
                processed = self._last_estimates_stats.get('processed', 0)
                limit_applied = self._last_estimates_stats.get('limit_applied', False)
            else:
                total_examined = 0
                processed = 0
                limit_applied = False
        
        # Log scan completion
        self.logger.log_scan_complete(
            rule.name, len(events), total_examined, processed, duration_ms, limit_applied
        )
        
        # Log each detected event
        for event in events:
            self.logger.log_event_detected(
                event.event_type, event.entity_type, event.entity_id,
                event.severity.value, event.details, rule.name
            )
        
        return EventScanResult(
            rule_name=rule.name,
            total_events=len(events),
            events_by_severity=dict(events_by_severity),
            events=events,
            scan_duration_ms=duration_ms,
            tables_scanned=rule.target_tables,
            config_used=config,
            total_entities_examined=total_examined,
            entities_processed=processed,
            processing_limit_applied=limit_applied
        )
    
    def _scan_overdue_maintenance(self, rule: EventRule, config: EventScanConfig) -> List[EventResult]:
        """Implement overdue maintenance detection logic"""
        
        # Load required tables
        jobs_df = self.load_master_table("jobs")
        locations_df = self.load_master_table("locations")
        customers_df = self.load_master_table("customers")
        
        if jobs_df is None:
            return []
        
        # Find maintenance jobs
        maintenance_jobs = self._filter_maintenance_jobs(jobs_df)
        if maintenance_jobs.empty:
            return []
        
        # Calculate last maintenance per location and customer
        last_by_location, last_by_customer = self._calculate_last_maintenance(maintenance_jobs)
        
        # Generate events based on threshold range
        events = []
        today = date.today()
        threshold_min = config.months_threshold_min
        threshold_max = config.months_threshold_max
        
        # Apply processing limits
        location_items = list(last_by_location.items())
        customer_items = list(last_by_customer.items())
        
        if config.processing_limit is not None:
            # Split limit between locations and customers
            if not config.only_customers and not config.only_locations:
                # Split evenly
                location_limit = config.processing_limit // 2
                customer_limit = config.processing_limit - location_limit
            elif config.only_locations:
                location_limit = config.processing_limit
                customer_limit = 0
            elif config.only_customers:
                location_limit = 0
                customer_limit = config.processing_limit
            else:
                location_limit = config.processing_limit // 2
                customer_limit = config.processing_limit - location_limit
            
            location_items = location_items[:location_limit] if location_limit > 0 else []
            customer_items = customer_items[:customer_limit] if customer_limit > 0 else []
        
        total_to_process = len(location_items) + len(customer_items)
        
        # Setup progress tracking
        progress_tracker = None
        if config.show_progress and total_to_process > 0:
            from .progress import create_streamlit_progress
            progress_tracker = create_streamlit_progress(
                total_to_process,
                f"Checking maintenance for {total_to_process} entities"
            )
        
        # Process locations
        if not config.only_customers and locations_df is not None:
            for location_id, last_date in location_items:
                if progress_tracker:
                    progress_tracker.update(1, f"Checking location {location_id}")
                
                months_overdue = self._months_between(last_date, today)
                if config.is_in_range(months_overdue):
                    # Get enriched location data
                    location_data = self._get_location_details(locations_df, location_id)
                    
                    event = EventResult(
                        event_type="overdue_maintenance",
                        entity_type="location",
                        entity_id=location_id,
                        severity=self._determine_severity(months_overdue, threshold_min),
                        detected_at=datetime.now(UTC),
                        details={
                            "months_overdue": months_overdue,
                            "last_maintenance_date": last_date.isoformat(),
                            "threshold_min": threshold_min,
                            "threshold_max": threshold_max,
                            **location_data
                        },
                        rule_name=rule.name,
                        months_overdue=months_overdue,
                        last_maintenance_date=last_date
                    )
                    events.append(event)
        
        # Process customers  
        if not config.only_locations and customers_df is not None:
            for customer_id, last_date in customer_items:
                if progress_tracker:
                    progress_tracker.update(1, f"Checking customer {customer_id}")
                
                months_overdue = self._months_between(last_date, today)
                if config.is_in_range(months_overdue):
                    # Get enriched customer data
                    customer_data = self._get_customer_details(customers_df, customer_id)
                    
                    event = EventResult(
                        event_type="overdue_maintenance",
                        entity_type="customer", 
                        entity_id=customer_id,
                        severity=self._determine_severity(months_overdue, threshold_min),
                        detected_at=datetime.now(UTC),
                        details={
                            "months_overdue": months_overdue,
                            "last_maintenance_date": last_date.isoformat(),
                            "threshold_min": threshold_min,
                            "threshold_max": threshold_max,
                            **customer_data
                        },
                        rule_name=rule.name,
                        months_overdue=months_overdue,
                        last_maintenance_date=last_date
                    )
                    events.append(event)
        
        # Store statistics for result calculation
        total_available = len(list(last_by_location.items()) + list(last_by_customer.items()))
        total_processed = len(location_items) + len(customer_items)
        limit_was_applied = config.processing_limit is not None and total_available > total_processed
        
        self._last_maintenance_stats = {
            'total_examined': total_available,
            'processed': total_processed,
            'limit_applied': limit_was_applied
        }
        
        return events
    
    def _filter_maintenance_jobs(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Filter jobs to find maintenance jobs based on legacy logic"""
        
        # Normalize column names (case-insensitive lookup)
        col_map = {col.lower(): col for col in jobs_df.columns}
        
        job_class_col = col_map.get('job class') or col_map.get('jobclass') or col_map.get('class')
        job_type_col = col_map.get('job type') or col_map.get('jobtype') or col_map.get('type')
        summary_col = col_map.get('summary') or col_map.get('description') or col_map.get('job summary')
        
        if not job_class_col and not job_type_col:
            return pd.DataFrame()
        
        # Create maintenance filter
        maintenance_mask = pd.Series([False] * len(jobs_df))
        
        if job_class_col:
            class_maintenance = jobs_df[job_class_col].astype(str).str.lower().str.strip() == 'maintenance'
            maintenance_mask |= class_maintenance
        
        if job_type_col:
            type_maintenance = jobs_df[job_type_col].astype(str).str.lower().str.strip() == 'maintenance'
            maintenance_mask |= type_maintenance
        
        maintenance_jobs = jobs_df[maintenance_mask].copy()
        
        # Filter out test jobs
        if summary_col and not maintenance_jobs.empty:
            summary_series = maintenance_jobs[summary_col].astype(str).str.lower()
            test_mask = (
                (summary_series == 'test') | 
                summary_series.str.contains('testing', na=False)
            )
            maintenance_jobs = maintenance_jobs[~test_mask]
        
        return maintenance_jobs
    
    def _calculate_last_maintenance(self, maintenance_jobs: pd.DataFrame) -> Tuple[Dict[str, date], Dict[str, date]]:
        """Calculate last maintenance date per location and customer"""
        
        col_map = {col.lower(): col for col in maintenance_jobs.columns}
        
        # Find date columns
        completion_col = (col_map.get('completion date') or 
                         col_map.get('completed date') or 
                         col_map.get('date completed'))
        created_col = (col_map.get('created date') or 
                      col_map.get('date created') or 
                      col_map.get('job date'))
        
        location_col = (col_map.get('location id') or 
                       col_map.get('locationid') or 
                       col_map.get('location'))
        customer_col = (col_map.get('customer id') or 
                       col_map.get('customerid') or 
                       col_map.get('customer'))
        
        last_by_location = {}
        last_by_customer = {}
        
        for _, row in maintenance_jobs.iterrows():
            # Determine maintenance date (completion date preferred, then created date)
            maintenance_date = None
            
            if completion_col and pd.notna(row[completion_col]):
                maintenance_date = self._parse_date(row[completion_col])
            
            if not maintenance_date and created_col and pd.notna(row[created_col]):
                maintenance_date = self._parse_date(row[created_col])
            
            if not maintenance_date:
                continue
            
            # Update last maintenance by location
            if location_col and pd.notna(row[location_col]):
                location_id = str(row[location_col]).strip()
                if location_id and (location_id not in last_by_location or 
                                  maintenance_date > last_by_location[location_id]):
                    last_by_location[location_id] = maintenance_date
            
            # Update last maintenance by customer  
            if customer_col and pd.notna(row[customer_col]):
                customer_id = str(row[customer_col]).strip()
                if customer_id and (customer_id not in last_by_customer or 
                                  maintenance_date > last_by_customer[customer_id]):
                    last_by_customer[customer_id] = maintenance_date
        
        return last_by_location, last_by_customer
    
    def _parse_date(self, date_value: Any) -> Optional[date]:
        """Parse date from various formats, matching legacy logic"""
        if pd.isna(date_value):
            return None
        
        if isinstance(date_value, date):
            return date_value
        
        if isinstance(date_value, datetime):
            return date_value.date()
        
        date_str = str(date_value).strip()
        if not date_str or date_str.lower() in ('', 'nan', 'nat', 'none'):
            return None
        
        # Try multiple date formats
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%m-%d-%Y', 
            '%Y/%m/%d',
            '%m/%d/%y',
            '%m-%d-%y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        
        # Try pandas parsing as fallback
        try:
            parsed = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(parsed):
                return parsed.date()
        except Exception:
            pass
        
        return None
    
    def _months_between(self, start: date, end: date) -> int:
        """Calculate months between dates using legacy logic"""
        if start > end:
            return 0
        
        months = (end.year - start.year) * 12 + (end.month - start.month)
        if end.day < start.day:
            months -= 1
        
        return max(0, months)
    
    def _determine_severity(self, months_overdue: int, threshold: int) -> EventSeverity:
        """Determine event severity based on how overdue maintenance is"""
        if months_overdue >= threshold * 2:  # 24+ months
            return EventSeverity.CRITICAL
        elif months_overdue >= threshold * 1.5:  # 18+ months  
            return EventSeverity.HIGH
        elif months_overdue >= threshold * 1.25:  # 15+ months
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.LOW
    
    def _get_location_details(self, locations_df: pd.DataFrame, location_id: str) -> Dict[str, Any]:
        """Get enriched location details for event"""
        col_map = {col.lower(): col for col in locations_df.columns}
        
        # Find location row
        location_id_col = None
        for possible_col in ['location id', 'locationid', 'id']:
            if possible_col in col_map:
                location_id_col = col_map[possible_col]
                break
        
        if not location_id_col:
            return {}
        
        location_row = locations_df[locations_df[location_id_col].astype(str) == location_id]
        if location_row.empty:
            return {}
        
        row = location_row.iloc[0]
        details = {}
        
        # Extract common fields
        field_mapping = {
            'name': ['name', 'location name', 'site name'],
            'phone': ['phone', 'phone number', 'contact phone'],
            'city': ['city'],
            'state': ['state'],
            'zip': ['zip', 'zip code', 'postal code']
        }
        
        for detail_key, possible_cols in field_mapping.items():
            for col in possible_cols:
                if col in col_map and pd.notna(row[col_map[col]]):
                    details[detail_key] = str(row[col_map[col]]).strip()
                    break
        
        return details
    
    def _get_customer_details(self, customers_df: pd.DataFrame, customer_id: str) -> Dict[str, Any]:
        """Get enriched customer details for event"""
        col_map = {col.lower(): col for col in customers_df.columns}
        
        # Find customer row
        customer_id_col = None
        for possible_col in ['customer id', 'customerid', 'id']:
            if possible_col in col_map:
                customer_id_col = col_map[possible_col]
                break
        
        if not customer_id_col:
            return {}
        
        customer_row = customers_df[customers_df[customer_id_col].astype(str) == customer_id]
        if customer_row.empty:
            return {}
        
        row = customer_row.iloc[0]
        details = {}
        
        # Extract common fields
        field_mapping = {
            'name': ['name', 'customer name', 'full name', 'first name'],
            'phone': ['phone', 'phone number', 'home phone', 'mobile phone'],
            'city': ['city'],
            'state': ['state'],
            'zip': ['zip', 'zip code', 'postal code']
        }
        
        for detail_key, possible_cols in field_mapping.items():
            for col in possible_cols:
                if col in col_map and pd.notna(row[col_map[col]]):
                    details[detail_key] = str(row[col_map[col]]).strip()
                    break
        
        return details
