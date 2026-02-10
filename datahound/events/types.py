from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum


class EventSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RuleType(Enum):
    SINGLE_TABLE = "single_table"
    CROSS_TABLE = "cross_table"
    CALCULATED = "calculated"
    STATISTICAL = "statistical"
    LLM_ANALYSIS = "llm_analysis"
    ADDRESS_MATCHING = "address_matching"
    DATA_AGGREGATION = "data_aggregation"
    INCREMENTAL_PROCESSING = "incremental_processing"


@dataclass
class EventRule:
    name: str
    description: str
    event_type: str
    rule_type: RuleType
    target_tables: List[str]
    detection_logic: Dict[str, Any]
    output_fields: List[str]
    severity: EventSeverity = EventSeverity.MEDIUM
    enabled: bool = True
    threshold_months: Optional[int] = None  # For maintenance-type events


@dataclass
class EventResult:
    event_type: str
    entity_type: str  # 'location', 'customer', 'job'
    entity_id: str
    severity: EventSeverity
    detected_at: datetime
    details: Dict[str, Any]
    rule_name: str
    months_overdue: Optional[int] = None  # For maintenance events
    last_maintenance_date: Optional[date] = None


@dataclass
class MaintenanceResult:
    """Specialized result for maintenance events"""
    location_id: Optional[str]
    customer_id: Optional[str]
    last_maintenance_date: date
    months_without_maintenance: int
    entity_name: Optional[str] = None
    phone: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None


@dataclass
class SystemAgeResult:
    """Result from LLM system age analysis"""
    location_id: str
    description: str
    age: int
    job_date: str
    text_snippet: str
    reasoning: str = ""


@dataclass
class PermitMatchResult:
    """Result from permit address matching"""
    location_id: str
    match_type: str  # EXACT, FUZZY, NO_MATCH
    score: float
    matched_permits: List[Dict[str, Any]]
    distance: int = 0


@dataclass
class PermitReplacementResult:
    """Result from LLM permit replacement analysis"""
    location_id: str
    description: str
    most_recent_issue_date: str
    years_since_replacement: int
    reasoning: str
    permits_count: int
    most_recent_replacement_contractor: str
    has_mccullough_replacement: bool
    most_recent_replacement_permit_id: str


@dataclass
class LostCustomerResult:
    """Result from lost customer analysis (other contractor permits)"""
    year: int
    unique_customers_other_contractor: int
    total_other_contractor_permits: int
    unique_customers_other_contractor_installs: int
    total_other_contractor_installs: int
    unique_customers_with_any_permit: int
    pct_customers_other_contractor_of_matched: float
    pct_customers_other_contractor_installs_of_matched: float


@dataclass
class EventScanConfig:
    """Configuration for event scanning"""
    months_threshold_min: int = 12
    months_threshold_max: Optional[int] = None  # None means no upper limit
    overwrite_existing: bool = False
    backup_tables: bool = True
    only_locations: bool = False
    only_customers: bool = False
    include_enriched_data: bool = True
    processing_limit: Optional[int] = None  # None means no limit
    show_progress: bool = True
    chunk_size: int = 100  # Process in chunks for progress updates
    
    # New detection modes
    use_change_log_detection: bool = False  # Use change logs instead of master data
    change_log_hours_back: int = 24  # Hours to look back in change logs
    
    # Event persistence
    persist_events: bool = True  # Save events to parquet storage
    update_existing_events: bool = True  # Update existing events when rescanned
    
    # Payload enrichment options
    enrichment: Dict[str, bool] = None  # Will be set in __post_init__
    
    def __post_init__(self):
        if self.enrichment is None:
            self.enrichment = {
                "include_rfm": False,
                "include_demographics": False, 
                "include_marketable": False,
                "include_segmentation": False
            }
    
    @property
    def months_threshold(self) -> int:
        """Backward compatibility - returns minimum threshold"""
        return self.months_threshold_min
    
    def is_in_range(self, months: int) -> bool:
        """Check if months value is within the configured range"""
        if months < self.months_threshold_min:
            return False
        if self.months_threshold_max is not None and months > self.months_threshold_max:
            return False
        return True


@dataclass
class EventScanResult:
    """Results from a complete event scan"""
    rule_name: str
    total_events: int
    events_by_severity: Dict[EventSeverity, int]
    events: List[EventResult]
    scan_duration_ms: int
    tables_scanned: List[str]
    config_used: EventScanConfig
    total_entities_examined: int = 0
    entities_processed: int = 0
    processing_limit_applied: bool = False


# Event-Specific Configuration Classes

@dataclass
class EventTypeConfig:
    """Base configuration for all event types"""
    enabled: bool = False
    severity_threshold: EventSeverity = EventSeverity.MEDIUM
    processing_limit: Optional[int] = None
    selected_columns: List[str] = field(default_factory=list)
    detection_mode: str = "master_data"  # "master_data" or "change_log"
    enrichment: Dict[str, bool] = field(default_factory=dict)
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.enrichment:
            self.enrichment = {
                "include_rfm": False,
                "include_demographics": False,
                "include_marketable": False,
                "include_segmentation": False,
                "include_permit_data": False
            }


@dataclass
class MaintenanceEventConfig(EventTypeConfig):
    """Configuration for overdue maintenance events"""
    months_threshold_min: int = 12
    # Removed: months_threshold_max (per requirements)
    # Removed: only_customers (per requirements - only scan locations)
    only_locations: bool = True  # Changed to True - only scan locations
    include_last_service_details: bool = True
    # Added: customizable data columns
    data_columns_to_include: List[str] = field(default_factory=lambda: [
        "Location ID", "Customer ID", "Service Address", "Customer Name", 
        "Phone", "Email", "City", "State", "Zip", "Last Service Date"
    ])


@dataclass
class AgingSystemsConfig(EventTypeConfig):
    """Configuration for aging systems detection and system age audit"""
    min_age_years: int = 10
    llm_analysis_enabled: bool = True
    max_jobs_to_analyze: int = 10
    prioritize_recent_jobs: bool = True
    # Removed: system_types_to_analyze (per requirements)
    
    # Concurrent LLM processing (per requirements)
    concurrent_llm_calls: int = 20  # Default value of 20
    
    # System age audit settings (from system_age_dependencies.md)
    enable_permit_based_age: bool = True
    permit_llm_analysis: bool = True
    update_locations_table: bool = False  # Whether to update Locations.xlsx with results
    overwrite_existing_ages: bool = False  # Whether to overwrite existing age data
    # Removed: use_current_system_age (per requirements)
    
    # Age determination logic (Step 5 from dependencies)
    prefer_permit_age_over_job_age: bool = True  # Prefer permit-based age when available
    min_permit_age_confidence: float = 0.7  # Minimum confidence for permit-based age
    
    # Pipeline execution settings
    run_full_pipeline: bool = True  # Execute complete pipeline from dependencies
    build_job_histories: bool = True  # Step 1: Build location job histories
    execute_job_llm_analysis: bool = True  # Step 2: LLM analysis of jobs
    execute_permit_matching: bool = True  # Step 3: Address matching to permits
    execute_permit_llm_analysis: bool = True  # Step 4: LLM analysis of permits
    calculate_final_ages: bool = True  # Step 5: Determine final system ages
    
    # Data artifacts (from dependencies coupling style)
    save_location_jobs_data: bool = True  # Save location_jobs_data.json
    save_llm_results: bool = True  # Save llm_results.json and .csv
    save_permit_results: bool = True  # Save permit_results.json and .csv
    
    # Audit reporting
    include_age_discrepancies: bool = True  # Report when job-age != permit-age
    include_replacement_recommendations: bool = True
    target_replacement_age: int = 15  # Age threshold for replacement recommendations
    enable_system_age_audit_events: bool = True  # Generate audit events
    
    # System age audit specific (legacy scan_system_age_audit)
    system_age_audit_min_age: int = 15  # Minimum age for audit events
    include_ep_system_age: bool = True  # Include EP System Age in results
    include_years_since_replacement: bool = True  # Include permit-based age
    include_current_system_age: bool = True  # Include final calculated age


@dataclass
class CanceledJobsConfig(EventTypeConfig):
    """Configuration for canceled jobs detection"""
    hours_back: int = 24
    include_cancellation_reason: bool = True
    exclude_customer_requested: bool = False
    min_job_value: Optional[float] = None
    # Removed: severity_threshold (per requirements)
    # Added: customizable data columns
    data_columns_to_include: List[str] = field(default_factory=lambda: [
        "Job ID", "Customer ID", "Location ID", "Job Status", "Cancellation Reason",
        "Job Value", "Created Date", "Customer Name", "Phone", "Service Address"
    ])


@dataclass
class UnsoldEstimatesConfig(EventTypeConfig):
    """Configuration for unsold estimates detection"""
    days_back: int = 30
    min_estimate_value: Optional[float] = None
    exclude_statuses: List[str] = field(default_factory=lambda: ["Declined by Customer"])
    include_follow_up_dates: bool = True
    # Removed: severity_threshold (per requirements)
    # Added: customizable data columns
    data_columns_to_include: List[str] = field(default_factory=lambda: [
        "Estimate ID", "Customer ID", "Location ID", "Estimate Value", "Status",
        "Created Date", "Follow Up Date", "Customer Name", "Phone", "Service Address"
    ])


@dataclass
class PermitMatchingConfig(EventTypeConfig):
    """Configuration for permit matching events"""
    match_radius_miles: float = 0.1
    min_match_score: float = 0.8
    include_permit_details: bool = True
    match_types: List[str] = field(default_factory=lambda: ["address", "name"])


@dataclass
class LostCustomersConfig(EventTypeConfig):
    """Configuration for lost customer detection"""
    months_threshold: int = 12
    min_historical_jobs: int = 2
    include_competitor_analysis: bool = False
    exclude_do_not_service: bool = True
    # Removed: severity_threshold (per requirements)
    # Added: customizable data columns
    data_columns_to_include: List[str] = field(default_factory=lambda: [
        "Customer ID", "Customer Name", "Phone", "Email", "Service Address",
        "Last Service Date", "Historical Job Count", "Months Since Last Service"
    ])


@dataclass
class MarketShareConfig(EventTypeConfig):
    """Configuration for market share analysis"""
    analysis_radius_miles: float = 5.0
    min_permits_for_analysis: int = 10
    include_competitor_breakdown: bool = True
    time_period_months: int = 12


@dataclass
class SystemAgeAuditConfig(EventTypeConfig):
    """Configuration for system age audit"""
    target_age_years: int = 15
    include_replacement_recommendations: bool = True
    priority_system_types: List[str] = field(default_factory=lambda: ["HVAC", "Water Heater"])


@dataclass
class PermitReplacementsConfig(EventTypeConfig):
    """Configuration for permit replacement analysis"""
    llm_analysis_enabled: bool = True
    confidence_threshold: float = 0.7
    include_system_details: bool = True
    max_permits_to_analyze: int = 50


# Master Event Configuration Container

@dataclass
class EventSystemConfig:
    """Master configuration for the entire event system"""
    overdue_maintenance: MaintenanceEventConfig = field(default_factory=MaintenanceEventConfig)
    aging_systems: AgingSystemsConfig = field(default_factory=AgingSystemsConfig)
    canceled_jobs: CanceledJobsConfig = field(default_factory=CanceledJobsConfig)
    unsold_estimates: UnsoldEstimatesConfig = field(default_factory=UnsoldEstimatesConfig)
    permit_matches: PermitMatchingConfig = field(default_factory=PermitMatchingConfig)
    lost_customers: LostCustomersConfig = field(default_factory=LostCustomersConfig)
    market_share: MarketShareConfig = field(default_factory=MarketShareConfig)
    system_age_audit: SystemAgeAuditConfig = field(default_factory=SystemAgeAuditConfig)
    permit_replacements: PermitReplacementsConfig = field(default_factory=PermitReplacementsConfig)
    
    # Global settings
    global_processing_limit: Optional[int] = None
    show_progress: bool = True
    chunk_size: int = 100
    backup_before_scan: bool = True
    
    def get_config_for_event_type(self, event_type: str) -> Optional[EventTypeConfig]:
        """Get configuration for a specific event type"""
        config_map = {
            "overdue_maintenance": self.overdue_maintenance,
            "aging_systems": self.aging_systems,
            "canceled_jobs": self.canceled_jobs,
            "unsold_estimates": self.unsold_estimates,
            "permit_matches": self.permit_matches,
            "lost_customers": self.lost_customers,
            "market_share": self.market_share,
            "system_age_audit": self.system_age_audit,
            "permit_replacements": self.permit_replacements
        }
        return config_map.get(event_type)
    
    def get_enabled_event_types(self) -> List[str]:
        """Get list of enabled event types"""
        enabled_types = []
        for event_type in ["overdue_maintenance", "aging_systems", "canceled_jobs", 
                          "unsold_estimates", "permit_matches", "lost_customers", 
                          "market_share", "system_age_audit", "permit_replacements"]:
            config = self.get_config_for_event_type(event_type)
            if config and config.enabled:
                enabled_types.append(event_type)
        return enabled_types