"""Types and data models for custom extraction system"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum


class TimeFilterType(Enum):
    DAYS_BACK = "days_back"
    DATE_RANGE = "date_range"
    STATIC_DATE = "static_date"


class NumericFilterType(Enum):
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than" 
    EQUALS = "equals"
    BETWEEN = "between"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"


@dataclass
class TimeFilter:
    """Configuration for time-based filtering of events"""
    enabled: bool
    filter_type: TimeFilterType
    field_name: str
    days_back: Optional[int] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    static_date: Optional[date] = None
    
    def __post_init__(self):
        if self.enabled:
            if self.filter_type == TimeFilterType.DAYS_BACK and self.days_back is None:
                raise ValueError("days_back required for DAYS_BACK filter type")
            elif self.filter_type == TimeFilterType.DATE_RANGE:
                if self.start_date is None or self.end_date is None:
                    raise ValueError("start_date and end_date required for DATE_RANGE filter type")
            elif self.filter_type == TimeFilterType.STATIC_DATE and self.static_date is None:
                raise ValueError("static_date required for STATIC_DATE filter type")


@dataclass
class NumericFilter:
    """Configuration for numeric value filtering"""
    enabled: bool
    field_name: str
    filter_type: NumericFilterType
    value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def __post_init__(self):
        if self.enabled:
            if self.filter_type in [NumericFilterType.LESS_THAN, NumericFilterType.GREATER_THAN, 
                                   NumericFilterType.EQUALS, NumericFilterType.LESS_THAN_OR_EQUAL, 
                                   NumericFilterType.GREATER_THAN_OR_EQUAL]:
                if self.value is None:
                    raise ValueError(f"value required for {self.filter_type.value} filter type")
            elif self.filter_type == NumericFilterType.BETWEEN:
                if self.min_value is None or self.max_value is None:
                    raise ValueError("min_value and max_value required for BETWEEN filter type")


@dataclass
class EnrichmentConfig:
    """Configuration for data enrichment options"""
    include_customer_core_data: bool = True
    include_rfm_analysis: bool = False
    include_demographics: bool = False
    include_segmentation: bool = False
    customer_id_field: str = "Customer ID"


@dataclass
class ExtractionConfig:
    """Configuration for a custom extraction rule"""
    name: str
    description: str
    enabled: bool
    source_event_type: str
    source_file_name: str
    time_filter: TimeFilter
    numeric_filter: NumericFilter
    enrichment: EnrichmentConfig
    output_file_name: str
    max_records: Optional[int] = None
    enable_inbound_call_check: bool = True
    inbound_call_window_days: Optional[int] = 0
    inbound_call_directions: List[str] = field(default_factory=lambda: ["Inbound"])
    enable_sms_activity_check: bool = True
    sms_window_minutes: Optional[int] = 0
    invalid_reason_field: Optional[str] = None
    
    @classmethod
    def create_recent_cancellations(cls, days_back: int = 30) -> ExtractionConfig:
        """Create config for recent cancellations extraction"""
        return cls(
            name="Recent Cancellations",
            description=f"Canceled jobs within the last {days_back} days with customer data",
            enabled=True,
            source_event_type="canceled_jobs",
            source_file_name="canceled_jobs_master.parquet",
            time_filter=TimeFilter(
                enabled=True,
                filter_type=TimeFilterType.DAYS_BACK,
                field_name="completion_date",
                days_back=days_back
            ),
            numeric_filter=NumericFilter(
                enabled=False,
                field_name="",
                filter_type=NumericFilterType.LESS_THAN
            ),
            enrichment=EnrichmentConfig(
                include_customer_core_data=True,
                customer_id_field="Customer ID"
            ),
            output_file_name="recent_cancellations.parquet",
            enable_inbound_call_check=True,
            inbound_call_window_days=0,
            inbound_call_directions=["Inbound"],
            invalid_reason_field="invalidation_reason"
        )
    
    @classmethod
    def create_recent_lost_customers(cls, days_back: int = 30) -> ExtractionConfig:
        """Create config for recent lost customers extraction"""
        return cls(
            name="Recent Lost Customers", 
            description=f"Lost customers identified within the last {days_back} days",
            enabled=True,
            source_event_type="lost_customers",
            source_file_name="lost_customers_master.parquet",
            time_filter=TimeFilter(
                enabled=False,
                filter_type=TimeFilterType.DAYS_BACK,
                field_name="detected_at",
                days_back=days_back
            ),
            numeric_filter=NumericFilter(
                enabled=True,
                field_name="months_since_last_contact",
                filter_type=NumericFilterType.LESS_THAN,
                value=13
            ),
            enrichment=EnrichmentConfig(
                include_customer_core_data=True,
                customer_id_field="entity_id"
            ),
            output_file_name="recent_lost_customers.parquet",
            invalid_reason_field="invalidation_reason"
        )
    
    @classmethod
    def create_recent_aging_systems(cls, days_back: int = 7) -> ExtractionConfig:
        """Create config for recently detected aging systems"""
        return cls(
            name="Recent Aging Systems",
            description=f"Aging systems detected within the last {days_back} days",
            enabled=True,
            source_event_type="aging_systems", 
            source_file_name="aging_systems_master.parquet",
            time_filter=TimeFilter(
                enabled=True,
                filter_type=TimeFilterType.DAYS_BACK,
                field_name="detected_at",
                days_back=days_back
            ),
            numeric_filter=NumericFilter(
                enabled=False,
                field_name="",
                filter_type=NumericFilterType.GREATER_THAN
            ),
            enrichment=EnrichmentConfig(
                include_customer_core_data=True,
                customer_id_field="entity_id"
            ),
            output_file_name="recent_aging_systems.parquet"
        )
    
    @classmethod
    def create_recent_second_chance_leads(cls) -> ExtractionConfig:
        """Create config for recent second chance leads extraction"""
        return cls(
            name="Recent Second Chance Leads",
            description="Second chance leads identified from call analysis",
            enabled=True,
            source_event_type="second_chance_leads",
            source_file_name="second_chance_leads.parquet",
            time_filter=TimeFilter(
                enabled=False,  # No time filter needed - filter by is_second_chance_lead=True
                filter_type=TimeFilterType.DAYS_BACK,
                field_name="analysis_timestamp",
                days_back=30
            ),
            numeric_filter=NumericFilter(
                enabled=False,
                field_name="",
                filter_type=NumericFilterType.LESS_THAN
            ),
            enrichment=EnrichmentConfig(
                include_customer_core_data=True,
                customer_id_field="Customer ID"
            ),
            output_file_name="recent_second_chance_leads.parquet",
            invalid_reason_field="invalidation_reason"
        )


@dataclass
class ExtractionResult:
    """Result from a custom extraction operation"""
    config_name: str
    success: bool
    records_found: int
    records_enriched: int
    records_saved: int
    output_file: Path
    duration_ms: int
    error_message: Optional[str] = None
    enrichment_stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def enrichment_rate(self) -> float:
        """Calculate percentage of records successfully enriched"""
        if self.records_found == 0:
            return 0.0
        return (self.records_enriched / self.records_found) * 100.0


@dataclass
class ExtractionBatch:
    """Batch configuration for running multiple extractions"""
    extractions: List[ExtractionConfig]
    company: str
    run_timestamp: datetime = field(default_factory=lambda: datetime.now())
    parallel_execution: bool = False
    
    @property
    def enabled_extractions(self) -> List[ExtractionConfig]:
        """Get only enabled extraction configurations"""
        return [config for config in self.extractions if config.enabled]
