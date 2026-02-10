"""Customer profile data types and configurations"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from enum import Enum


class ProfileBuildMode(Enum):
    """Profile building modes"""
    NEW_CUSTOMERS_ONLY = "new_customers_only"
    ALL_CUSTOMERS = "all_customers"


@dataclass
class CustomerProfile:
    """Complete customer profile data structure - core + enrichment all in one"""
    profile_id: str
    customer_id: str
    
    # Core Location data
    location_ids: Set[str] = field(default_factory=set)
    location_count: int = 0
    
    # Core Address data
    customer_addresses: Set[str] = field(default_factory=set)
    normalized_addresses: Set[str] = field(default_factory=set)
    duplicate_addresses: List[Dict[str, Any]] = field(default_factory=list)
    address_count: int = 0
    
    # Core Service history data
    job_ids: Set[str] = field(default_factory=set)
    job_count: int = 0
    
    estimate_ids: Set[str] = field(default_factory=set)
    estimate_count: int = 0
    
    invoice_ids: Set[str] = field(default_factory=set)
    invoice_count: int = 0
    
    call_ids: Set[str] = field(default_factory=set)
    call_count: int = 0
    
    membership_ids: Set[str] = field(default_factory=set)
    membership_count: int = 0
    
    # Enrichment: RFM Analysis
    rfm_recency: Optional[int] = None
    rfm_frequency: Optional[int] = None
    rfm_monetary: Optional[float] = None
    rfm_score: Optional[str] = None
    rfm_segment: Optional[str] = None
    
    # Enrichment: Demographics
    household_income: Optional[float] = None
    property_value: Optional[float] = None
    
    # Enrichment: Permit data
    permit_matches: List[Dict[str, Any]] = field(default_factory=list)
    permit_count: int = 0
    competitor_permits: List[Dict[str, Any]] = field(default_factory=list)
    competitor_permit_count: int = 0
    
    # Enrichment: Marketable status
    is_marketable: Optional[bool] = None
    do_not_call: Optional[bool] = None
    do_not_service: Optional[bool] = None
    
    # Enrichment: Customer segments
    customer_tier: Optional[str] = None
    customer_segment: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    enriched_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for parquet storage"""
        return {
            "profile_id": self.profile_id,
            "customer_id": self.customer_id,
            # Core data
            "location_ids": list(self.location_ids),
            "location_count": self.location_count,
            "customer_addresses": list(self.customer_addresses),
            "normalized_addresses": list(self.normalized_addresses),
            "duplicate_addresses": self.duplicate_addresses,
            "address_count": self.address_count,
            "job_ids": list(self.job_ids),
            "job_count": self.job_count,
            "estimate_ids": list(self.estimate_ids),
            "estimate_count": self.estimate_count,
            "invoice_ids": list(self.invoice_ids),
            "invoice_count": self.invoice_count,
            "call_ids": list(self.call_ids),
            "call_count": self.call_count,
            "membership_ids": list(self.membership_ids),
            "membership_count": self.membership_count,
            # Enrichment data
            "rfm_recency": self.rfm_recency,
            "rfm_frequency": self.rfm_frequency,
            "rfm_monetary": self.rfm_monetary,
            "rfm_score": self.rfm_score,
            "rfm_segment": self.rfm_segment,
            "household_income": self.household_income,
            "property_value": self.property_value,
            "permit_matches": self.permit_matches,
            "permit_count": self.permit_count,
            "competitor_permits": self.competitor_permits,
            "competitor_permit_count": self.competitor_permit_count,
            "is_marketable": self.is_marketable,
            "do_not_call": self.do_not_call,
            "do_not_service": self.do_not_service,
            "customer_tier": self.customer_tier,
            "customer_segment": self.customer_segment,
            # Metadata
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "enriched_at": self.enriched_at.isoformat() if self.enriched_at else None
        }


# Legacy classes - kept for backward compatibility
@dataclass
class ProfileCoreData:
    """Legacy: Core customer profile data structure - use CustomerProfile instead"""
    profile_id: str
    customer_id: str
    
    # Location data
    location_ids: Set[str] = field(default_factory=set)
    location_count: int = 0
    
    # Address data
    customer_addresses: Set[str] = field(default_factory=set)
    normalized_addresses: Set[str] = field(default_factory=set)
    duplicate_addresses: List[Dict[str, Any]] = field(default_factory=list)
    address_count: int = 0
    
    # Service history data
    job_ids: Set[str] = field(default_factory=set)
    job_count: int = 0
    
    estimate_ids: Set[str] = field(default_factory=set)
    estimate_count: int = 0
    
    invoice_ids: Set[str] = field(default_factory=set)
    invoice_count: int = 0
    
    call_ids: Set[str] = field(default_factory=set)
    call_count: int = 0
    
    membership_ids: Set[str] = field(default_factory=set)
    membership_count: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for parquet storage"""
        return {
            "profile_id": self.profile_id,
            "customer_id": self.customer_id,
            "location_ids": list(self.location_ids),
            "location_count": self.location_count,
            "customer_addresses": list(self.customer_addresses),
            "normalized_addresses": list(self.normalized_addresses),
            "duplicate_addresses": self.duplicate_addresses,
            "address_count": self.address_count,
            "job_ids": list(self.job_ids),
            "job_count": self.job_count,
            "estimate_ids": list(self.estimate_ids),
            "estimate_count": self.estimate_count,
            "invoice_ids": list(self.invoice_ids),
            "invoice_count": self.invoice_count,
            "call_ids": list(self.call_ids),
            "call_count": self.call_count,
            "membership_ids": list(self.membership_ids),
            "membership_count": self.membership_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

@dataclass
class ProfileEnrichmentData:
    """Legacy: Extended profile data from enrichment analysis - use CustomerProfile instead"""
    profile_id: str
    
    # RFM Analysis
    rfm_recency: Optional[int] = None
    rfm_frequency: Optional[int] = None
    rfm_monetary: Optional[float] = None
    rfm_score: Optional[str] = None
    rfm_segment: Optional[str] = None
    
    # Demographics
    household_income: Optional[float] = None
    property_value: Optional[float] = None
    
    # Permit data
    permit_matches: List[Dict[str, Any]] = field(default_factory=list)
    permit_count: int = 0
    competitor_permits: List[Dict[str, Any]] = field(default_factory=list)
    competitor_permit_count: int = 0
    
    # Marketable status
    is_marketable: Optional[bool] = None
    do_not_call: Optional[bool] = None
    do_not_service: Optional[bool] = None
    
    # Customer segments
    customer_tier: Optional[str] = None
    customer_segment: Optional[str] = None
    
    # Metadata
    enriched_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for parquet storage"""
        return {
            "profile_id": self.profile_id,
            "rfm_recency": self.rfm_recency,
            "rfm_frequency": self.rfm_frequency,
            "rfm_monetary": self.rfm_monetary,
            "rfm_score": self.rfm_score,
            "rfm_segment": self.rfm_segment,
            "household_income": self.household_income,
            "property_value": self.property_value,
            "permit_matches": self.permit_matches,
            "permit_count": self.permit_count,
            "competitor_permits": self.competitor_permits,
            "competitor_permit_count": self.competitor_permit_count,
            "is_marketable": self.is_marketable,
            "do_not_call": self.do_not_call,
            "do_not_service": self.do_not_service,
            "customer_tier": self.customer_tier,
            "customer_segment": self.customer_segment,
            "enriched_at": self.enriched_at.isoformat()
        }


@dataclass
class ProfileBuildConfig:
    """Configuration for profile building process"""
    mode: ProfileBuildMode = ProfileBuildMode.NEW_CUSTOMERS_ONLY
    
    # Processing settings
    processing_limit: Optional[int] = None
    show_progress: bool = True
    chunk_size: int = 100
    
    # Address deduplication settings
    levenshtein_threshold: float = 0.9  # Strict similarity threshold
    normalize_addresses: bool = True
    
    # Enrichment settings
    include_rfm: bool = True
    include_demographics: bool = True
    include_permits: bool = True
    include_marketable: bool = True
    include_segments: bool = True
    
    # Performance settings
    max_concurrent_workers: int = 4
    enable_caching: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_detailed_progress: bool = True


@dataclass
class ProfileBuildResult:
    """Results from profile building operation"""
    total_customers_processed: int
    new_profiles_created: int
    existing_profiles_updated: int
    errors_encountered: int
    
    # Phase-specific metrics
    phase1_duration_seconds: float
    phase2_duration_seconds: float
    total_duration_seconds: float
    
    # Data quality metrics
    customers_with_locations: int
    customers_with_duplicate_addresses: int
    customers_with_service_history: int
    
    # Error details
    error_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Processing statistics
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "total_customers_processed": self.total_customers_processed,
            "new_profiles_created": self.new_profiles_created,
            "existing_profiles_updated": self.existing_profiles_updated,
            "errors_encountered": self.errors_encountered,
            "phase1_duration_seconds": self.phase1_duration_seconds,
            "phase2_duration_seconds": self.phase2_duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
            "customers_with_locations": self.customers_with_locations,
            "customers_with_duplicate_addresses": self.customers_with_duplicate_addresses,
            "customers_with_service_history": self.customers_with_service_history,
            "error_details": self.error_details,
            "processing_stats": self.processing_stats
        }
