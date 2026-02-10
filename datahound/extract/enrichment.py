"""Data enrichment utilities for custom extraction"""

from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import time

from .types import EnrichmentConfig


class DataEnrichment:
    """Handles enrichment of extracted event data with customer information"""
    
    def __init__(self, company: str, parquet_dir: Path):
        self.company = company
        self.parquet_dir = parquet_dir
        self._customer_core_cache: Optional[pd.DataFrame] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes
    
    def load_customer_core_data(self, force_reload: bool = False) -> Optional[pd.DataFrame]:
        """Load customer core data with caching"""
        
        # Check cache validity
        if not force_reload and self._customer_core_cache is not None:
            if self._cache_timestamp and (datetime.now(UTC) - self._cache_timestamp).seconds < self._cache_ttl_seconds:
                return self._customer_core_cache
        
        # Try new core data file first, then legacy
        core_data_files = [
            self.parquet_dir / "customer_core_data.parquet",
            self.parquet_dir / "customer_profiles_core_data.parquet"
        ]
        
        for core_file in core_data_files:
            if core_file.exists():
                try:
                    df = pd.read_parquet(core_file)
                    
                    # Cache the data
                    self._customer_core_cache = df
                    self._cache_timestamp = datetime.now(UTC)
                    
                    return df
                except Exception as e:
                    continue
        
        return None
    
    def enrich_event_data(self, event_df: pd.DataFrame, config: EnrichmentConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enrich event data with customer information"""
        
        start_time = time.perf_counter()
        stats = {
            "records_input": len(event_df),
            "records_enriched": 0,
            "enrichment_fields_added": 0,
            "missing_customer_ids": 0,
            "failed_lookups": 0,
            "invalid_reasons": {}
        }
        
        if event_df.empty:
            return event_df, stats
        
        # Load customer core data
        customer_df = self.load_customer_core_data()
        if customer_df is None:
            stats["error"] = "Could not load customer core data"
            return event_df, stats
        
        # Prepare for merge
        enriched_df = event_df.copy()
        
        if config.include_customer_core_data:
            # Find customer ID field in event data
            customer_id_field = self._find_customer_id_field(enriched_df, config.customer_id_field)
            if not customer_id_field:
                stats["error"] = f"Customer ID field '{config.customer_id_field}' not found in event data"
                return enriched_df, stats
            
            # Find customer ID field in core data
            core_customer_id_field = self._find_customer_id_field(customer_df, "Customer ID")
            if not core_customer_id_field:
                stats["error"] = "Customer ID field not found in core data"
                return enriched_df, stats
            
            # Clean and prepare customer IDs for matching
            enriched_df[customer_id_field] = enriched_df[customer_id_field].astype(str).str.strip()
            customer_df[core_customer_id_field] = customer_df[core_customer_id_field].astype(str).str.strip()
            
            # Count records before enrichment
            initial_count = len(enriched_df)
            
            # Perform left join to add customer data
            enriched_df = enriched_df.merge(
                customer_df,
                left_on=customer_id_field,
                right_on=core_customer_id_field,
                how='left',
                suffixes=('', '_customer_core')
            )
            if 'invalidation_reason_customer_core' in enriched_df.columns:
                reasons = enriched_df['invalidation_reason_customer_core'].fillna('')
                counts = reasons[reasons != ''].value_counts().to_dict()
                stats['invalid_reasons'] = counts
            elif 'invalidation_reason' in enriched_df.columns:
                reasons = enriched_df['invalidation_reason'].fillna('')
                counts = reasons[reasons != ''].value_counts().to_dict()
                stats['invalid_reasons'] = counts
            
            # Calculate enrichment statistics
            successful_matches = ~enriched_df[core_customer_id_field].isna()
            stats["records_enriched"] = successful_matches.sum()
            stats["failed_lookups"] = initial_count - stats["records_enriched"]
            stats["enrichment_fields_added"] = len([col for col in enriched_df.columns if col.endswith('_customer_core') or col in customer_df.columns]) - 1
            
            # Handle specific enrichment options
            if not config.include_rfm_analysis:
                # Remove RFM columns if not requested
                rfm_columns = [col for col in enriched_df.columns if any(rfm in col.lower() for rfm in ['recency', 'frequency', 'monetary', 'rfm_score'])]
                enriched_df = enriched_df.drop(columns=rfm_columns, errors='ignore')
            
            if not config.include_demographics:
                # Remove demographic columns if not requested
                demo_columns = [col for col in enriched_df.columns if any(demo in col.lower() for demo in ['age', 'income', 'education', 'demographic'])]
                enriched_df = enriched_df.drop(columns=demo_columns, errors='ignore')
            
            if not config.include_segmentation:
                # Remove segmentation columns if not requested  
                segment_columns = [col for col in enriched_df.columns if any(seg in col.lower() for seg in ['tier', 'segment', 'category', 'classification'])]
                enriched_df = enriched_df.drop(columns=segment_columns, errors='ignore')
        
        # Record processing time
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        stats["processing_time_ms"] = duration_ms
        
        return enriched_df, stats
    
    def _find_customer_id_field(self, df: pd.DataFrame, preferred_name: str) -> Optional[str]:
        """Find customer ID field in DataFrame with flexible matching"""
        
        # Exact match first
        if preferred_name in df.columns:
            return preferred_name
        
        # Case-insensitive match
        for col in df.columns:
            if col.lower() == preferred_name.lower():
                return col
        
        # Partial matches for common variations
        customer_id_variations = [
            'customer_id', 'customerid', 'customer id', 'cust_id', 'custid',
            'entity_id', 'entityid', 'entity id'
        ]
        
        for variation in customer_id_variations:
            for col in df.columns:
                if variation.lower() in col.lower():
                    return col
        
        return None
    
    def get_enrichment_summary(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable enrichment summary"""
        
        summary = {
            "total_records": stats.get("records_input", 0),
            "enriched_count": stats.get("records_enriched", 0), 
            "enrichment_rate": 0.0,
            "fields_added": stats.get("enrichment_fields_added", 0),
            "processing_time_ms": stats.get("processing_time_ms", 0)
        }
        
        if summary["total_records"] > 0:
            summary["enrichment_rate"] = (summary["enriched_count"] / summary["total_records"]) * 100.0
        
        return summary
    
    def clear_cache(self):
        """Clear cached customer core data"""
        self._customer_core_cache = None
        self._cache_timestamp = None
