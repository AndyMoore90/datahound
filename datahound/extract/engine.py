"""Custom extraction engine for event data with time filtering and enrichment"""

import json
import re
import time
from datetime import datetime, UTC, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from .types import (
    ExtractionConfig, ExtractionResult, ExtractionBatch, 
    TimeFilter, TimeFilterType, NumericFilter, NumericFilterType, EnrichmentConfig
)
from .enrichment import DataEnrichment


class CustomExtractionEngine:
    """Engine for extracting and enriching event data based on configurable rules"""
    
    def __init__(self, company: str, data_dir: Path, parquet_dir: Path):
        self.company = company
        self.data_dir = data_dir
        self.parquet_dir = parquet_dir
        
        # Set up directories - events are in the company-specific data directory
        self.events_dir = data_dir / "events"  
        self.master_files_dir = self.events_dir / "master_files"
        self.recent_events_dir = data_dir / "recent_events"
        self.recent_events_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize enrichment engine
        self.enrichment = DataEnrichment(company, parquet_dir)
        
        from central_logging.config import extraction_dir
        self.logs_dir = extraction_dir(company)
        self.log_file = self.logs_dir / "custom_extraction.jsonl"
    
    def extract_single(self, config: ExtractionConfig) -> ExtractionResult:
        """Execute a single extraction configuration"""
        
        # Special case for Second Chance Leads - use dedicated method
        if config.source_event_type == "second_chance_leads":
            return self.extract_second_chance_leads()
        
        start_time = time.perf_counter()
        
        # Log extraction start
        self._log_event("info", f"Starting extraction: {config.name}", {
            "config_name": config.name,
            "source_event": config.source_event_type,
            "time_filter": config.time_filter.__dict__,
            "enrichment_enabled": config.enrichment.include_customer_core_data
        })
        
        try:
            # Load source event data
            event_df = self._load_event_data(config)
            if event_df is None:
                return ExtractionResult(
                    config_name=config.name,
                    success=False,
                    records_found=0,
                    records_enriched=0,
                    records_saved=0,
                    output_file=self.recent_events_dir / config.output_file_name,
                    duration_ms=0,
                    error_message=f"Could not load source data: {config.source_file_name}"
                )
            invalid_counts: Dict[str, int] = {}
            
            # Apply time filter if enabled
            filtered_df = event_df
            if config.time_filter.enabled:
                filtered_df = self._apply_time_filter(filtered_df, config.time_filter)
            
            # Apply numeric filter if enabled
            if config.numeric_filter.enabled:
                filtered_df = self._apply_numeric_filter(filtered_df, config.numeric_filter)
            
            records_after_filter = len(filtered_df)
            
            self._log_event("info", f"Time filter applied", {
                "config_name": config.name,
                "records_before": len(event_df),
                "records_after": records_after_filter,
                "filter_field": config.time_filter.field_name
            })
            
            # Apply record limit if specified
            if config.max_records and len(filtered_df) > config.max_records:
                filtered_df = filtered_df.head(config.max_records)
                records_after_filter = len(filtered_df)
            
            # Enrich data if requested
            enriched_df = filtered_df
            enrichment_stats = {}
            records_enriched = 0
            
            if config.enrichment.include_customer_core_data and not filtered_df.empty:
                enriched_df, enrichment_stats = self.enrichment.enrich_event_data(
                    filtered_df, config.enrichment
                )
                records_enriched = enrichment_stats.get("records_enriched", 0)
                invalid_counts = enrichment_stats.get("invalid_reasons", {}) or {}
                
                self._log_event("info", f"Data enrichment completed", {
                    "config_name": config.name,
                    "enrichment_rate": f"{enrichment_stats.get('records_enriched', 0) / max(1, len(filtered_df)) * 100:.1f}%",
                    "fields_added": enrichment_stats.get("enrichment_fields_added", 0)
                })
            
            # Save results
            output_file = self.recent_events_dir / config.output_file_name
            records_saved = 0
            
            if not enriched_df.empty:
                # Add extraction metadata
                enriched_df['extraction_timestamp'] = datetime.now(UTC).isoformat()
                enriched_df['extraction_config'] = config.name
                enriched_df['source_event_type'] = config.source_event_type
                if config.invalid_reason_field and config.invalid_reason_field not in enriched_df.columns:
                    enriched_df[config.invalid_reason_field] = ""
                
                enriched_df.to_parquet(output_file, index=False)
                records_saved = len(enriched_df)
                
                self._log_event("info", f"Extraction results saved", {
                    "config_name": config.name,
                    "output_file": str(output_file),
                    "records_saved": records_saved
                })
            else:
                self._log_event("info", f"No records to save - empty result set", {
                    "config_name": config.name,
                    "intended_output_file": str(output_file),
                    "records_after_filter": records_after_filter
                })
            
            # Calculate duration
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            
            # Create result
            result = ExtractionResult(
                config_name=config.name,
                success=True,
                records_found=records_after_filter,
                records_enriched=records_enriched,
                records_saved=records_saved,
                output_file=output_file,
                duration_ms=duration_ms,
                enrichment_stats=enrichment_stats
            )
            
            self._log_event("info", f"Extraction completed successfully", {
                "config_name": config.name,
                "duration_ms": duration_ms,
                "success": True,
                "result_summary": {
                    "records_found": result.records_found,
                    "records_enriched": result.records_enriched,
                    "enrichment_rate": f"{result.enrichment_rate:.1f}%",
                    "invalid_counts": invalid_counts
                }
            })
            if config.invalid_reason_field and not invalid_counts:
                self._log_event("warning", "No invalidation reasons detected after extraction", {
                    "config_name": config.name,
                    "expected_column": config.invalid_reason_field,
                    "records_saved": result.records_saved
                })
            
            return result
            
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = str(e)
            
            self._log_event("error", f"Extraction failed: {config.name}", {
                "config_name": config.name,
                "error": error_msg,
                "duration_ms": duration_ms
            })
            
            return ExtractionResult(
                config_name=config.name,
                success=False,
                records_found=0,
                records_enriched=0,
                records_saved=0,
                output_file=self.recent_events_dir / config.output_file_name,
                duration_ms=duration_ms,
                error_message=error_msg
            )
    
    def extract_batch(self, batch: ExtractionBatch) -> List[ExtractionResult]:
        """Execute multiple extractions in a batch"""
        
        self._log_event("info", f"Starting batch extraction", {
            "company": batch.company,
            "total_extractions": len(batch.extractions),
            "enabled_extractions": len(batch.enabled_extractions),
            "run_timestamp": batch.run_timestamp.isoformat()
        })
        
        results = []
        for config in batch.enabled_extractions:
            result = self.extract_single(config)
            results.append(result)
        
        # Log batch summary
        successful = sum(1 for r in results if r.success)
        total_records = sum(r.records_saved for r in results)
        
        self._log_event("info", f"Batch extraction completed", {
            "successful_extractions": successful,
            "failed_extractions": len(results) - successful,
            "total_records_extracted": total_records
        })
        
        return results
    
    def _load_event_data(self, config: ExtractionConfig) -> Optional[pd.DataFrame]:
        """Load event data from master parquet file"""
        
        source_file = self.master_files_dir / config.source_file_name
        
        if not source_file.exists():
            # Try alternative locations
            alt_locations = [
                self.parquet_dir / config.source_file_name,
                self.events_dir / config.source_file_name
            ]
            
            if config.source_event_type == "second_chance_leads":
                alt_locations.append(Path("data") / self.company / "second_chance" / config.source_file_name)
            
            for alt_file in alt_locations:
                if alt_file.exists():
                    source_file = alt_file
                    break
            else:
                return None
        
        try:
            df = pd.read_parquet(source_file)
            return df
        except Exception as e:
            self._log_event("error", f"Failed to load event data", {
                "source_file": str(source_file),
                "error": str(e)
            })
            return None
    
    def _apply_time_filter(self, df: pd.DataFrame, time_filter: TimeFilter) -> pd.DataFrame:
        """Apply time-based filtering to event data"""
        
        if df.empty or not time_filter.enabled:
            return df
        
        # Find the time filter field
        filter_field = self._find_date_field(df, time_filter.field_name)
        if not filter_field:
            # Return all data if filter field not found
            return df
        
        # Parse dates in the field
        df_filtered = df.copy()
        try:
            df_filtered[filter_field] = pd.to_datetime(df_filtered[filter_field], errors='coerce')
        except Exception:
            return df
        
        # Remove rows with unparseable dates
        df_filtered = df_filtered.dropna(subset=[filter_field])
        
        if df_filtered.empty:
            return df_filtered
        
        # Apply the appropriate filter
        today = date.today()
        
        if time_filter.filter_type == TimeFilterType.DAYS_BACK:
            cutoff_date = today - timedelta(days=time_filter.days_back)
            mask = df_filtered[filter_field].dt.date >= cutoff_date
            
        elif time_filter.filter_type == TimeFilterType.DATE_RANGE:
            start_mask = df_filtered[filter_field].dt.date >= time_filter.start_date
            end_mask = df_filtered[filter_field].dt.date <= time_filter.end_date
            mask = start_mask & end_mask
            
        elif time_filter.filter_type == TimeFilterType.STATIC_DATE:
            mask = df_filtered[filter_field].dt.date == time_filter.static_date
            
        else:
            return df_filtered
        
        return df_filtered[mask]
    
    def _apply_numeric_filter(self, df: pd.DataFrame, numeric_filter: NumericFilter) -> pd.DataFrame:
        """Apply numeric filtering to event data"""
        
        if df.empty or not numeric_filter.enabled:
            return df
        
        # Find the numeric filter field
        filter_field = self._find_numeric_field(df, numeric_filter.field_name)
        if not filter_field:
            # Return all data if filter field not found
            self._log_event("warning", f"Numeric filter field '{numeric_filter.field_name}' not found", {
                "available_columns": list(df.columns)[:10]
            })
            return df
        
        # Convert to numeric, coercing errors to NaN
        df_filtered = df.copy()
        try:
            df_filtered[filter_field] = pd.to_numeric(df_filtered[filter_field], errors='coerce')
        except Exception as e:
            self._log_event("error", f"Error converting '{filter_field}' to numeric", {
                "error": str(e),
                "field": filter_field
            })
            return df
        
        # Remove rows with unparseable numeric values
        df_filtered = df_filtered.dropna(subset=[filter_field])
        
        if df_filtered.empty:
            return df_filtered
        
        # Apply the appropriate numeric filter
        if numeric_filter.filter_type == NumericFilterType.LESS_THAN:
            mask = df_filtered[filter_field] < numeric_filter.value
            
        elif numeric_filter.filter_type == NumericFilterType.GREATER_THAN:
            mask = df_filtered[filter_field] > numeric_filter.value
            
        elif numeric_filter.filter_type == NumericFilterType.EQUALS:
            mask = df_filtered[filter_field] == numeric_filter.value
            
        elif numeric_filter.filter_type == NumericFilterType.LESS_THAN_OR_EQUAL:
            mask = df_filtered[filter_field] <= numeric_filter.value
            
        elif numeric_filter.filter_type == NumericFilterType.GREATER_THAN_OR_EQUAL:
            mask = df_filtered[filter_field] >= numeric_filter.value
            
        elif numeric_filter.filter_type == NumericFilterType.BETWEEN:
            mask = (df_filtered[filter_field] >= numeric_filter.min_value) & (df_filtered[filter_field] <= numeric_filter.max_value)
            
        else:
            return df_filtered
        
        filtered_result = df_filtered[mask]
        
        # Log numeric filtering results
        self._log_event("info", f"Numeric filter applied", {
            "field": filter_field,
            "filter_type": numeric_filter.filter_type.value,
            "value": numeric_filter.value,
            "min_value": numeric_filter.min_value,
            "max_value": numeric_filter.max_value,
            "records_before": len(df_filtered),
            "records_after": len(filtered_result)
        })
        
        return filtered_result
    
    def _find_numeric_field(self, df: pd.DataFrame, preferred_name: str) -> Optional[str]:
        """Find numeric field in DataFrame with flexible matching"""
        
        # Exact match first
        if preferred_name in df.columns:
            return preferred_name
        
        # Case-insensitive match
        for col in df.columns:
            if col.lower() == preferred_name.lower():
                return col
        
        # Partial matches for the preferred name
        for col in df.columns:
            if preferred_name.lower() in col.lower():
                return col
        
        return None
    
    def _find_date_field(self, df: pd.DataFrame, preferred_name: str) -> Optional[str]:
        """Find date field in DataFrame with flexible matching"""
        
        # Exact match first
        if preferred_name in df.columns:
            return preferred_name
        
        # Case-insensitive match
        for col in df.columns:
            if col.lower() == preferred_name.lower():
                return col
        
        # Partial matches for common date fields
        date_field_variations = [
            'completion_date', 'detected_at', 'created_date', 'updated_at',
            'timestamp', 'date', 'last_updated', 'scan_timestamp'
        ]
        
        for variation in date_field_variations:
            for col in df.columns:
                if variation.lower() in col.lower():
                    return col
        
        return None
    
    def get_available_extractions(self) -> List[ExtractionConfig]:
        """Get list of pre-configured extraction templates"""
        
        return [
            ExtractionConfig.create_recent_cancellations(14),
            ExtractionConfig.create_recent_lost_customers(30),
            ExtractionConfig.create_recent_aging_systems(7),
            ExtractionConfig.create_recent_second_chance_leads(),
            
            # Additional templates
            ExtractionConfig(
                name="Recent Unsold Estimates",
                description="Unsold estimates from the last 21 days with customer data",
                enabled=False,
                source_event_type="unsold_estimates",
                source_file_name="unsold_estimates_master.parquet",
                time_filter=TimeFilter(
                    enabled=True,
                    filter_type=TimeFilterType.DAYS_BACK,
                    field_name="detected_at",
                    days_back=21
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
                output_file_name="recent_unsold_estimates.parquet"
            ),
            
            ExtractionConfig(
                name="Recent Overdue Maintenance",
                description="Overdue maintenance opportunities from the last 14 days",
                enabled=False,
                source_event_type="overdue_maintenance",
                source_file_name="overdue_maintenance_master.parquet",
                time_filter=TimeFilter(
                    enabled=True,
                    filter_type=TimeFilterType.DAYS_BACK,
                    field_name="detected_at",
                    days_back=14
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
                output_file_name="recent_overdue_maintenance.parquet"
            )
        ]
    
    def extract_second_chance_leads(self) -> ExtractionResult:
        """Extract second chance leads with phone normalization and enrichment"""
        
        start_time = time.perf_counter()
        
        second_chance_dir = Path("data") / self.company / "second_chance"
        source_file = second_chance_dir / "second_chance_leads.parquet"

        self._log_event("info", "Starting Second Chance Leads extraction", {
            "company": self.company,
            "source_file": str(source_file),
        })

        try:
            if not source_file.exists():
                self._log_event("error", "Second chance leads file not found", {
                    "file_path": str(source_file)
                })
                return ExtractionResult(
                    config_name="Recent Second Chance Leads",
                    success=False,
                    records_found=0,
                    records_enriched=0,
                    records_saved=0,
                    output_file=self.recent_events_dir / "recent_second_chance_leads.parquet",
                    duration_ms=0,
                    error_message="Second chance leads file not found"
                )
            
            # Load and filter data
            df = pd.read_parquet(source_file)
            second_chance_df = df[df['is_second_chance_lead'] == True].copy()
            
            self._log_event("info", "Second chance leads filtered", {
                "total_records": len(df),
                "second_chance_records": len(second_chance_df)
            })
            
            if second_chance_df.empty:
                self._log_event("info", "No second chance leads found", {})
                return ExtractionResult(
                    config_name="Recent Second Chance Leads",
                    success=True,
                    records_found=0,
                    records_enriched=0,
                    records_saved=0,
                    output_file=self.recent_events_dir / "recent_second_chance_leads.parquet",
                    duration_ms=int((time.perf_counter() - start_time) * 1000)
                )
            
            # Normalize phone numbers and enrich with customer data
            enriched_df = self._enrich_second_chance_leads(second_chance_df)
            
            # Add Event Type column
            enriched_df['Event Type'] = 'Second Chance'
            
            # Save results
            output_file = self.recent_events_dir / "recent_second_chance_leads.parquet"
            enriched_df.to_parquet(output_file, index=False)
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            
            self._log_event("info", "Second Chance Leads extraction completed", {
                "records_found": len(second_chance_df),
                "records_enriched": len(enriched_df),
                "output_file": str(output_file),
                "duration_ms": duration_ms
            })
            
            return ExtractionResult(
                config_name="Recent Second Chance Leads",
                success=True,
                records_found=len(second_chance_df),
                records_enriched=len(enriched_df),
                records_saved=len(enriched_df),
                output_file=output_file,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = str(e)
            
            self._log_event("error", f"Second Chance Leads extraction failed: {error_msg}", {
                "duration_ms": duration_ms
            })
            
            return ExtractionResult(
                config_name="Recent Second Chance Leads",
                success=False,
                records_found=0,
                records_enriched=0,
                records_saved=0,
                output_file=self.recent_events_dir / "recent_second_chance_leads.parquet",
                duration_ms=duration_ms,
                error_message=error_msg
            )
    
    def _enrich_second_chance_leads(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich second chance leads with customer and core data"""
        
        try:
            # Load customer data
            customers_file = self.parquet_dir / "Customers.parquet"
            if not customers_file.exists():
                self._log_event("error", "Customers file not found", {
                    "file_path": str(customers_file)
                })
                return df
            
            customers_df = pd.read_parquet(customers_file)
            
            # Load customer core data
            core_data_file = self.parquet_dir / "customer_core_data.parquet"
            core_data_df = None
            if core_data_file.exists():
                core_data_df = pd.read_parquet(core_data_file)
            else:
                self._log_event("warning", "Customer core data file not found", {
                    "file_path": str(core_data_file)
                })
            
            # Normalize phone numbers for matching
            df['normalized_phone'] = df['customer_phone'].astype(str).str.replace(r'[^0-9]', '', regex=True)
            
            # Create phone lookup for customers
            customers_df['phone_list'] = customers_df['Phone Number'].fillna('').astype(str)
            customers_df['normalized_phones'] = customers_df['phone_list'].apply(self._normalize_phone_list)
            phone_lookup = self._build_phone_lookup(customers_df)

            matched_customers = []
            for normalized_phone in df['normalized_phone']:
                customer_match = phone_lookup.get(normalized_phone)
                matched_customers.append(dict(customer_match) if customer_match else {})
            
            # Merge customer data
            customer_df = pd.DataFrame(matched_customers)
            enriched_df = pd.concat([df.reset_index(drop=True), customer_df.reset_index(drop=True)], axis=1)
            
            # Merge core data if available
            if core_data_df is not None and 'Customer ID' in enriched_df.columns:
                enriched_df = enriched_df.merge(
                    core_data_df,
                    left_on='Customer ID',
                    right_on='customer_id',
                    how='left',
                    suffixes=('', '_core')
                )
            
            # Select and rename columns as specified
            final_columns = {
                'customer_phone': 'Customer Phone',
                'analysis_timestamp': 'Analysis Timestamp',
                'is_second_chance_lead': 'Second Chance Lead',
                'reasoning': 'LLM Reasoning',
                'was_customer_call': 'Customer Call',
                'was_service_request': 'Service Request',
                'was_booked': 'Was Booked',
                'booking_failure_reason': 'Failed Booking Reason',
                'conversion_potential': 'Conversion Potential',
                'agent_training_notes': 'Training Suggestion',
                'call_id': 'Call ID',
                'primary_call_direction': 'Call Direction',
                'primary_call_date': 'Call Date',
                'primary_call_agent_name': 'Call Agent',
                'Customer ID': 'Customer ID',
                'Customer Name': 'Customer Name',
                'Full Address': 'Address',
                'Do Not Mail': 'Do Not Mail',
                'Do Not Service': 'Do Not Service',
                'customer_tier': 'Customer Tier',
                'household_income': 'Household Income',
                'property_value': 'Property Value',
                'rfm_score': 'RFM Score'
            }
            
            # Select only columns that exist
            available_columns = {k: v for k, v in final_columns.items() if k in enriched_df.columns}
            result_df = enriched_df[list(available_columns.keys())].rename(columns=available_columns)
            
            self._log_event("info", "Second chance leads enriched", {
                "original_columns": len(df.columns),
                "enriched_columns": len(result_df.columns),
                "columns_added": len(available_columns)
            })
            
            return result_df
            
        except Exception as e:
            self._log_event("error", f"Error enriching second chance leads: {str(e)}", {})
            return df
    
    def _normalize_phone_list(self, phone_string: str) -> List[str]:
        """Normalize a list of phone numbers"""
        if pd.isna(phone_string) or phone_string == '':
            return []
        
        phones = [p.strip() for p in phone_string.split(',')]
        normalized = []
        for phone in phones:
            if phone:
                normalized.append(re.sub(r'[^0-9]', '', phone))
        
        return normalized

    def _build_phone_lookup(self, customers_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        for _, customer in customers_df.iterrows():
            normalized_phones: List[str] = customer['normalized_phones']
            if not normalized_phones:
                continue
            customer_data = customer.drop(labels=['normalized_phones']).to_dict()
            for phone in normalized_phones:
                if phone and phone not in lookup:
                    lookup[phone] = customer_data
        return lookup

    def _log_event(self, level: str, message: str, details: Dict[str, Any]):
        """Log extraction events to JSONL file"""
        
        log_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "company": self.company,
            "level": level,
            "message": message,
            "component": "CustomExtractionEngine",
            "details": details
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record) + "\n")
        except Exception:
            pass  # Don't fail extraction if logging fails
