"""Enhanced core customer data builder with RFM, segmentation, and permit matching"""

import hashlib
import time
from datetime import datetime, UTC, date
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
import pandas as pd
import numpy as np

from .types import CustomerProfile, ProfileBuildConfig, ProfileBuildResult, ProfileBuildMode
from .core_data import CustomerProfileCoreDataBuilder
from ..events.address_utils import normalize_address_street, extract_street_from_full_address
from ..events.fuzzy_matching import enhanced_address_match


class EnhancedCustomerCoreDataBuilder(CustomerProfileCoreDataBuilder):
    """Enhanced customer core data builder with RFM analysis, segmentation, and permit matching"""
    
    def __init__(self, company: str, parquet_dir: Path, data_dir: Path):
        super().__init__(company, parquet_dir, data_dir)
        
        # Enhanced profile storage
        self.enhanced_profile_file = parquet_dir / "customer_core_data.parquet"
        
        # Cache for permit and demographics data
        self.permit_data_cache: Optional[pd.DataFrame] = None
        self.demographics_cache: Optional[pd.DataFrame] = None
        
    def build_enhanced_customer_profiles(self, config: ProfileBuildConfig) -> ProfileBuildResult:
        """Build enhanced customer profiles with RFM, segmentation, and permit matching"""
        
        start_time = time.perf_counter()
        
        self._log_event("info", "Starting enhanced customer profile build", {
            "mode": config.mode.value,
            "processing_limit": config.processing_limit,
            "company": self.company
        })
        
        # Load required master tables
        customers_df = self.load_master_table("customers")
        if customers_df is None:
            raise ValueError("Cannot build profiles without Customers.parquet")
        
        # Load supporting tables
        locations_df = self.load_master_table("locations")
        jobs_df = self.load_master_table("jobs")
        estimates_df = self.load_master_table("estimates")
        invoices_df = self.load_master_table("invoices")
        calls_df = self.load_master_table("calls")
        memberships_df = self.load_master_table("memberships")
        
        # Load permit and demographics data
        self._load_permit_data()
        self._load_demographics_data()
        
        # Determine which customers to process
        if config.mode == ProfileBuildMode.NEW_CUSTOMERS_ONLY:
            customer_ids = self._get_new_enhanced_customer_ids(customers_df)
        else:
            customer_ids = self._get_all_customer_ids(customers_df)
        
        # Apply processing limit
        if config.processing_limit:
            customer_ids = customer_ids[:config.processing_limit]
        
        self._log_event("info", f"Processing {len(customer_ids)} customers for enhanced profiles", {
            "mode": config.mode.value,
            "total_customers": len(customer_ids)
        })
        
        # Build enhanced profiles
        enhanced_profiles = self._build_enhanced_profiles(
            customer_ids, customers_df, locations_df, jobs_df, 
            estimates_df, invoices_df, calls_df, memberships_df, config
        )
        
        # Save enhanced profiles
        self._save_enhanced_profiles_to_parquet(enhanced_profiles, config)
        
        total_duration = time.perf_counter() - start_time
        
        # Build result summary
        result = ProfileBuildResult(
            total_customers_processed=len(customer_ids),
            new_profiles_created=len(enhanced_profiles),
            existing_profiles_updated=0,
            errors_encountered=0,
            phase1_duration_seconds=total_duration,
            phase2_duration_seconds=0.0,
            total_duration_seconds=total_duration,
            customers_with_locations=sum(1 for p in enhanced_profiles if p.location_count > 0),
            customers_with_duplicate_addresses=sum(1 for p in enhanced_profiles if len(p.duplicate_addresses) > 0),
            customers_with_service_history=sum(1 for p in enhanced_profiles if p.job_count > 0)
        )
        
        self._log_event("info", "Enhanced customer profile build completed", result.to_dict())
        
        return result
    
    def _build_enhanced_profiles(self, customer_ids: List[str], customers_df: pd.DataFrame,
                               locations_df: Optional[pd.DataFrame], jobs_df: Optional[pd.DataFrame],
                               estimates_df: Optional[pd.DataFrame], invoices_df: Optional[pd.DataFrame],
                               calls_df: Optional[pd.DataFrame], memberships_df: Optional[pd.DataFrame],
                               config: ProfileBuildConfig) -> List[CustomerProfile]:
        """Build enhanced profiles for each customer"""
        
        profiles = []
        
        # Setup progress tracking
        if config.show_progress:
            try:
                import streamlit as st
                progress_bar = st.progress(0)
                status_text = st.empty()
            except:
                progress_bar = None
                status_text = None
        else:
            progress_bar = None
            status_text = None
        
        total_customers = len(customer_ids)
        
        for i, customer_id in enumerate(customer_ids):
            if progress_bar:
                progress = (i + 1) / total_customers
                progress_bar.progress(progress)
                status_text.text(f"Processing customer {i+1}/{total_customers}: {customer_id}")
            
            profile = self._build_single_enhanced_profile(
                customer_id, customers_df, locations_df, jobs_df,
                estimates_df, invoices_df, calls_df, memberships_df, config
            )
            
            if profile:
                profiles.append(profile)
            
            # Log progress periodically
            if (i + 1) % config.chunk_size == 0:
                self._log_event("info", f"Processed {i+1}/{total_customers} enhanced customers", {
                    "progress_percent": round((i+1)/total_customers * 100, 2),
                    "profiles_created": len(profiles)
                })
        
        if progress_bar:
            progress_bar.progress(1.0)
            status_text.text(f"Completed: {len(profiles)} enhanced profiles created")
        
        return profiles
    
    def _build_single_enhanced_profile(self, customer_id: str, customers_df: pd.DataFrame,
                                     locations_df: Optional[pd.DataFrame], jobs_df: Optional[pd.DataFrame],
                                     estimates_df: Optional[pd.DataFrame], invoices_df: Optional[pd.DataFrame],
                                     calls_df: Optional[pd.DataFrame], memberships_df: Optional[pd.DataFrame],
                                     config: ProfileBuildConfig) -> Optional[CustomerProfile]:
        """Build enhanced profile for a single customer"""
        
        try:
            profile_id = self._generate_profile_id(customer_id)
            
            profile = CustomerProfile(
                profile_id=profile_id,
                customer_id=customer_id
            )
            
            # Collect core data (locations, addresses, service history)
            if locations_df is not None:
                location_data = self._collect_location_data(customer_id, locations_df)
                profile.location_ids = location_data["location_ids"]
                profile.location_count = len(profile.location_ids)
                
                address_data = self._collect_address_data(customer_id, locations_df, config)
                profile.customer_addresses = address_data["addresses"]
                profile.normalized_addresses = address_data["normalized_addresses"]
                profile.duplicate_addresses = address_data["duplicates"]
                profile.address_count = len(profile.customer_addresses)
            
            # Collect service history data
            profile.job_ids, profile.job_count = self._collect_service_data(customer_id, jobs_df, "job id")
            profile.estimate_ids, profile.estimate_count = self._collect_service_data(customer_id, estimates_df, "estimate id")
            profile.invoice_ids, profile.invoice_count = self._collect_service_data(customer_id, invoices_df, "invoice id")
            profile.call_ids, profile.call_count = self._collect_service_data(customer_id, calls_df, "call id")
            profile.membership_ids, profile.membership_count = self._collect_service_data(customer_id, memberships_df, "membership id")
            
            # Enhanced enrichment
            if config.include_rfm:
                self._enrich_with_rfm_analysis(profile, jobs_df, invoices_df)
            
            if config.include_demographics:
                self._enrich_with_demographics(profile, locations_df)
            
            if config.include_permits:
                self._enrich_with_permit_data(profile, locations_df)
            
            if config.include_marketable:
                self._enrich_with_marketable_status(profile, customers_df)
            
            if config.include_segments:
                self._enrich_with_customer_segments(profile)
            
            # Set enrichment timestamp
            profile.enriched_at = datetime.now(UTC)
            
            return profile
            
        except Exception as e:
            self._log_event("error", f"Failed to build enhanced profile for customer {customer_id}", {
                "customer_id": customer_id,
                "error": str(e)
            })
            return None
    
    def _enrich_with_rfm_analysis(self, profile: CustomerProfile, 
                                 jobs_df: Optional[pd.DataFrame], 
                                 invoices_df: Optional[pd.DataFrame]):
        """Enrich profile with RFM analysis using existing logic"""
        
        try:
            rfm_data = self._calculate_rfm_scores_enhanced(profile.customer_id, jobs_df, invoices_df)
            if rfm_data:
                profile.rfm_recency = rfm_data.get("rfm_recency_score")
                profile.rfm_frequency = rfm_data.get("rfm_frequency_score")
                profile.rfm_monetary = rfm_data.get("rfm_monetary_value")
                profile.rfm_score = str(rfm_data.get("rfm_score", ""))
                profile.rfm_segment = rfm_data.get("customer_segment")
                
        except Exception as e:
            self._log_event("warning", f"RFM analysis failed for customer {profile.customer_id}: {e}")
    
    def _enrich_with_demographics(self, profile: CustomerProfile, 
                                 locations_df: Optional[pd.DataFrame]):
        """Enrich profile with demographics data"""
        
        try:
            if self.demographics_cache is None or locations_df is None:
                return
            
            # Get customer addresses for demographics matching
            customer_addresses = list(profile.customer_addresses)
            if not customer_addresses:
                return
            
            # Match to demographics by ZIP code (simplified approach)
            for address in customer_addresses:
                try:
                    zip_code = self._extract_zip_from_address(address)
                    if zip_code:
                        demo_data = self._get_demographics_for_zip(zip_code)
                        if demo_data:
                            if demo_data.get("household_income") and pd.notna(demo_data["household_income"]):
                                profile.household_income = demo_data["household_income"]
                            if demo_data.get("property_value") and pd.notna(demo_data["property_value"]):
                                profile.property_value = demo_data["property_value"]
                            break
                except Exception as addr_error:
                    # Log but continue with next address
                    continue
                        
        except Exception as e:
            self._log_event("warning", f"Demographics analysis failed for customer {profile.customer_id}: {e}")
    
    def _enrich_with_permit_data(self, profile: CustomerProfile, 
                                locations_df: Optional[pd.DataFrame]):
        """Enrich profile with permit matching data"""
        
        try:
            if self.permit_data_cache is None:
                return
            
            # Get customer addresses for permit matching
            customer_addresses = list(profile.customer_addresses)
            if not customer_addresses:
                return
            
            all_permit_matches = []
            competitor_permits = []
            
            # Match each address to permits (limit to first 3 addresses for performance)
            for address in customer_addresses[:3]:
                try:
                    permit_matches = self._find_permit_matches_for_address(address)
                    
                    for match in permit_matches:
                        contractor = match.get("contractor", "").lower()
                        
                        # Check if this is a McCullough permit or competitor
                        if self._is_mccullough_contractor(contractor):
                            all_permit_matches.append(match)
                        else:
                            competitor_permits.append(match)
                except Exception:
                    # Continue with next address if this one fails
                    continue
            
            profile.permit_matches = all_permit_matches
            profile.permit_count = len(all_permit_matches)
            profile.competitor_permits = competitor_permits
            profile.competitor_permit_count = len(competitor_permits)
            
        except Exception as e:
            self._log_event("warning", f"Permit analysis failed for customer {profile.customer_id}: {e}")
    
    def _enrich_with_marketable_status(self, profile: CustomerProfile, 
                                      customers_df: Optional[pd.DataFrame]):
        """Enrich profile with marketable status"""
        
        try:
            if customers_df is None:
                return
            
            # Find customer record
            customer_row = self._find_customer_record(profile.customer_id, customers_df)
            if customer_row is None:
                return
            
            # Check for opt-out flags in customer data
            profile.do_not_call = self._check_do_not_call(customer_row)
            profile.do_not_service = self._check_do_not_service(customer_row)
            profile.is_marketable = not (profile.do_not_call or profile.do_not_service)
            
        except Exception as e:
            self._log_event("warning", f"Marketable status analysis failed for customer {profile.customer_id}: {e}")
    
    def _enrich_with_customer_segments(self, profile: CustomerProfile):
        """Enrich profile with customer tier and segment classification"""
        
        try:
            # Calculate customer tier based on service activity and RFM
            tier = self._calculate_customer_tier(profile)
            profile.customer_tier = tier
            
            # Set customer segment (same as RFM segment for now)
            if profile.rfm_segment:
                profile.customer_segment = profile.rfm_segment
            else:
                profile.customer_segment = self._classify_customer_segment(profile)
                
        except Exception as e:
            self._log_event("warning", f"Segmentation failed for customer {profile.customer_id}: {e}")
    
    def _calculate_rfm_scores_enhanced(self, customer_id: str, 
                                     jobs_df: Optional[pd.DataFrame], 
                                     invoices_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate RFM scores using enhanced logic based on existing implementation"""
        
        if jobs_df is None:
            return {}
        
        try:
            # Get customer jobs
            customer_jobs = self._get_customer_jobs(customer_id, jobs_df)
            
            if customer_jobs.empty:
                return {}
            
            # Find completed jobs
            completed_jobs = self._filter_completed_jobs(customer_jobs)
            
            if completed_jobs.empty:
                return {}
            
            # Get job dates and values
            job_dates = self._extract_job_dates(completed_jobs)
            job_values = self._extract_job_values(completed_jobs, invoices_df, customer_id)
            
            if not job_dates:
                return {}
            
            # Calculate RFM components
            today = date.today()
            
            # Recency: Days since last job
            most_recent_date = max(job_dates)
            rfm_recency_value = (today - most_recent_date).days
            
            # Frequency: Number of completed jobs
            rfm_frequency_value = len(completed_jobs)
            
            # Monetary: Total value of jobs/invoices
            rfm_monetary_value = sum(job_values) if job_values else 0
            
            # Calculate scores (1-5 scale)
            rfm_recency_score = self._calculate_recency_score(rfm_recency_value)
            rfm_frequency_score = self._calculate_frequency_score(rfm_frequency_value)
            rfm_monetary_score = self._calculate_monetary_score(rfm_monetary_value)
            
            # Total RFM score and segmentation
            rfm_score = rfm_recency_score + rfm_frequency_score + rfm_monetary_score
            customer_segment = self._determine_customer_segment(rfm_score)
            
            return {
                "rfm_recency_value": rfm_recency_value,
                "rfm_frequency_value": rfm_frequency_value,
                "rfm_monetary_value": rfm_monetary_value,
                "rfm_recency_score": rfm_recency_score,
                "rfm_frequency_score": rfm_frequency_score,
                "rfm_monetary_score": rfm_monetary_score,
                "rfm_score": rfm_score,
                "customer_segment": customer_segment,
                "completed_jobs_found": len(completed_jobs),
                "recent_activity_dates": ";".join([d.isoformat() for d in sorted(job_dates)])
            }
            
        except Exception as e:
            self._log_event("warning", f"RFM calculation failed for customer {customer_id}: {e}")
            return {}
    
    def _get_customer_jobs(self, customer_id: str, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Get jobs for a specific customer"""
        
        col_map = {col.lower(): col for col in jobs_df.columns}
        customer_col = None
        
        for possible_col in ['customer id', 'customerid', 'customer']:
            if possible_col in col_map:
                customer_col = col_map[possible_col]
                break
        
        if not customer_col:
            return pd.DataFrame()
        
        return jobs_df[jobs_df[customer_col].astype(str) == str(customer_id)]
    
    def _filter_completed_jobs(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Filter to completed jobs only"""
        
        col_map = {col.lower(): col for col in jobs_df.columns}
        
        # Look for status column
        status_col = None
        for possible_col in ['job status', 'status', 'job_status']:
            if possible_col in col_map:
                status_col = col_map[possible_col]
                break
        
        if status_col:
            # Filter to completed jobs
            completed_statuses = ['completed', 'complete', 'finished', 'done', 'closed']
            mask = jobs_df[status_col].astype(str).str.lower().isin(completed_statuses)
            return jobs_df[mask]
        
        # If no status column, assume all jobs are completed
        return jobs_df
    
    def _extract_job_dates(self, jobs_df: pd.DataFrame) -> List[date]:
        """Extract job completion dates"""
        
        col_map = {col.lower(): col for col in jobs_df.columns}
        
        # Look for completion date columns
        date_col = None
        for possible_col in ['completion date', 'completed date', 'finish date', 'job date', 'created date']:
            if possible_col in col_map:
                date_col = col_map[possible_col]
                break
        
        if not date_col:
            return []
        
        dates = []
        for _, row in jobs_df.iterrows():
            try:
                date_value = row[date_col]
                if pd.notna(date_value):
                    if isinstance(date_value, str):
                        parsed_date = pd.to_datetime(date_value).date()
                    else:
                        parsed_date = pd.to_datetime(date_value).date()
                    dates.append(parsed_date)
            except:
                continue
        
        return dates
    
    def _extract_job_values(self, jobs_df: pd.DataFrame, 
                           invoices_df: Optional[pd.DataFrame], 
                           customer_id: str) -> List[float]:
        """Extract monetary values from jobs and invoices"""
        
        values = []
        
        # Try to get values from jobs first
        col_map = {col.lower(): col for col in jobs_df.columns}
        value_col = None
        
        for possible_col in ['total', 'amount', 'job total', 'job amount', 'value']:
            if possible_col in col_map:
                value_col = col_map[possible_col]
                break
        
        if value_col:
            for _, row in jobs_df.iterrows():
                try:
                    value = pd.to_numeric(row[value_col], errors='coerce')
                    if pd.notna(value) and value > 0:
                        values.append(float(value))
                except:
                    continue
        
        # If no values from jobs, try invoices
        if not values and invoices_df is not None:
            customer_invoices = self._get_customer_invoices(customer_id, invoices_df)
            
            col_map = {col.lower(): col for col in customer_invoices.columns}
            value_col = None
            
            for possible_col in ['total', 'amount', 'invoice total', 'invoice amount']:
                if possible_col in col_map:
                    value_col = col_map[possible_col]
                    break
            
            if value_col:
                for _, row in customer_invoices.iterrows():
                    try:
                        value = pd.to_numeric(row[value_col], errors='coerce')
                        if pd.notna(value) and value > 0:
                            values.append(float(value))
                    except:
                        continue
        
        return values
    
    def _get_customer_invoices(self, customer_id: str, invoices_df: pd.DataFrame) -> pd.DataFrame:
        """Get invoices for a specific customer"""
        
        col_map = {col.lower(): col for col in invoices_df.columns}
        customer_col = None
        
        for possible_col in ['customer id', 'customerid', 'customer']:
            if possible_col in col_map:
                customer_col = col_map[possible_col]
                break
        
        if not customer_col:
            return pd.DataFrame()
        
        return invoices_df[invoices_df[customer_col].astype(str) == str(customer_id)]
    
    def _calculate_recency_score(self, days_since_last: int) -> int:
        """Calculate recency score (1-5, higher is better)"""
        
        if days_since_last <= 90:
            return 5
        elif days_since_last <= 180:
            return 4
        elif days_since_last <= 365:
            return 3
        elif days_since_last <= 730:
            return 2
        else:
            return 1
    
    def _calculate_frequency_score(self, job_count: int) -> int:
        """Calculate frequency score (1-5, higher is better)"""
        
        if job_count >= 10:
            return 5
        elif job_count >= 5:
            return 4
        elif job_count >= 3:
            return 3
        elif job_count >= 2:
            return 2
        else:
            return 1
    
    def _calculate_monetary_score(self, total_value: float) -> int:
        """Calculate monetary score (1-5, higher is better)"""
        
        if total_value >= 10000:
            return 5
        elif total_value >= 3750:
            return 4
        elif total_value >= 1500:
            return 3
        elif total_value >= 750:
            return 2
        else:
            return 1
    
    def _determine_customer_segment(self, rfm_score: int) -> str:
        """Determine customer segment based on RFM score"""
        
        if rfm_score >= 13:
            return "T1 - Premium Customer"
        elif rfm_score >= 10:
            return "T2 - Steady Customer"
        elif rfm_score >= 7:
            return "T3 - Emerging Customer"
        elif rfm_score >= 4:
            return "T4 - Occasional Customer"
        else:
            return "T5 - Lost Customer"
    
    def _calculate_customer_tier(self, profile: CustomerProfile) -> str:
        """Calculate customer tier based on multiple factors"""
        
        # Use RFM segment as primary tier indicator
        if profile.rfm_segment:
            return profile.rfm_segment
        
        # Fallback tier calculation based on service activity
        total_services = profile.job_count + profile.estimate_count + profile.call_count
        
        if total_services >= 10:
            return "T1 - Premium Customer"
        elif total_services >= 5:
            return "T2 - Steady Customer"
        elif total_services >= 2:
            return "T3 - Emerging Customer"
        elif total_services >= 1:
            return "T4 - Occasional Customer"
        else:
            return "T5 - Inactive Customer"
    
    def _classify_customer_segment(self, profile: CustomerProfile) -> str:
        """Classify customer segment based on profile data"""
        
        # Simple classification based on activity
        if profile.job_count >= 5:
            return "High Value"
        elif profile.job_count >= 2:
            return "Regular Customer"
        elif profile.estimate_count >= 3:
            return "Prospect"
        else:
            return "Inactive"
    
    def _load_permit_data(self):
        """Load permit data for matching"""
        
        try:
            permit_files = [
                Path("global_data/permits/permit_data.csv"),
                *sorted(Path("global_data/permits").glob("permits_austin_*.csv"), reverse=True)  # Most recent first
            ]
            
            permit_path = None
            for path in permit_files:
                if path.exists():
                    permit_path = path
                    break
            
            if permit_path:
                # Load with chunking for large files and limit to reasonable size for performance
                try:
                    # Try to load the full file first
                    file_size = permit_path.stat().st_size / (1024 * 1024)  # Size in MB
                    
                    if file_size > 100:  # If larger than 100MB, load in chunks
                        self._log_event("info", f"Large permit file detected ({file_size:.1f}MB), loading sample")
                        # Load first 100,000 records for performance
                        self.permit_data_cache = pd.read_csv(permit_path, dtype=str, low_memory=False, nrows=100000)
                    else:
                        self.permit_data_cache = pd.read_csv(permit_path, dtype=str, low_memory=False)
                    
                    self._log_event("info", f"Loaded permit data: {len(self.permit_data_cache)} records from {permit_path.name}")
                    
                except Exception as load_error:
                    # If main file fails, try a smaller recent file
                    recent_files = list(Path("global_data/permits").glob("permits_austin_*.csv"))
                    if recent_files:
                        recent_file = sorted(recent_files, reverse=True)[0]  # Most recent
                        self.permit_data_cache = pd.read_csv(recent_file, dtype=str, low_memory=False)
                        self._log_event("info", f"Loaded fallback permit data: {len(self.permit_data_cache)} records from {recent_file.name}")
                    else:
                        raise load_error
            else:
                self._log_event("warning", "No permit data found")
                
        except Exception as e:
            self._log_event("warning", f"Failed to load permit data: {e}")
            self.permit_data_cache = None
    
    def _load_demographics_data(self):
        """Load demographics data for enrichment"""
        
        try:
            demo_path = Path("global_data/demographics/demographics.parquet")
            if demo_path.exists():
                self.demographics_cache = pd.read_parquet(demo_path)
                self._log_event("info", f"Loaded demographics data: {len(self.demographics_cache)} records")
            else:
                self._log_event("warning", "No demographics data found")
                
        except Exception as e:
            self._log_event("warning", f"Failed to load demographics data: {e}")
    
    def _find_permit_matches_for_address(self, address: str) -> List[Dict[str, Any]]:
        """Find permit matches for a given address"""
        
        if self.permit_data_cache is None:
            return []
        
        try:
            matches = []
            normalized_address = normalize_address_street(extract_street_from_full_address(address))
            
            if not normalized_address:
                return []
            
            # Find correct column names in permit data
            address_cols = ['permit_location', 'Address', 'original_address1', 'address', 'location']
            contractor_cols = ['contractor_company_name', 'Contractor Company Name', 'contractor', 'company']
            permit_id_cols = ['permit_number', 'Permit Number', 'permit_id', 'id']
            date_cols = ['issue_date', 'Issued Date', 'issued_date', 'permit_date']
            desc_cols = ['description', 'Description', 'work_description', 'desc']
            
            # Find actual column names
            address_col = self._find_column_name(address_cols, self.permit_data_cache.columns)
            contractor_col = self._find_column_name(contractor_cols, self.permit_data_cache.columns)
            permit_id_col = self._find_column_name(permit_id_cols, self.permit_data_cache.columns)
            date_col = self._find_column_name(date_cols, self.permit_data_cache.columns)
            desc_col = self._find_column_name(desc_cols, self.permit_data_cache.columns)
            
            if not address_col:
                return []
            
            # Batch process for better performance (limit to first 10000 permits for now)
            sample_permits = self.permit_data_cache.head(10000)
            
            for _, row in sample_permits.iterrows():
                permit_address = str(row.get(address_col, ''))
                if permit_address and permit_address != 'nan':
                    try:
                        normalized_permit = normalize_address_street(extract_street_from_full_address(permit_address))
                        if normalized_permit == normalized_address:
                            matches.append({
                                "permit_id": str(row.get(permit_id_col, '')) if permit_id_col else '',
                                "contractor": str(row.get(contractor_col, '')) if contractor_col else '',
                                "permit_date": str(row.get(date_col, '')) if date_col else '',
                                "description": str(row.get(desc_col, '')) if desc_col else '',
                                "address": permit_address
                            })
                    except Exception:
                        continue
            
            return matches
            
        except Exception as e:
            self._log_event("warning", f"Permit matching failed for address {address}: {e}")
            return []
    
    def _find_column_name(self, possible_names: List[str], available_columns: List[str]) -> Optional[str]:
        """Find the actual column name from a list of possibilities"""
        for name in possible_names:
            if name in available_columns:
                return name
        return None
    
    def _is_mccullough_contractor(self, contractor_name: str) -> bool:
        """Check if contractor is McCullough"""
        
        mccullough_names = ['mccullough', 'mccullough heating', 'mccullough hvac']
        return any(name in contractor_name.lower() for name in mccullough_names)
    
    def _extract_zip_from_address(self, address: str) -> Optional[str]:
        """Extract ZIP code from address string"""
        
        import re
        zip_pattern = r'\b\d{5}(?:-\d{4})?\b'
        match = re.search(zip_pattern, address)
        return match.group(0)[:5] if match else None
    
    def _get_demographics_for_zip(self, zip_code: str) -> Optional[Dict[str, Any]]:
        """Get demographics data for a ZIP code"""
        
        if self.demographics_cache is None:
            return None
        
        try:
            # Find the correct ZIP code column name
            zip_col = None
            possible_zip_cols = ['Zip Code', 'zip_code', 'ZIP', 'zipcode', 'postal_code']
            
            for col in possible_zip_cols:
                if col in self.demographics_cache.columns:
                    zip_col = col
                    break
            
            if not zip_col:
                return None
            
            # Find matching ZIP code
            zip_match = self.demographics_cache[
                self.demographics_cache[zip_col].astype(str) == zip_code
            ]
            
            if zip_match.empty:
                return None
            
            row = zip_match.iloc[0]
            
            # Find income and property value columns
            income_col = None
            property_col = None
            
            income_cols = ['Household Income', 'household_income', 'median_household_income', 'income']
            property_cols = ['Property Value', 'property_value', 'median_property_value', 'home_value']
            
            for col in income_cols:
                if col in row.index:
                    income_col = col
                    break
            
            for col in property_cols:
                if col in row.index:
                    property_col = col
                    break
            
            result = {}
            if income_col:
                result["household_income"] = pd.to_numeric(row.get(income_col), errors='coerce')
            if property_col:
                result["property_value"] = pd.to_numeric(row.get(property_col), errors='coerce')
            
            return result if result else None
            
        except Exception as e:
            self._log_event("warning", f"Demographics lookup failed for ZIP {zip_code}: {e}")
            return None
    
    def _find_customer_record(self, customer_id: str, customers_df: pd.DataFrame) -> Optional[pd.Series]:
        """Find customer record in customers DataFrame"""
        
        col_map = {col.lower(): col for col in customers_df.columns}
        customer_col = None
        
        for possible_col in ['customer id', 'customerid', 'id']:
            if possible_col in col_map:
                customer_col = col_map[possible_col]
                break
        
        if not customer_col:
            return None
        
        matches = customers_df[customers_df[customer_col].astype(str) == str(customer_id)]
        return matches.iloc[0] if not matches.empty else None
    
    def _check_do_not_call(self, customer_row: pd.Series) -> bool:
        """Check if customer has do not call flag"""
        
        dnc_columns = ['do not call', 'do_not_call', 'dnc', 'no_call']
        
        for col in dnc_columns:
            if col in customer_row.index:
                value = str(customer_row[col]).lower()
                if value in ['true', 'yes', '1', 'y']:
                    return True
        
        return False
    
    def _check_do_not_service(self, customer_row: pd.Series) -> bool:
        """Check if customer has do not service flag"""
        
        dns_columns = ['do not service', 'do_not_service', 'dns', 'no_service']
        
        for col in dns_columns:
            if col in customer_row.index:
                value = str(customer_row[col]).lower()
                if value in ['true', 'yes', '1', 'y']:
                    return True
        
        return False
    
    def _get_new_enhanced_customer_ids(self, customers_df: pd.DataFrame) -> List[str]:
        """Get customer IDs that don't exist in current enhanced profiles"""
        
        # Get existing enhanced profile customer IDs
        existing_customer_ids = set()
        if self.enhanced_profile_file.exists():
            try:
                existing_profiles_df = pd.read_parquet(self.enhanced_profile_file)
                if 'customer_id' in existing_profiles_df.columns:
                    existing_customer_ids = set(existing_profiles_df['customer_id'].astype(str))
            except Exception as e:
                self._log_event("warning", f"Could not load existing enhanced profiles: {e}")
        
        # Get all customer IDs from master file
        all_customer_ids = self._get_all_customer_ids(customers_df)
        
        # Return only new ones
        new_customer_ids = [cid for cid in all_customer_ids if cid not in existing_customer_ids]
        
        self._log_event("info", f"Found {len(new_customer_ids)} new customers for enhanced profiles", {
            "total_customers": len(all_customer_ids),
            "existing_enhanced": len(existing_customer_ids),
            "new_customers": len(new_customer_ids)
        })
        
        return new_customer_ids
    
    def _save_enhanced_profiles_to_parquet(self, profiles: List[CustomerProfile], config: ProfileBuildConfig):
        """Save enhanced profiles to parquet file"""
        
        if not profiles:
            self._log_event("warning", "No enhanced profiles to save")
            return
        
        # Convert profiles to DataFrame
        profile_dicts = [profile.to_dict() for profile in profiles]
        new_df = pd.DataFrame(profile_dicts)
        
        # Handle existing enhanced profiles
        if self.enhanced_profile_file.exists() and config.mode == ProfileBuildMode.NEW_CUSTOMERS_ONLY:
            try:
                existing_df = pd.read_parquet(self.enhanced_profile_file)
                # Append new enhanced profiles
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                self._log_event("error", f"Could not load existing enhanced profiles, overwriting: {e}")
                combined_df = new_df
        else:
            combined_df = new_df
        
        # Save to parquet
        try:
            combined_df.to_parquet(self.enhanced_profile_file, index=False)
            self._log_event("info", f"Saved {len(profiles)} enhanced profiles to parquet", {
                "file": str(self.enhanced_profile_file),
                "total_enhanced_profiles": len(combined_df),
                "new_enhanced_profiles": len(profiles)
            })
        except Exception as e:
            self._log_event("error", f"Failed to save enhanced profiles: {e}")
            raise
