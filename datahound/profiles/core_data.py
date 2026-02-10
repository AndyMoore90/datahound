"""Core data collection for customer profiles"""

import hashlib
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
import pandas as pd
import numpy as np
from Levenshtein import distance as levenshtein_distance
import re

from .types import ProfileCoreData, ProfileBuildConfig, ProfileBuildResult, ProfileBuildMode
from ..events.address_utils import normalize_address_street, extract_street_from_full_address


class CustomerProfileCoreDataBuilder:
    """Builds core customer profile data from master parquet files"""
    
    def __init__(self, company: str, parquet_dir: Path, data_dir: Path):
        self.company = company
        self.parquet_dir = parquet_dir
        self.data_dir = data_dir
        
        # Profile storage
        self.profile_file = parquet_dir / "customer_profiles_core_data.parquet"
        self.log_file = data_dir / "logs" / "customer_profile_build_log.jsonl"
        
        # Ensure directories exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Master data cache
        self.master_data: Dict[str, pd.DataFrame] = {}
        
    def load_master_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load a master parquet file and cache it"""
        if table_name in self.master_data:
            return self.master_data[table_name]
        
        parquet_path = self.parquet_dir / f"{table_name.capitalize()}.parquet"
        if not parquet_path.exists():
            self._log_event("warning", f"Master table not found: {table_name}", {
                "table": table_name,
                "path": str(parquet_path)
            })
            return None
        
        try:
            df = pd.read_parquet(parquet_path)
            self.master_data[table_name] = df
            self._log_event("info", f"Loaded master table: {table_name}", {
                "table": table_name,
                "rows": len(df),
                "columns": len(df.columns)
            })
            return df
        except Exception as e:
            self._log_event("error", f"Failed to load master table: {table_name}", {
                "table": table_name,
                "error": str(e)
            })
            return None
    
    def build_customer_profiles(self, config: ProfileBuildConfig) -> ProfileBuildResult:
        """Build customer profiles according to configuration"""
        
        start_time = time.perf_counter()
        
        self._log_event("info", "Starting customer profile build", {
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
        
        # Determine which customers to process
        if config.mode == ProfileBuildMode.NEW_CUSTOMERS_ONLY:
            customer_ids = self._get_new_customer_ids(customers_df)
        else:
            customer_ids = self._get_all_customer_ids(customers_df)
        
        # Apply processing limit
        if config.processing_limit:
            customer_ids = customer_ids[:config.processing_limit]
        
        self._log_event("info", f"Processing {len(customer_ids)} customers", {
            "mode": config.mode.value,
            "total_customers": len(customer_ids)
        })
        
        # Phase 1: Build core data
        phase1_start = time.perf_counter()
        core_profiles = self._build_core_data_phase1(
            customer_ids, customers_df, locations_df, jobs_df, 
            estimates_df, invoices_df, calls_df, memberships_df, config
        )
        phase1_duration = time.perf_counter() - phase1_start
        
        # Phase 2: Enrich profiles (placeholder for now)
        phase2_start = time.perf_counter()
        # TODO: Implement Phase 2 enrichment
        phase2_duration = time.perf_counter() - phase2_start
        
        # Save profiles to parquet
        self._save_profiles_to_parquet(core_profiles, config)
        
        total_duration = time.perf_counter() - start_time
        
        # Build result summary
        result = ProfileBuildResult(
            total_customers_processed=len(customer_ids),
            new_profiles_created=len(core_profiles),  # TODO: Distinguish new vs updated
            existing_profiles_updated=0,  # TODO: Implement update logic
            errors_encountered=0,  # TODO: Track errors
            phase1_duration_seconds=phase1_duration,
            phase2_duration_seconds=phase2_duration,
            total_duration_seconds=total_duration,
            customers_with_locations=sum(1 for p in core_profiles if p.location_count > 0),
            customers_with_duplicate_addresses=sum(1 for p in core_profiles if len(p.duplicate_addresses) > 0),
            customers_with_service_history=sum(1 for p in core_profiles if p.job_count > 0)
        )
        
        self._log_event("info", "Customer profile build completed", result.to_dict())
        
        return result
    
    def _build_core_data_phase1(self, customer_ids: List[str], customers_df: pd.DataFrame,
                               locations_df: Optional[pd.DataFrame], jobs_df: Optional[pd.DataFrame],
                               estimates_df: Optional[pd.DataFrame], invoices_df: Optional[pd.DataFrame],
                               calls_df: Optional[pd.DataFrame], memberships_df: Optional[pd.DataFrame],
                               config: ProfileBuildConfig) -> List[ProfileCoreData]:
        """Phase 1: Build core data for each customer"""
        
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
            
            profile = self._build_single_customer_core_data(
                customer_id, customers_df, locations_df, jobs_df,
                estimates_df, invoices_df, calls_df, memberships_df, config
            )
            
            if profile:
                profiles.append(profile)
            
            # Log progress periodically
            if (i + 1) % config.chunk_size == 0:
                self._log_event("info", f"Processed {i+1}/{total_customers} customers", {
                    "progress_percent": round((i+1)/total_customers * 100, 2),
                    "profiles_created": len(profiles)
                })
        
        if progress_bar:
            progress_bar.progress(1.0)
            status_text.text(f"Completed: {len(profiles)} profiles created")
        
        return profiles
    
    def _build_single_customer_core_data(self, customer_id: str, customers_df: pd.DataFrame,
                                       locations_df: Optional[pd.DataFrame], jobs_df: Optional[pd.DataFrame],
                                       estimates_df: Optional[pd.DataFrame], invoices_df: Optional[pd.DataFrame],
                                       calls_df: Optional[pd.DataFrame], memberships_df: Optional[pd.DataFrame],
                                       config: ProfileBuildConfig) -> Optional[ProfileCoreData]:
        """Build core data for a single customer"""
        
        try:
            profile_id = self._generate_profile_id(customer_id)
            
            profile = ProfileCoreData(
                profile_id=profile_id,
                customer_id=customer_id
            )
            
            # Collect location data
            if locations_df is not None:
                location_data = self._collect_location_data(customer_id, locations_df)
                profile.location_ids = location_data["location_ids"]
                profile.location_count = len(profile.location_ids)
            
            # Collect address data with deduplication
            if locations_df is not None:
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
            
            return profile
            
        except Exception as e:
            self._log_event("error", f"Failed to build profile for customer {customer_id}", {
                "customer_id": customer_id,
                "error": str(e)
            })
            return None
    
    def _collect_location_data(self, customer_id: str, locations_df: pd.DataFrame) -> Dict[str, Any]:
        """Collect location data for a customer"""
        
        # Find customer ID column
        col_map = {col.lower(): col for col in locations_df.columns}
        customer_col = None
        for possible_col in ['customer id', 'customerid', 'customer']:
            if possible_col in col_map:
                customer_col = col_map[possible_col]
                break
        
        if not customer_col:
            return {"location_ids": set()}
        
        # Find location ID column
        location_col = None
        for possible_col in ['location id', 'locationid', 'id']:
            if possible_col in col_map:
                location_col = col_map[possible_col]
                break
        
        if not location_col:
            return {"location_ids": set()}
        
        # Get matching locations
        customer_locations = locations_df[
            locations_df[customer_col].astype(str) == str(customer_id)
        ]
        
        location_ids = set()
        for _, row in customer_locations.iterrows():
            if pd.notna(row[location_col]):
                location_ids.add(str(row[location_col]))
        
        return {"location_ids": location_ids}
    
    def _collect_address_data(self, customer_id: str, locations_df: pd.DataFrame, 
                            config: ProfileBuildConfig) -> Dict[str, Any]:
        """Collect and deduplicate address data for a customer"""
        
        # Find columns
        col_map = {col.lower(): col for col in locations_df.columns}
        customer_col = None
        for possible_col in ['customer id', 'customerid', 'customer']:
            if possible_col in col_map:
                customer_col = col_map[possible_col]
                break
        
        address_col = None
        for possible_col in ['customer address', 'address', 'street address']:
            if possible_col in col_map:
                address_col = col_map[possible_col]
                break
        
        if not customer_col or not address_col:
            return {
                "addresses": set(),
                "normalized_addresses": set(),
                "duplicates": []
            }
        
        # Get customer addresses
        customer_locations = locations_df[
            locations_df[customer_col].astype(str) == str(customer_id)
        ]
        
        addresses = set()
        for _, row in customer_locations.iterrows():
            if pd.notna(row[address_col]):
                address = str(row[address_col]).strip()
                if address:
                    addresses.add(address)
        
        # Normalize addresses
        normalized_addresses = set()
        address_to_normalized = {}
        
        for address in addresses:
            if config.normalize_addresses:
                street = extract_street_from_full_address(address)
                normalized = normalize_address_street(street)
                normalized_addresses.add(normalized)
                address_to_normalized[address] = normalized
            else:
                normalized_addresses.add(address)
                address_to_normalized[address] = address
        
        # Find duplicates using Levenshtein distance
        duplicates = []
        address_list = list(addresses)
        
        for i in range(len(address_list)):
            for j in range(i + 1, len(address_list)):
                addr1 = address_list[i]
                addr2 = address_list[j]
                norm1 = address_to_normalized[addr1]
                norm2 = address_to_normalized[addr2]
                
                # Calculate similarity
                max_len = max(len(norm1), len(norm2))
                if max_len > 0:
                    distance = levenshtein_distance(norm1, norm2)
                    similarity = 1 - (distance / max_len)
                    
                    if similarity >= config.levenshtein_threshold and addr1 != addr2:
                        duplicates.append({
                            "address1": addr1,
                            "address2": addr2,
                            "normalized1": norm1,
                            "normalized2": norm2,
                            "similarity": round(similarity, 3),
                            "levenshtein_distance": distance
                        })
        
        return {
            "addresses": addresses,
            "normalized_addresses": normalized_addresses,
            "duplicates": duplicates
        }
    
    def _collect_service_data(self, customer_id: str, service_df: Optional[pd.DataFrame], 
                            id_column_name: str) -> Tuple[Set[str], int]:
        """Collect service data (jobs, estimates, etc.) for a customer"""
        
        if service_df is None:
            return set(), 0
        
        # Find columns
        col_map = {col.lower(): col for col in service_df.columns}
        
        customer_col = None
        for possible_col in ['customer id', 'customerid', 'customer']:
            if possible_col in col_map:
                customer_col = col_map[possible_col]
                break
        
        id_col = None
        id_variations = [id_column_name.lower(), id_column_name.lower().replace(' ', ''), 'id']
        for possible_col in id_variations:
            if possible_col in col_map:
                id_col = col_map[possible_col]
                break
        
        if not customer_col or not id_col:
            return set(), 0
        
        # Get matching records
        customer_records = service_df[
            service_df[customer_col].astype(str) == str(customer_id)
        ]
        
        service_ids = set()
        for _, row in customer_records.iterrows():
            if pd.notna(row[id_col]):
                service_ids.add(str(row[id_col]))
        
        return service_ids, len(service_ids)
    
    def _get_new_customer_ids(self, customers_df: pd.DataFrame) -> List[str]:
        """Get customer IDs that don't exist in current profiles"""
        
        # Get existing profile customer IDs
        existing_customer_ids = set()
        if self.profile_file.exists():
            try:
                existing_profiles_df = pd.read_parquet(self.profile_file)
                if 'customer_id' in existing_profiles_df.columns:
                    existing_customer_ids = set(existing_profiles_df['customer_id'].astype(str))
            except Exception as e:
                self._log_event("warning", f"Could not load existing profiles: {e}")
        
        # Get all customer IDs from master file
        all_customer_ids = self._get_all_customer_ids(customers_df)
        
        # Return only new ones
        new_customer_ids = [cid for cid in all_customer_ids if cid not in existing_customer_ids]
        
        self._log_event("info", f"Found {len(new_customer_ids)} new customers", {
            "total_customers": len(all_customer_ids),
            "existing_customers": len(existing_customer_ids),
            "new_customers": len(new_customer_ids)
        })
        
        return new_customer_ids
    
    def _get_all_customer_ids(self, customers_df: pd.DataFrame) -> List[str]:
        """Get all customer IDs from customers DataFrame"""
        
        # Find customer ID column
        col_map = {col.lower(): col for col in customers_df.columns}
        customer_id_col = None
        for possible_col in ['customer id', 'customerid', 'id']:
            if possible_col in col_map:
                customer_id_col = col_map[possible_col]
                break
        
        if not customer_id_col:
            raise ValueError("Cannot find Customer ID column in Customers.parquet")
        
        customer_ids = []
        for _, row in customers_df.iterrows():
            if pd.notna(row[customer_id_col]):
                customer_ids.append(str(row[customer_id_col]))
        
        return customer_ids
    
    def _save_profiles_to_parquet(self, profiles: List[ProfileCoreData], config: ProfileBuildConfig):
        """Save profiles to parquet file"""
        
        if not profiles:
            self._log_event("warning", "No profiles to save")
            return
        
        # Convert profiles to DataFrame
        profile_dicts = [profile.to_dict() for profile in profiles]
        new_df = pd.DataFrame(profile_dicts)
        
        # Handle existing profiles
        if self.profile_file.exists() and config.mode == ProfileBuildMode.NEW_CUSTOMERS_ONLY:
            try:
                existing_df = pd.read_parquet(self.profile_file)
                # Append new profiles
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                self._log_event("error", f"Could not load existing profiles, overwriting: {e}")
                combined_df = new_df
        else:
            combined_df = new_df
        
        # Save to parquet
        try:
            combined_df.to_parquet(self.profile_file, index=False)
            self._log_event("info", f"Saved {len(profiles)} profiles to parquet", {
                "file": str(self.profile_file),
                "total_profiles": len(combined_df),
                "new_profiles": len(profiles)
            })
        except Exception as e:
            self._log_event("error", f"Failed to save profiles: {e}")
            raise
    
    def _generate_profile_id(self, customer_id: str) -> str:
        """Generate unique profile ID for customer"""
        # Use company + customer_id + timestamp for uniqueness
        key_string = f"{self.company}|{customer_id}|{datetime.now(UTC).isoformat()}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _log_event(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log event to JSONL file"""
        
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
            "company": self.company,
            "component": "customer_profile_builder"
        }
        
        if details:
            log_entry.update(details)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                import json
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass  # Don't let logging errors break the main process
