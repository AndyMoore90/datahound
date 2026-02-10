"""Core profile manager wrapper (enrichment removed)"""

import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from .types import ProfileBuildConfig, ProfileBuildResult
from .core_data import CustomerProfileCoreDataBuilder


class CustomerProfileEnrichmentBuilder:
    """Deprecated placeholder removed by cleanup."""
    def __init__(self, company: str, parquet_dir: Path, data_dir: Path):
        self.company = company
        self.parquet_dir = parquet_dir
        self.data_dir = data_dir
    def enrich_customer_profiles(self, config: ProfileBuildConfig) -> ProfileBuildResult:
        return ProfileBuildResult(
            total_customers_processed=0,
            new_profiles_created=0,
            existing_profiles_updated=0,
            errors_encountered=0,
            phase1_duration_seconds=0.0,
            phase2_duration_seconds=0.0,
            total_duration_seconds=0.0,
            customers_with_locations=0,
            customers_with_duplicate_addresses=0,
            customers_with_service_history=0,
        )
    
    def _enrich_profiles_batch(self, profiles_df: pd.DataFrame, 
                             config: ProfileBuildConfig) -> List[ProfileEnrichmentData]:
        """Enrich a batch of profiles"""
        
        enriched_profiles = []
        
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
        
        total_profiles = len(profiles_df)
        
        for i, (_, profile_row) in enumerate(profiles_df.iterrows()):
            if progress_bar:
                progress = (i + 1) / total_profiles
                progress_bar.progress(progress)
                status_text.text(f"Enriching profile {i+1}/{total_profiles}: {profile_row['customer_id']}")
            
            enriched_profile = self._enrich_single_profile(profile_row, config)
            if enriched_profile:
                enriched_profiles.append(enriched_profile)
            
            # Log progress periodically
            if (i + 1) % config.chunk_size == 0:
                self._log_event("info", f"Enriched {i+1}/{total_profiles} profiles", {
                    "progress_percent": round((i+1)/total_profiles * 100, 2),
                    "enriched_profiles": len(enriched_profiles)
                })
        
        if progress_bar:
            progress_bar.progress(1.0)
            status_text.text(f"Completed: {len(enriched_profiles)} profiles enriched")
        
        return enriched_profiles
    
    def _enrich_single_profile(self, profile_row: pd.Series, 
                             config: ProfileBuildConfig) -> Optional[ProfileEnrichmentData]:
        """Enrich a single customer profile"""
        
        try:
            profile_id = profile_row['profile_id']
            customer_id = profile_row['customer_id']
            
            enrichment_data = ProfileEnrichmentData(profile_id=profile_id)
            
            # RFM Analysis
            if config.include_rfm:
                rfm_data = self._get_rfm_analysis(customer_id, profile_row)
                if rfm_data:
                    enrichment_data.rfm_recency = rfm_data.get('recency')
                    enrichment_data.rfm_frequency = rfm_data.get('frequency')
                    enrichment_data.rfm_monetary = rfm_data.get('monetary')
                    enrichment_data.rfm_score = rfm_data.get('rfm_score')
                    enrichment_data.rfm_segment = rfm_data.get('rfm_segment')
            
            # Demographics
            if config.include_demographics:
                demo_data = self._get_demographics_data(customer_id, profile_row)
                if demo_data:
                    enrichment_data.household_income = demo_data.get('household_income')
                    enrichment_data.property_value = demo_data.get('property_value')
            
            # Permit data
            if config.include_permits:
                permit_data = self._get_permit_data(customer_id, profile_row)
                if permit_data:
                    enrichment_data.permit_matches = permit_data.get('permit_matches', [])
                    enrichment_data.permit_count = len(enrichment_data.permit_matches)
                    enrichment_data.competitor_permits = permit_data.get('competitor_permits', [])
                    enrichment_data.competitor_permit_count = len(enrichment_data.competitor_permits)
            
            # Marketable status
            if config.include_marketable:
                marketable_data = self._get_marketable_status(customer_id, profile_row)
                if marketable_data:
                    enrichment_data.is_marketable = marketable_data.get('is_marketable')
                    enrichment_data.do_not_call = marketable_data.get('do_not_call')
                    enrichment_data.do_not_service = marketable_data.get('do_not_service')
            
            # Customer segments
            if config.include_segments:
                segment_data = self._get_customer_segments(customer_id, profile_row)
                if segment_data:
                    enrichment_data.customer_tier = segment_data.get('customer_tier')
                    enrichment_data.customer_segment = segment_data.get('customer_segment')
            
            return enrichment_data
            
        except Exception as e:
            self._log_event("error", f"Failed to enrich profile {profile_row.get('profile_id', 'unknown')}", {
                "customer_id": profile_row.get('customer_id', 'unknown'),
                "error": str(e)
            })
            return None
    
    def _get_rfm_analysis(self, customer_id: str, profile_row: pd.Series) -> Optional[Dict[str, Any]]:
        """Get RFM analysis for customer using existing enricher logic"""
        
        try:
            # Create a mock event result to use existing enricher
            from ..events.types import EventResult, EventSeverity
            
            mock_event = EventResult(
                event_type="rfm_analysis",
                entity_type="customer",
                entity_id=customer_id,
                severity=EventSeverity.LOW,
                detected_at=datetime.now(UTC),
                details={"customer_id": customer_id},
                rule_name="rfm_enrichment"
            )
            
            # Use existing enricher logic
            enrichment_config = {"include_rfm": True}
            enriched_payload = self.enricher.enrich_event(mock_event, enrichment_config)
            
            # Extract RFM data from enriched payload
            rfm_data = {}
            for key, value in enriched_payload.items():
                if key.startswith('rfm_'):
                    rfm_data[key.replace('rfm_', '')] = value
            
            return rfm_data if rfm_data else None
            
        except Exception as e:
            self._log_event("warning", f"RFM analysis failed for customer {customer_id}: {e}")
            return None
    
    def _get_demographics_data(self, customer_id: str, profile_row: pd.Series) -> Optional[Dict[str, Any]]:
        """Get demographics data for customer"""
        
        try:
            # Use existing enricher logic for demographics
            from ..events.types import EventResult, EventSeverity
            
            mock_event = EventResult(
                event_type="demographics_analysis",
                entity_type="customer",
                entity_id=customer_id,
                severity=EventSeverity.LOW,
                detected_at=datetime.now(UTC),
                details={"customer_id": customer_id},
                rule_name="demographics_enrichment"
            )
            
            enrichment_config = {"include_demographics": True}
            enriched_payload = self.enricher.enrich_event(mock_event, enrichment_config)
            
            # Extract demographics data
            demo_data = {}
            if 'household_income' in enriched_payload:
                demo_data['household_income'] = enriched_payload['household_income']
            if 'property_value' in enriched_payload:
                demo_data['property_value'] = enriched_payload['property_value']
            
            return demo_data if demo_data else None
            
        except Exception as e:
            self._log_event("warning", f"Demographics analysis failed for customer {customer_id}: {e}")
            return None
    
    def _get_permit_data(self, customer_id: str, profile_row: pd.Series) -> Optional[Dict[str, Any]]:
        """Get permit data for customer"""
        
        try:
            # Use existing enricher logic for permit data
            from ..events.types import EventResult, EventSeverity
            
            mock_event = EventResult(
                event_type="permit_analysis",
                entity_type="customer",
                entity_id=customer_id,
                severity=EventSeverity.LOW,
                detected_at=datetime.now(UTC),
                details={"customer_id": customer_id},
                rule_name="permit_enrichment"
            )
            
            enrichment_config = {"include_permit_data": True}
            enriched_payload = self.enricher.enrich_event(mock_event, enrichment_config)
            
            # Extract permit data
            permit_data = {}
            if 'permit_matches' in enriched_payload:
                permit_data['permit_matches'] = enriched_payload['permit_matches']
            if 'competitor_permits' in enriched_payload:
                permit_data['competitor_permits'] = enriched_payload['competitor_permits']
            
            return permit_data if permit_data else None
            
        except Exception as e:
            self._log_event("warning", f"Permit analysis failed for customer {customer_id}: {e}")
            return None
    
    def _get_marketable_status(self, customer_id: str, profile_row: pd.Series) -> Optional[Dict[str, Any]]:
        """Get marketable status for customer"""
        
        try:
            # Use existing enricher logic for marketable status
            from ..events.types import EventResult, EventSeverity
            
            mock_event = EventResult(
                event_type="marketable_analysis",
                entity_type="customer",
                entity_id=customer_id,
                severity=EventSeverity.LOW,
                detected_at=datetime.now(UTC),
                details={"customer_id": customer_id},
                rule_name="marketable_enrichment"
            )
            
            enrichment_config = {"include_marketable": True}
            enriched_payload = self.enricher.enrich_event(mock_event, enrichment_config)
            
            # Extract marketable data
            marketable_data = {}
            if 'is_marketable' in enriched_payload:
                marketable_data['is_marketable'] = enriched_payload['is_marketable']
            if 'do_not_call' in enriched_payload:
                marketable_data['do_not_call'] = enriched_payload['do_not_call']
            if 'do_not_service' in enriched_payload:
                marketable_data['do_not_service'] = enriched_payload['do_not_service']
            
            return marketable_data if marketable_data else None
            
        except Exception as e:
            self._log_event("warning", f"Marketable analysis failed for customer {customer_id}: {e}")
            return None
    
    def _get_customer_segments(self, customer_id: str, profile_row: pd.Series) -> Optional[Dict[str, Any]]:
        """Get customer segment data"""
        
        try:
            # Use existing enricher logic for segmentation
            from ..events.types import EventResult, EventSeverity
            
            mock_event = EventResult(
                event_type="segmentation_analysis",
                entity_type="customer",
                entity_id=customer_id,
                severity=EventSeverity.LOW,
                detected_at=datetime.now(UTC),
                details={"customer_id": customer_id},
                rule_name="segmentation_enrichment"
            )
            
            enrichment_config = {"include_segmentation": True}
            enriched_payload = self.enricher.enrich_event(mock_event, enrichment_config)
            
            # Extract segmentation data
            segment_data = {}
            if 'customer_tier' in enriched_payload:
                segment_data['customer_tier'] = enriched_payload['customer_tier']
            if 'customer_segment' in enriched_payload:
                segment_data['customer_segment'] = enriched_payload['customer_segment']
            
            return segment_data if segment_data else None
            
        except Exception as e:
            self._log_event("warning", f"Segmentation analysis failed for customer {customer_id}: {e}")
            return None
    
    def _get_new_profiles_for_enrichment(self, core_profiles_df: pd.DataFrame) -> pd.DataFrame:
        """Get profiles that need enrichment (new ones)"""
        
        if not self.enriched_profile_file.exists():
            return core_profiles_df
        
        try:
            existing_enriched_df = pd.read_parquet(self.enriched_profile_file)
            existing_profile_ids = set(existing_enriched_df['profile_id'])
            
            # Return core profiles that don't have enrichment yet
            new_profiles = core_profiles_df[
                ~core_profiles_df['profile_id'].isin(existing_profile_ids)
            ]
            
            self._log_event("info", f"Found {len(new_profiles)} profiles needing enrichment", {
                "total_core_profiles": len(core_profiles_df),
                "existing_enriched": len(existing_enriched_df),
                "new_to_enrich": len(new_profiles)
            })
            
            return new_profiles
            
        except Exception as e:
            self._log_event("warning", f"Could not load existing enriched profiles: {e}")
            return core_profiles_df
    
    def _save_enriched_profiles(self, enriched_profiles: List[ProfileEnrichmentData], 
                              config: ProfileBuildConfig):
        """Save enriched profiles to parquet file"""
        
        if not enriched_profiles:
            self._log_event("warning", "No enriched profiles to save")
            return
        
        # Convert to DataFrame
        enrichment_dicts = [profile.to_dict() for profile in enriched_profiles]
        new_df = pd.DataFrame(enrichment_dicts)
        
        # Handle existing enriched profiles
        if self.enriched_profile_file.exists() and config.mode == ProfileBuildMode.NEW_CUSTOMERS_ONLY:
            try:
                existing_df = pd.read_parquet(self.enriched_profile_file)
                # Append new enriched profiles
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                self._log_event("error", f"Could not load existing enriched profiles, overwriting: {e}")
                combined_df = new_df
        else:
            combined_df = new_df
        
        # Save to parquet
        try:
            combined_df.to_parquet(self.enriched_profile_file, index=False)
            self._log_event("info", f"Saved {len(enriched_profiles)} enriched profiles", {
                "file": str(self.enriched_profile_file),
                "total_enriched_profiles": len(combined_df),
                "new_enriched_profiles": len(enriched_profiles)
            })
        except Exception as e:
            self._log_event("error", f"Failed to save enriched profiles: {e}")
            raise
    
    def _log_event(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log event to JSONL file"""
        
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
            "company": self.company,
            "component": "customer_profile_enrichment"
        }
        
        if details:
            log_entry.update(details)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                import json
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass  # Don't let logging errors break the main process


class CustomerProfileManager:
    """Main manager for core customer profile building only"""
    
    def __init__(self, company: str, parquet_dir: Path, data_dir: Path):
        self.company = company
        self.parquet_dir = parquet_dir
        self.data_dir = data_dir
        
        self.core_builder = CustomerProfileCoreDataBuilder(company, parquet_dir, data_dir)
    
    def build_complete_profiles(self, config: ProfileBuildConfig) -> ProfileBuildResult:
        """Build core customer profiles only."""
        return self.core_builder.build_customer_profiles(config)
    
    def build_legacy_profiles(self, config: ProfileBuildConfig) -> ProfileBuildResult:
        return self.core_builder.build_customer_profiles(config)
