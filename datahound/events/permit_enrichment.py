"""Permit data enrichment utilities for event payloads"""

from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from .fuzzy_matching import enhanced_location_permit_matching
from .address_utils import normalize_address_street


class PermitEnricher:
    """Enriches event payloads with permit data through address matching"""
    
    def __init__(self, permit_data: pd.DataFrame):
        self.permit_data = permit_data
        self._permit_cache = {}
        self._build_permit_indexes()
    
    def _build_permit_indexes(self):
        """Build indexes for fast permit lookups"""
        
        if self.permit_data.empty:
            return
        
        # Build contractor index
        self.contractor_index = {}
        if 'Contractor Company Name' in self.permit_data.columns:
            for _, row in self.permit_data.iterrows():
                contractor = str(row['Contractor Company Name']).strip()
                if contractor and contractor.lower() != 'nan':
                    if contractor not in self.contractor_index:
                        self.contractor_index[contractor] = []
                    self.contractor_index[contractor].append(row.to_dict())
        
        # Build date index for faster lookups
        self.date_index = {}
        if 'issue_date' in self.permit_data.columns:
            self.permit_data['issue_date_parsed'] = pd.to_datetime(self.permit_data['issue_date'], errors='coerce')
            for _, row in self.permit_data.iterrows():
                if pd.notna(row.get('issue_date_parsed')):
                    year_month = row['issue_date_parsed'].strftime('%Y-%m')
                    if year_month not in self.date_index:
                        self.date_index[year_month] = []
                    self.date_index[year_month].append(row.to_dict())
    
    def enrich_with_permit_data(self, event_details: Dict[str, Any], 
                               customer_address: Optional[str] = None,
                               location_address: Optional[str] = None,
                               customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Add permit data enrichment to event details"""
        
        enriched = event_details.copy()
        
        # Try to find address for matching
        address_to_match = customer_address or location_address
        
        if not address_to_match:
            return enriched
        
        # Get permit matches for this address
        permit_matches = self._get_permit_matches_for_address(address_to_match)
        
        if permit_matches:
            # Add permit summary data
            enriched.update(self._calculate_permit_summary(permit_matches))
            
            # Add recent activity
            recent_permits = self._get_recent_permits(permit_matches, days_back=365)
            enriched.update(self._calculate_recent_permit_activity(recent_permits))
            
            # Add contractor analysis
            enriched.update(self._analyze_contractors(permit_matches))
        
        return enriched
    
    def _get_permit_matches_for_address(self, address: str) -> List[Dict]:
        """Get permit matches for a given address"""
        
        if address in self._permit_cache:
            return self._permit_cache[address]
        
        matches = []
        normalized_address = normalize_address_street(address)
        
        # Simple matching for now - could be enhanced with fuzzy matching
        for _, row in self.permit_data.iterrows():
            permit_address = str(row.get('permit_location', ''))
            if permit_address and normalize_address_street(permit_address) == normalized_address:
                matches.append(row.to_dict())
        
        self._permit_cache[address] = matches
        return matches
    
    def _calculate_permit_summary(self, permits: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from permits"""
        
        if not permits:
            return {
                "permit_count_total": 0,
                "permit_count_mechanical": 0,
                "has_permits": False
            }
        
        mechanical_permits = [p for p in permits if p.get('permittype') == 'MP']
        
        return {
            "permit_count_total": len(permits),
            "permit_count_mechanical": len(mechanical_permits),
            "has_permits": True,
            "earliest_permit_date": min([p.get('issue_date', '') for p in permits if p.get('issue_date')]),
            "latest_permit_date": max([p.get('issue_date', '') for p in permits if p.get('issue_date')])
        }
    
    def _get_recent_permits(self, permits: List[Dict], days_back: int = 365) -> List[Dict]:
        """Filter permits to recent ones"""
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent = []
        
        for permit in permits:
            issue_date = permit.get('issue_date')
            if issue_date:
                try:
                    permit_date = pd.to_datetime(issue_date)
                    if permit_date >= cutoff_date:
                        recent.append(permit)
                except:
                    continue
        
        return recent
    
    def _calculate_recent_permit_activity(self, recent_permits: List[Dict]) -> Dict[str, Any]:
        """Calculate recent permit activity metrics"""
        
        if not recent_permits:
            return {
                "permits_last_year": 0,
                "mechanical_permits_last_year": 0,
                "recent_permit_activity": False
            }
        
        mechanical_recent = [p for p in recent_permits if p.get('permittype') == 'MP']
        
        return {
            "permits_last_year": len(recent_permits),
            "mechanical_permits_last_year": len(mechanical_recent),
            "recent_permit_activity": len(recent_permits) > 0,
            "most_recent_permit_date": max([p.get('issue_date', '') for p in recent_permits if p.get('issue_date')])
        }
    
    def _analyze_contractors(self, permits: List[Dict]) -> Dict[str, Any]:
        """Analyze contractor patterns in permits"""
        
        if not permits:
            return {
                "unique_contractors": 0,
                "mccullough_permits": 0,
                "other_contractor_permits": 0,
                "last_contractor": "",
                "has_mccullough_permits": False
            }
        
        contractors = []
        mccullough_count = 0
        
        for permit in permits:
            contractor = permit.get('Contractor Company Name', '').strip()
            if contractor and contractor.lower() != 'nan':
                contractors.append(contractor)
                
                # Check if it's McCullough (case insensitive)
                if 'mccullough' in contractor.lower():
                    mccullough_count += 1
        
        # Get most recent contractor
        last_contractor = ""
        if permits:
            # Sort by date and get most recent
            dated_permits = [(p, p.get('issue_date', '')) for p in permits if p.get('issue_date')]
            if dated_permits:
                try:
                    sorted_permits = sorted(dated_permits, key=lambda x: pd.to_datetime(x[1]), reverse=True)
                    last_contractor = sorted_permits[0][0].get('Contractor Company Name', '').strip()
                except:
                    pass
        
        return {
            "unique_contractors": len(set(contractors)),
            "mccullough_permits": mccullough_count,
            "other_contractor_permits": len(permits) - mccullough_count,
            "last_contractor": last_contractor,
            "has_mccullough_permits": mccullough_count > 0,
            "contractors_used": list(set(contractors))
        }


def create_permit_enricher(permit_data_path: Path) -> Optional[PermitEnricher]:
    """Create a permit enricher from permit data file"""
    
    if not permit_data_path.exists():
        return None
    
    try:
        if permit_data_path.suffix.lower() == '.parquet':
            permit_data = pd.read_parquet(permit_data_path)
        else:
            permit_data = pd.read_csv(permit_data_path)
        
        if permit_data.empty:
            return None
        
        return PermitEnricher(permit_data)
        
    except Exception:
        return None
