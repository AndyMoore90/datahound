"""Additional scan methods for the EventScanner class"""

import asyncio
import concurrent.futures
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional
import pandas as pd

from .types import EventResult, EventRule, EventScanConfig, EventSeverity
from .progress import ProgressTracker, create_streamlit_progress


def determine_aging_severity(age: float) -> EventSeverity:
    """Determine severity based on system age"""
    if age >= 20:
        return EventSeverity.CRITICAL
    elif age >= 15:
        return EventSeverity.HIGH
    elif age >= 10:
        return EventSeverity.MEDIUM
    else:
        return EventSeverity.LOW


def scan_aging_systems(scanner, rule: EventRule, config: EventScanConfig) -> List[EventResult]:
    """Implement aging systems detection using LLM analysis of job histories"""
    
    # Load required tables
    jobs_df = scanner.load_master_table("jobs")
    locations_df = scanner.load_master_table("locations")
    
    if jobs_df is None or locations_df is None:
        return []
    
    events = []
    
    # Get location IDs to analyze
    all_location_ids = get_location_ids_from_table(locations_df)
    
    # Apply processing limit if specified
    if config.processing_limit is not None:
        location_ids = all_location_ids[:config.processing_limit]
        limited = len(all_location_ids) > config.processing_limit
    else:
        location_ids = all_location_ids
        limited = False
    
    # Build job histories per location (simplified version of legacy logic)
    location_jobs = build_location_job_histories(jobs_df, location_ids)
    
    # Filter to only locations with jobs
    locations_with_jobs = {loc_id: jobs for loc_id, jobs in location_jobs.items() if jobs}
    
    # Setup progress tracking
    total_to_process = len(locations_with_jobs)
    progress_tracker = None
    
    if config.show_progress and total_to_process > 0:
        progress_tracker = create_streamlit_progress(
            total_to_process, 
            f"Analyzing {total_to_process} locations with LLM" + (" (limited)" if limited else "")
        )
    
    # Get concurrent processing configuration
    aging_config = scanner.get_aging_systems_config() if hasattr(scanner, 'get_aging_systems_config') else None
    concurrent_calls = aging_config.concurrent_llm_calls if aging_config else 20  # Default to 20
    
    # Process locations with concurrent LLM calls
    processed_count = 0
    locations_items = list(locations_with_jobs.items())
    
    # Process in batches for concurrent LLM calls
    batch_number = 0
    total_batches = (len(locations_items) + concurrent_calls - 1) // concurrent_calls
    
    for i in range(0, len(locations_items), concurrent_calls):
        batch = locations_items[i:i + concurrent_calls]
        batch_number += 1
        
        # Update batch progress
        if progress_tracker:
            progress_tracker.set_operation(f"Processing batch {batch_number}/{total_batches} ({len(batch)} locations)")
        
        # Process batch concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_calls) as executor:
            # Submit all LLM analysis tasks for this batch
            future_to_location = {}
            for location_id, jobs in batch:
                future = executor.submit(scanner.system_age_analyzer.analyze_location_jobs, location_id, jobs)
                future_to_location[future] = (location_id, jobs)
            
            # Collect results as they complete
            completed_in_batch = 0
            for future in concurrent.futures.as_completed(future_to_location):
                location_id, jobs = future_to_location[future]
                completed_in_batch += 1
                
                # Update progress with detailed information
                if progress_tracker:
                    progress_tracker.update(1, f"Batch {batch_number}/{total_batches}: Analyzed location {location_id} ({completed_in_batch}/{len(batch)})")
                
                try:
                    analysis_result = future.result()
                    processed_count += 1
                    
                    if analysis_result and analysis_result.get('age', 0) > 0:
                        # Get enriched location data
                        location_data = scanner._get_location_details(locations_df, location_id) if locations_df is not None else {}
                        
                        # Determine if this qualifies as an "aging system" event
                        age = analysis_result.get('age', 0)
                        min_age_threshold = rule.detection_logic.get('min_age_years', 15)  # Default 15+ years is "aging"
                        
                        if age >= min_age_threshold:
                            event = EventResult(
                                event_type="aging_systems",
                                entity_type="location",
                                entity_id=location_id,
                                severity=determine_aging_severity(age),
                                detected_at=datetime.now(UTC),
                                details={
                                    "system_age": age,
                                    "description": analysis_result.get('description', ''),
                                    "job_date": analysis_result.get('job_date', ''),
                                    "text_snippet": analysis_result.get('text_snippet', ''),
                                    "reasoning": analysis_result.get('reasoning', ''),
                                    "confidence": analysis_result.get('confidence', 0.0),
                                    "source_type": analysis_result.get('source_type', 'unknown'),
                                    "replacement_detected": analysis_result.get('replacement_detected', False),
                                    "min_age_threshold": min_age_threshold,
                                    "analysis_method": "job_history_llm",
                                    "llm_model_used": "deepseek",
                                    **location_data
                                },
                                rule_name=rule.name
                            )
                            events.append(event)
                
                except Exception as e:
                    # Log error but continue processing other locations
                    if hasattr(scanner, 'logger'):
                        scanner.logger.log_scan_error(f"aging_systems_location_{location_id}", str(e), {
                            "location_id": location_id,
                            "batch_processing": True
                        })
                    processed_count += 1
    
    # Final progress update
    if progress_tracker:
        progress_tracker.set_operation(f"Completed: {len(events)} aging systems found")
    
    # Store statistics for the scanner to use
    scanner._last_aging_stats = {
        'total_examined': len(all_location_ids),
        'processed': processed_count,
        'limit_applied': limited
    }
    
    return events


def scan_canceled_jobs(scanner, rule: EventRule, config: EventScanConfig) -> List[EventResult]:
    """Implement canceled jobs detection logic"""
    
    # Load jobs table
    jobs_df = scanner.load_master_table("jobs")
    if jobs_df is None:
        return []
    
    events = []
    col_map = {col.lower(): col for col in jobs_df.columns}
    
    # Find status column
    status_col = None
    for possible_col in ['status', 'job status', 'current status']:
        if possible_col in col_map:
            status_col = col_map[possible_col]
            break
    
    if not status_col:
        return []
    
    # Find ID column
    id_col = None
    for possible_col in ['job id', 'jobid', 'id']:
        if possible_col in col_map:
            id_col = col_map[possible_col]
            break
    
    if not id_col:
        return []
    
    # Get configuration
    canceled_values = rule.detection_logic.get('canceled_values', ['Canceled', 'Cancelled'])
    
    # Filter to canceled jobs
    canceled_mask = jobs_df[status_col].astype(str).str.lower().isin([v.lower() for v in canceled_values])
    all_canceled_jobs = jobs_df[canceled_mask].copy()
    
    # Apply processing limit
    if config.processing_limit is not None:
        canceled_jobs = all_canceled_jobs.head(config.processing_limit)
        limited = len(all_canceled_jobs) > config.processing_limit
    else:
        canceled_jobs = all_canceled_jobs
        limited = False
    
    # Setup progress tracking
    if config.show_progress and len(canceled_jobs) > 0:
        progress_tracker = create_streamlit_progress(
            len(canceled_jobs),
            f"Processing {len(canceled_jobs)} canceled jobs"
        )
    else:
        progress_tracker = None
    
    # Process each canceled job
    for idx, (_, row) in enumerate(canceled_jobs.iterrows()):
        if progress_tracker:
            progress_tracker.update(1, f"Processing job {row[id_col]}")
        
        job_id = str(row[id_col])
        
        # Get additional job details
        job_details = {}
        detail_columns = ['customer id', 'location id', 'summary', 'created date', 'completion date']
        for detail_col in detail_columns:
            if detail_col in col_map and pd.notna(row[col_map[detail_col]]):
                job_details[detail_col.replace(' ', '_')] = str(row[col_map[detail_col]])
        
        event = EventResult(
            event_type="canceled_jobs",
            entity_type="job",
            entity_id=job_id,
            severity=EventSeverity.MEDIUM,
            detected_at=datetime.now(UTC),
            details={
                "status": str(row[status_col]),
                **job_details
            },
            rule_name=rule.name
        )
        events.append(event)
    
    # Store statistics
    scanner._last_canceled_stats = {
        'total_examined': len(all_canceled_jobs),
        'processed': len(canceled_jobs),
        'limit_applied': limited
    }
    
    return events


def scan_unsold_estimates(scanner, rule: EventRule, config: EventScanConfig) -> List[EventResult]:
    """Implement unsold estimates detection logic"""
    
    # Load estimates table
    estimates_df = scanner.load_master_table("estimates")
    if estimates_df is None:
        return []
    
    events = []
    col_map = {col.lower(): col for col in estimates_df.columns}
    
    # Find required columns
    status_col = None
    for possible_col in ['estimate status', 'status', 'current status']:
        if possible_col in col_map:
            status_col = col_map[possible_col]
            break
    
    summary_col = None
    for possible_col in ['estimate summary', 'summary', 'description']:
        if possible_col in col_map:
            summary_col = col_map[possible_col]
            break
    
    id_col = None
    for possible_col in ['estimate id', 'estimateid', 'id']:
        if possible_col in col_map:
            id_col = col_map[possible_col]
            break
    
    if not status_col or not id_col:
        return []
    
    # Get configuration
    include_statuses = rule.detection_logic.get('include_statuses', ['Dismissed', 'Open'])
    exclude_substrings = rule.detection_logic.get('exclude_substrings', ['This is an empty'])
    
    # Filter to target statuses
    status_mask = estimates_df[status_col].astype(str).isin(include_statuses)
    all_target_estimates = estimates_df[status_mask].copy()
    
    # Filter out excluded summaries
    if summary_col:
        for exclude_text in exclude_substrings:
            exclude_mask = all_target_estimates[summary_col].astype(str).str.contains(exclude_text, na=False)
            all_target_estimates = all_target_estimates[~exclude_mask]
    
    # Apply processing limit
    if config.processing_limit is not None:
        target_estimates = all_target_estimates.head(config.processing_limit)
        limited = len(all_target_estimates) > config.processing_limit
    else:
        target_estimates = all_target_estimates
        limited = False
    
    # Setup progress tracking
    if config.show_progress and len(target_estimates) > 0:
        progress_tracker = create_streamlit_progress(
            len(target_estimates),
            f"Processing {len(target_estimates)} unsold estimates"
        )
    else:
        progress_tracker = None
    
    # Process each unsold estimate
    for idx, (_, row) in enumerate(target_estimates.iterrows()):
        if progress_tracker:
            progress_tracker.update(1, f"Processing estimate {row[id_col]}")
        
        estimate_id = str(row[id_col])
        
        # Get additional estimate details
        estimate_details = {}
        detail_columns = ['customer id', 'location id', 'summary', 'created date', 'total']
        for detail_col in detail_columns:
            if detail_col in col_map and pd.notna(row[col_map[detail_col]]):
                estimate_details[detail_col.replace(' ', '_')] = str(row[col_map[detail_col]])
        
        event = EventResult(
            event_type="unsold_estimates",
            entity_type="estimate",
            entity_id=estimate_id,
            severity=EventSeverity.MEDIUM,
            detected_at=datetime.now(UTC),
            details={
                "status": str(row[status_col]),
                "summary": str(row[summary_col]) if summary_col else "",
                **estimate_details
            },
            rule_name=rule.name
        )
        events.append(event)
    
    # Store statistics
    scanner._last_estimates_stats = {
        'total_examined': len(all_target_estimates),
        'processed': len(target_estimates),
        'limit_applied': limited
    }
    
    return events


def scan_permit_matching(scanner, rule: EventRule, config: EventScanConfig) -> List[EventResult]:
    """Implement permit address matching logic from legacy code"""
    
    # Load required data
    locations_df = scanner.load_master_table("locations")
    permit_data = scanner.load_permit_data()
    
    if locations_df is None or permit_data is None:
        return []
    
    events = []
    
    # Get location addresses and normalize them
    location_addresses = prepare_location_addresses(locations_df)
    
    # Build permit address index
    permit_index = build_permit_address_index(permit_data)
    
    # Apply processing limit
    if config.processing_limit is not None:
        location_items = list(location_addresses.items())[:config.processing_limit]
        limited = len(location_addresses) > config.processing_limit
    else:
        location_items = list(location_addresses.items())
        limited = False
    
    # Setup progress tracking
    if config.show_progress and location_items:
        progress_tracker = create_streamlit_progress(
            len(location_items),
            f"Matching {len(location_items)} locations to permits"
        )
    else:
        progress_tracker = None
    
    # Process each location
    for location_id, address_data in location_items:
        if progress_tracker:
            progress_tracker.update(1, f"Matching location {location_id}")
        
        # Perform address matching using legacy logic
        match_result = match_location_to_permits(location_id, address_data, permit_index)
        
        if match_result and match_result.get("match_type") != "NO_MATCH":
            # Create event for successful matches
            event = EventResult(
                event_type="permit_matches",
                entity_type="location",
                entity_id=location_id,
                severity=EventSeverity.LOW if match_result.get("match_type") == "EXACT" else EventSeverity.MEDIUM,
                detected_at=datetime.now(UTC),
                details={
                    "match_type": match_result.get("match_type", ""),
                    "score": match_result.get("score", 0.0),
                    "distance": match_result.get("distance", 0),
                    "permit_count": len(match_result.get("matched_permits", [])),
                    "permits": match_result.get("matched_permits", [])[:5]  # First 5 permits only
                },
                rule_name=rule.name
            )
            events.append(event)
    
    # Store statistics
    scanner._last_permit_match_stats = {
        'total_examined': len(location_addresses),
        'processed': len(location_items),
        'limit_applied': limited
    }
    
    return events


def scan_lost_customers(scanner, rule: EventRule, config: EventScanConfig) -> List[EventResult]:
    """Enhanced lost customers detection with call history and contractor timeline analysis"""
    
    # Load required data
    customers_df = scanner.load_master_table("customers")
    permit_data = scanner.load_permit_data()
    calls_df = scanner.load_master_table("calls")
    
    if customers_df is None or permit_data is None:
        return []
    
    events = []
    
    # Build customer address index
    customer_index = build_customer_address_index(customers_df)
    
    # Build call history index for first call dates
    call_history_index = build_call_history_index(calls_df) if calls_df is not None else {}
    
    # Filter permits to all contractors (we'll analyze McCullough vs others later)
    from .address_utils import is_mccullough
    
    # Find contractor and date columns
    col_map = {col.lower(): col for col in permit_data.columns}
    contractor_col = None
    for possible_col in ['contractor company name', 'contractor_company_name', 'contractor']:
        if possible_col in col_map:
            contractor_col = col_map[possible_col]
            break
    
    date_col = None
    for possible_col in ['issue date', 'issued date', 'permit date', 'date issued']:
        if possible_col in col_map:
            date_col = col_map[possible_col]
            break
    
    if not contractor_col:
        return []
    
    # Apply processing limit to all permits first
    permits_to_process = permit_data.copy()
    if config.processing_limit is not None:
        permits_to_process = permits_to_process.head(config.processing_limit)
        limited = len(permit_data) > config.processing_limit
    else:
        limited = False
    
    # Analyze customer permit histories
    customer_permit_histories = {}
    total_matches = 0
    
    # Note: Progress tracking is handled by the main event execution in apps/pages/4_Event_Configs.py
    
    for idx, (_, permit_row) in enumerate(permits_to_process.iterrows()):
        permit_id = permit_row.get('permit_id', f'permit_{idx}')
        contractor = str(permit_row[contractor_col]) if pd.notna(permit_row[contractor_col]) else ""
        permit_date = permit_row[date_col] if date_col and pd.notna(permit_row[date_col]) else None
        
        # Match permit address to customers
        matched_customers = match_permit_to_customers(permit_row, customer_index)
        
        if matched_customers:
            total_matches += len(matched_customers)
            
            # Build contractor timeline for each matched customer
            for customer_id in matched_customers:
                if customer_id not in customer_permit_histories:
                    customer_permit_histories[customer_id] = []
                
                customer_permit_histories[customer_id].append({
                    'permit_id': permit_id,
                    'contractor': contractor,
                    'date': permit_date,
                    'is_mccullough': is_mccullough(contractor)
                })
    
    # Analyze each customer's contractor timeline
    lost_customers = []
    recovered_customers = []
    intermittent_customers = []
    
    for customer_id, permit_history in customer_permit_histories.items():
        # Sort permits by date (oldest first)
        if permit_history and permit_history[0].get('date'):
            try:
                permit_history.sort(key=lambda x: pd.to_datetime(x['date'], errors='coerce'))
            except:
                pass  # Keep original order if date parsing fails
        
        # Get first call date for this customer
        first_call_date = call_history_index.get(customer_id, {}).get('first_call_date')
        
        # Analyze contractor sequence
        customer_classification = analyze_customer_contractor_timeline(
            customer_id, permit_history, first_call_date
        )
        
        if customer_classification['status'] == 'lost':
            lost_customers.append(customer_classification)
        elif customer_classification['status'] == 'recovered':
            recovered_customers.append(customer_classification)
        elif customer_classification['status'] == 'intermittent':
            intermittent_customers.append(customer_classification)
    
    # Add debug statistics
    def get_permit_address_for_debug(permit_row):
        address_fields = ['Original Address 1', 'original address 1', 'street_address', 'address', 'full_address']
        for field in address_fields:
            if field in permit_row.index and pd.notna(permit_row.get(field)):
                return str(permit_row[field]).strip()
        return 'N/A'
    
    scanner._debug_lost_customers = {
        'total_permits_checked': len(permits_to_process),
        'total_address_matches': total_matches,
        'unique_customers_analyzed': len(customer_permit_histories),
        'lost_customers_found': len(lost_customers),
        'recovered_customers_found': len(recovered_customers),
        'intermittent_customers_found': len(intermittent_customers),
        'customer_index_size': len(customer_index),
        'call_history_available': len(call_history_index),
        'sample_permit_addresses': [
            get_permit_address_for_debug(permit_row)
            for _, permit_row in permits_to_process.head(5).iterrows()
        ]
    }
    
    # Create events for lost customers only (main focus)
    for customer_analysis in lost_customers:
        customer_id = customer_analysis['customer_id']
        
        event = EventResult(
            event_type="lost_customers",
            entity_type="customer",
            entity_id=customer_id,
            severity=EventSeverity.HIGH,
            detected_at=datetime.now(UTC),
            details={
                "reason": "Used other contractor after McCullough",
                "analysis_period": "historical",
                "total_permits_analyzed": len(permits_to_process),
                "detection_method": "contractor_timeline_analysis",
                "first_call_date": customer_analysis.get('first_call_date'),
                "first_mccullough_date": customer_analysis.get('first_mccullough_date'),
                "last_other_contractor_date": customer_analysis.get('last_other_contractor_date'),
                "last_other_contractor": customer_analysis.get('last_other_contractor'),
                "permit_timeline": customer_analysis.get('permit_timeline', []),
                "total_mccullough_permits": customer_analysis.get('total_mccullough_permits', 0),
                "total_other_permits": customer_analysis.get('total_other_permits', 0),
                "customer_classification": "lost"
            },
            rule_name=rule.name
        )
        events.append(event)
    
    # Store comprehensive statistics
    scanner._last_lost_customers_stats = {
        'total_examined': len(permit_data),
        'processed': len(permits_to_process),
        'limit_applied': limited,
        'customers_analyzed': len(customer_permit_histories),
        'lost_customers': len(lost_customers),
        'recovered_customers': len(recovered_customers),
        'intermittent_customers': len(intermittent_customers)
    }
    
    return events


def scan_market_share(scanner, rule: EventRule, config: EventScanConfig) -> List[EventResult]:
    """Removed during repository cleanup."""
    return []


def scan_system_age_audit(scanner, rule: EventRule, config: EventScanConfig) -> List[EventResult]:
    """Implement system age audit scan from legacy code"""
    
    # Load locations table
    locations_df = scanner.load_master_table("locations")
    if locations_df is None:
        return []
    
    events = []
    col_map = {col.lower(): col for col in locations_df.columns}
    
    # Find system age column
    age_col = None
    for possible_col in ['current_system_age', 'system_age', 'current system age']:
        if possible_col in col_map:
            age_col = col_map[possible_col]
            break
    
    if not age_col:
        return []
    
    # Find location ID column
    id_col = None
    for possible_col in ['location id', 'locationid', 'id']:
        if possible_col in col_map:
            id_col = col_map[possible_col]
            break
    
    if not id_col:
        return []
    
    # Convert age to numeric and filter by minimum age
    locations_df['age_numeric'] = pd.to_numeric(locations_df[age_col], errors='coerce')
    min_age = rule.detection_logic.get('min_age', 15)
    
    # Filter to locations meeting age threshold
    age_mask = locations_df['age_numeric'] >= min_age
    eligible_locations = locations_df[age_mask].copy()
    
    # Apply processing limit
    if config.processing_limit is not None:
        eligible_locations = eligible_locations.head(config.processing_limit)
        limited = len(locations_df[age_mask]) > config.processing_limit
    else:
        limited = False
    
    # Setup progress tracking
    if config.show_progress and len(eligible_locations) > 0:
        progress_tracker = create_streamlit_progress(
            len(eligible_locations),
            f"Auditing {len(eligible_locations)} aging systems"
        )
    else:
        progress_tracker = None
    
    # Process each eligible location
    for idx, (_, row) in enumerate(eligible_locations.iterrows()):
        if progress_tracker:
            progress_tracker.update(1, f"Auditing location {row[id_col]}")
        
        location_id = str(row[id_col])
        age = int(row['age_numeric']) if pd.notna(row['age_numeric']) else 0
        
        # Get additional location details
        location_details = {}
        detail_columns = ['customer id', 'name', 'phone', 'city', 'state', 'zip']
        for detail_col in detail_columns:
            if detail_col in col_map and pd.notna(row[col_map[detail_col]]):
                location_details[detail_col.replace(' ', '_')] = str(row[col_map[detail_col]])
        
        # Determine severity based on age
        if age >= 25:
            severity = EventSeverity.CRITICAL
        elif age >= 20:
            severity = EventSeverity.HIGH
        elif age >= 15:
            severity = EventSeverity.MEDIUM
        else:
            severity = EventSeverity.LOW
        
        event = EventResult(
            event_type="system_age_audit",
            entity_type="location",
            entity_id=location_id,
            severity=severity,
            detected_at=datetime.now(UTC),
            details={
                "system_age": age,
                "min_age_threshold": min_age,
                **location_details
            },
            rule_name=rule.name
        )
        events.append(event)
    
    # Store statistics
    scanner._last_system_age_audit_stats = {
        'total_examined': len(locations_df[age_mask]),
        'processed': len(eligible_locations),
        'limit_applied': limited
    }
    
    return events


def scan_permit_replacements(scanner, rule: EventRule, config: EventScanConfig) -> List[EventResult]:
    """Implement permit replacement detection using LLM analysis"""
    # TODO: This requires matched permit data from permit matching scan
    # Will implement after permit matching is working
    return []


# Helper functions for permit analysis
def prepare_location_addresses(locations_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Prepare location addresses for matching"""
    from .address_utils import normalize_address_street, extract_street_from_full_address
    
    col_map = {col.lower(): col for col in locations_df.columns}
    
    # Find location ID and address columns
    location_id_col = None
    for possible_col in ['location id', 'locationid', 'id']:
        if possible_col in col_map:
            location_id_col = col_map[possible_col]
            break
    
    address_col = None
    for possible_col in ['customer address', 'address', 'street address', 'full address']:
        if possible_col in col_map:
            address_col = col_map[possible_col]
            break
    
    if not location_id_col or not address_col:
        return {}
    
    location_addresses = {}
    
    for _, row in locations_df.iterrows():
        location_id = str(row[location_id_col])
        full_address = str(row[address_col]) if pd.notna(row[address_col]) else ""
        
        if location_id and full_address:
            street = extract_street_from_full_address(full_address)
            normalized = normalize_address_street(street)
            
            location_addresses[location_id] = {
                "full_address": full_address,
                "street": street,
                "normalized": normalized
            }
    
    return location_addresses


def build_permit_address_index(permit_data: pd.DataFrame) -> Dict:
    """Build permit address index for fast matching"""
    # Simplified version - full implementation would match legacy exactly
    index = {}
    
    col_map = {col.lower(): col for col in permit_data.columns}
    
    # Find address columns in permit data
    address_col = None
    for possible_col in ['permit_location', 'address', 'location']:
        if possible_col in col_map:
            address_col = col_map[possible_col]
            break
    
    if not address_col:
        return index
    
    # Build simple address index
    for _, row in permit_data.iterrows():
        address = str(row[address_col]) if pd.notna(row[address_col]) else ""
        if address:
            # Store permit data by normalized address
            # This is a simplified version - legacy has complex fuzzy matching
            index[address.upper().strip()] = row.to_dict()
    
    return index


def build_call_history_index(calls_df: pd.DataFrame) -> Dict:
    """Build call history index to find first call dates for customers"""
    if calls_df is None:
        return {}
    
    col_map = {col.lower(): col for col in calls_df.columns}
    
    # Find customer ID and date columns
    customer_id_col = None
    for possible_col in ['customer id', 'customerid', 'customer_id']:
        if possible_col in col_map:
            customer_id_col = col_map[possible_col]
            break
    
    date_col = None
    for possible_col in ['created date', 'call date', 'date created', 'created_date']:
        if possible_col in col_map:
            date_col = col_map[possible_col]
            break
    
    if not customer_id_col or not date_col:
        return {}
    
    call_index = {}
    
    for _, row in calls_df.iterrows():
        customer_id = str(row[customer_id_col]) if pd.notna(row[customer_id_col]) else None
        call_date = row[date_col] if pd.notna(row[date_col]) else None
        
        if customer_id and call_date:
            try:
                call_date_parsed = pd.to_datetime(call_date, errors='coerce')
                if pd.notna(call_date_parsed):
                    if customer_id not in call_index:
                        call_index[customer_id] = {
                            'first_call_date': call_date_parsed,
                            'total_calls': 0
                        }
                    else:
                        # Update first call date if this one is earlier
                        if call_date_parsed < call_index[customer_id]['first_call_date']:
                            call_index[customer_id]['first_call_date'] = call_date_parsed
                    
                    call_index[customer_id]['total_calls'] += 1
            except:
                continue  # Skip invalid dates
    
    # Convert datetime objects to strings for JSON serialization
    for customer_id in call_index:
        call_index[customer_id]['first_call_date'] = call_index[customer_id]['first_call_date'].isoformat()
    
    return call_index


def analyze_customer_contractor_timeline(customer_id: str, permit_history: List[Dict], first_call_date: Optional[str]) -> Dict:
    """Analyze customer's contractor usage timeline to classify as lost/recovered/intermittent"""
    
    if not permit_history:
        return {'customer_id': customer_id, 'status': 'no_permits'}
    
    # Separate McCullough and other contractor permits
    mcc_permits = [p for p in permit_history if p['is_mccullough']]
    other_permits = [p for p in permit_history if not p['is_mccullough']]
    
    # If customer never used McCullough, they're not a "lost" customer
    if not mcc_permits:
        return {'customer_id': customer_id, 'status': 'never_customer'}
    
    # Get timeline information
    first_mcc_date = mcc_permits[0]['date'] if mcc_permits else None
    last_mcc_date = mcc_permits[-1]['date'] if mcc_permits else None
    last_other_date = other_permits[-1]['date'] if other_permits else None
    last_other_contractor = other_permits[-1]['contractor'] if other_permits else None
    
    # Get the most recent permit overall
    most_recent_permit = permit_history[-1] if permit_history else None
    
    # Classification logic
    result = {
        'customer_id': customer_id,
        'first_call_date': first_call_date,
        'first_mccullough_date': first_mcc_date,
        'last_mccullough_date': last_mcc_date,
        'last_other_contractor_date': last_other_date,
        'last_other_contractor': last_other_contractor,
        'total_mccullough_permits': len(mcc_permits),
        'total_other_permits': len(other_permits),
        'permit_timeline': permit_history
    }
    
    # Check if customer used another contractor after McCullough
    if not other_permits:
        result['status'] = 'loyal_customer'  # Only used McCullough
    elif most_recent_permit and most_recent_permit['is_mccullough']:
        if len(other_permits) > 0:
            result['status'] = 'recovered'  # Used others but came back to McCullough
        else:
            result['status'] = 'loyal_customer'
    elif last_other_date and first_mcc_date:
        try:
            # Parse dates for comparison
            last_other_parsed = pd.to_datetime(last_other_date, errors='coerce')
            first_mcc_parsed = pd.to_datetime(first_mcc_date, errors='coerce')
            
            if pd.notna(last_other_parsed) and pd.notna(first_mcc_parsed):
                if last_other_parsed > first_mcc_parsed:
                    # Check if they used McCullough again after other contractors
                    if last_mcc_date:
                        last_mcc_parsed = pd.to_datetime(last_mcc_date, errors='coerce')
                        if pd.notna(last_mcc_parsed) and last_mcc_parsed > last_other_parsed:
                            result['status'] = 'intermittent'  # Back and forth
                        else:
                            result['status'] = 'lost'  # Used others after McCullough and didn't return
                    else:
                        result['status'] = 'lost'
                else:
                    result['status'] = 'loyal_customer'  # Used others before McCullough
            else:
                result['status'] = 'unclear_timeline'  # Can't parse dates
        except:
            result['status'] = 'unclear_timeline'
    else:
        result['status'] = 'unclear_timeline'
    
    # Additional validation: check first call date
    if first_call_date and result['status'] == 'lost':
        try:
            first_call_parsed = pd.to_datetime(first_call_date, errors='coerce')
            last_other_parsed = pd.to_datetime(last_other_date, errors='coerce')
            
            if pd.notna(first_call_parsed) and pd.notna(last_other_parsed):
                if last_other_parsed < first_call_parsed:
                    # They used another contractor before their first call to McCullough
                    result['status'] = 'never_customer'
        except:
            pass  # Keep original classification if date parsing fails
    
    return result


def build_customer_address_index(customers_df: pd.DataFrame) -> Dict:
    """Build customer address index for lost customer analysis"""
    from .address_utils import normalize_address_street, extract_street_from_full_address
    
    col_map = {col.lower(): col for col in customers_df.columns}
    
    # Find customer ID and address columns
    customer_id_col = None
    for possible_col in ['customer id', 'customerid', 'id']:
        if possible_col in col_map:
            customer_id_col = col_map[possible_col]
            break
    
    address_col = None
    for possible_col in ['street address', 'address', 'full address']:
        if possible_col in col_map:
            address_col = col_map[possible_col]
            break
    
    if not customer_id_col or not address_col:
        return {}
    
    index = {}
    
    for _, row in customers_df.iterrows():
        customer_id = str(row[customer_id_col])
        full_address = str(row[address_col]) if pd.notna(row[address_col]) else ""
        
        if customer_id and full_address:
            street = extract_street_from_full_address(full_address)
            normalized = normalize_address_street(street)
            
            # Store customer by normalized address
            index[normalized] = index.get(normalized, set())
            index[normalized].add(customer_id)
    
    return index


def match_location_to_permits(location_id: str, address_data: Dict[str, str], permit_index: Dict) -> Optional[Dict]:
    """Match a location to permits using address"""
    # Simplified matching - legacy has complex fuzzy logic
    normalized = address_data.get("normalized", "")
    
    if not normalized:
        return {"match_type": "NO_MATCH", "score": 0.0, "matched_permits": []}
    
    # Try exact match first
    if normalized.upper() in permit_index:
        return {
            "match_type": "EXACT",
            "score": 1.0,
            "distance": 0,
            "matched_permits": [permit_index[normalized.upper()]]
        }
    
    # TODO: Implement fuzzy matching logic from legacy
    return {"match_type": "NO_MATCH", "score": 0.0, "matched_permits": []}


def match_permit_to_customers(permit_row: pd.Series, customer_index: Dict) -> List[str]:
    """Match a permit to customers by address using address normalization"""
    from .address_utils import normalize_address_street, extract_street_from_full_address
    
    matched_customers = []
    
    if not customer_index:
        return matched_customers
    
    # Find address column in permit data
    permit_address = None
    # Updated address fields based on actual permit data structure
    address_fields = ['Original Address 1', 'original address 1', 'street_address', 'address', 'full_address', 'location_address', 'site_address']
    
    for field in address_fields:
        if field in permit_row.index and pd.notna(permit_row.get(field)):
            permit_address = str(permit_row[field]).strip()
            break
    
    if not permit_address:
        return matched_customers
    
    # Normalize permit address using same logic as customer index
    try:
        street = extract_street_from_full_address(permit_address)
        normalized_permit_address = normalize_address_street(street)
        
        # Look up normalized address in customer index
        if normalized_permit_address in customer_index:
            matched_customers = list(customer_index[normalized_permit_address])
            
    except Exception:
        # If normalization fails, try exact match fallback
        permit_address_upper = permit_address.upper().strip()
        for normalized_addr, customer_ids in customer_index.items():
            if normalized_addr == permit_address_upper:
                matched_customers = list(customer_ids)
                break
    
    return matched_customers


def get_location_ids_from_table(locations_df: pd.DataFrame) -> List[str]:
    """Extract location IDs from locations table"""
    col_map = {col.lower(): col for col in locations_df.columns}
    
    location_id_col = None
    for possible_col in ['location id', 'locationid', 'id']:
        if possible_col in col_map:
            location_id_col = col_map[possible_col]
            break
    
    if not location_id_col:
        return []
    
    return locations_df[location_id_col].astype(str).dropna().unique().tolist()


def build_location_job_histories(jobs_df: pd.DataFrame, location_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Build job histories per location (simplified version of legacy logic)"""
    col_map = {col.lower(): col for col in jobs_df.columns}
    
    # Find relevant columns
    location_col = None
    for possible_col in ['location id', 'locationid', 'location']:
        if possible_col in col_map:
            location_col = col_map[possible_col]
            break
    
    job_id_col = col_map.get('job id') or col_map.get('jobid') or col_map.get('id')
    summary_col = col_map.get('summary') or col_map.get('job summary') or col_map.get('description')
    created_col = col_map.get('created date') or col_map.get('date created') or col_map.get('job date')
    
    if not location_col:
        return {}
    
    location_jobs: Dict[str, List[Dict[str, Any]]] = {}
    
    # Process each job
    for _, row in jobs_df.iterrows():
        location_id = str(row[location_col]) if pd.notna(row[location_col]) else None
        if not location_id or location_id not in location_ids:
            continue
        
        job_data = {
            'job_id': str(row[job_id_col]) if job_id_col and pd.notna(row[job_id_col]) else '',
            'summary': str(row[summary_col]) if summary_col and pd.notna(row[summary_col]) else '',
            'created_date': str(row[created_col]) if created_col and pd.notna(row[created_col]) else ''
        }
        
        # Only include jobs with meaningful summaries
        if job_data['summary'].strip():
            location_jobs.setdefault(location_id, []).append(job_data)
    
    # Sort jobs by date (most recent first) for each location
    for location_id in location_jobs:
        jobs = location_jobs[location_id]
        # Simple sort by created_date string (works for ISO dates)
        try:
            jobs.sort(key=lambda x: x['created_date'], reverse=True)
        except Exception:
            pass  # Keep original order if sorting fails
    
    return location_jobs


def scan_lost_customers(scanner, rule: EventRule, config: EventScanConfig) -> List[EventResult]:
    """Implement lost customers detection using permit data analysis"""
    
    # Log scan start with detailed info
    scanner.logger.log_scan_start(rule.name, "lost_customers_analysis", {
        "event_type": "lost_customers",
        "processing_limit": config.processing_limit,
        "show_progress": config.show_progress
    })
    
    # Load required tables
    customers_df = scanner.load_master_table("customers")
    calls_df = scanner.load_master_table("calls")
    
    if customers_df is None:
        scanner.logger.log_scan_error("lost_customers", "Failed to load customers data", {
            "table": "customers", "reason": "customers_df is None"
        })
        return []
    
    if calls_df is None:
        scanner.logger.log_scan_error("lost_customers", "Failed to load calls data", {
            "table": "calls", "reason": "calls_df is None"
        })
        return []
    
    # Load permit data
    permit_data = scanner.load_permit_data()
    if permit_data is None:
        scanner.logger.log_scan_error("lost_customers", "Failed to load permit data", {
            "table": "permits", "reason": "permit_data is None"
        })
        return []
    
    # Log data loaded successfully
    scanner.logger.log_processing_stats("lost_customers", "data_loaded", {
        "customers_count": len(customers_df),
        "calls_count": len(calls_df),
        "permits_count": len(permit_data),
        "customers_columns": list(customers_df.columns),
        "calls_columns": list(calls_df.columns)
    })
    
    events = []
    
    # Get company name from rule detection logic (configurable)
    company_names = rule.detection_logic.get("exclude_contractors", ["McCullough Heating & Air", "McCullough Heating and Air"])
    
    # Build customer address index for matching
    customer_index = build_customer_address_index(customers_df)
    
    # Get customer IDs to analyze
    col_map = {col.lower(): col for col in customers_df.columns}
    customer_id_col = None
    address_col = None
    
    for possible_col in ['customer id', 'customerid', 'id']:
        if possible_col in col_map:
            customer_id_col = col_map[possible_col]
            break
    
    for possible_col in ['full address', 'address', 'service address', 'street address']:
        if possible_col in col_map:
            address_col = col_map[possible_col]
            break
    
    if not customer_id_col or not address_col:
        scanner.logger.log_scan_error("lost_customers", "Required columns not found", {
            "customer_id_col": customer_id_col,
            "address_col": address_col,
            "available_columns": list(customers_df.columns),
            "reason": "Missing required customer ID or address column"
        })
        return events
    
    scanner.logger.log_processing_stats("lost_customers", "columns_mapped", {
        "customer_id_column": customer_id_col,
        "address_column": address_col,
        "company_names": company_names
    })
    
    all_customer_ids = customers_df[customer_id_col].astype(str).dropna().unique().tolist()
    
    # Apply processing limit if specified
    if config.processing_limit is not None:
        customer_ids = all_customer_ids[:config.processing_limit]
        limited = len(all_customer_ids) > config.processing_limit
    else:
        customer_ids = all_customer_ids
        limited = False
    
    # Setup progress tracking
    total_to_process = len(customer_ids)
    progress_tracker = None
    
    if config.show_progress and total_to_process > 0:
        progress_tracker = create_streamlit_progress(
            total_to_process,
            f"Analyzing {total_to_process} customers for lost customer detection" + (" (limited)" if limited else "")
        )
    
    # Log processing details
    scanner.logger.log_processing_stats("lost_customers", "processing_setup", {
        "total_customers": len(all_customer_ids),
        "customers_to_process": len(customer_ids),
        "processing_limit_applied": limited,
        "processing_limit": config.processing_limit
    })
    
    # Process each customer
    processed_count = 0
    skipped_no_row = 0
    skipped_no_address = 0
    skipped_no_calls = 0
    skipped_no_permits = 0
    analyzed_count = 0
    
    for customer_id in customer_ids:
        if progress_tracker:
            progress_tracker.update(1, f"Analyzing customer {customer_id}")
        
        processed_count += 1
        
        # Get customer details
        customer_row = customers_df[customers_df[customer_id_col].astype(str) == customer_id]
        if customer_row.empty:
            skipped_no_row += 1
            continue
            
        customer_data = customer_row.iloc[0]
        full_address = str(customer_data[address_col]) if pd.notna(customer_data[address_col]) else ""
        
        if not full_address:
            skipped_no_address += 1
            continue
        
        # Calculate contact dates from calls data
        first_contact_date, last_contact_date = _get_customer_contact_dates(calls_df, customer_id)
        
        if not first_contact_date or not last_contact_date:
            skipped_no_calls += 1
            continue
        
        # Find matching permits for this customer's address
        matched_permits = _find_permits_for_address(permit_data, full_address)
        
        if not matched_permits:
            skipped_no_permits += 1
            continue
        
        analyzed_count += 1
        
        # Log first few successful analyses for debugging
        if analyzed_count <= 5:
            scanner.logger.log_processing_stats("lost_customers", f"analyzing_customer_{analyzed_count}", {
                "customer_id": customer_id,
                "address": full_address,
                "first_contact": first_contact_date.isoformat() if first_contact_date else None,
                "last_contact": last_contact_date.isoformat() if last_contact_date else None,
                "permits_found": len(matched_permits)
            })
        
        # Analyze permits to determine lost customer status
        analysis_result = _analyze_customer_permits(
            matched_permits, 
            company_names, 
            first_contact_date, 
            last_contact_date
        )
        
        # Create event if customer is identified as lost
        if analysis_result["lost_customer"]:
            event_details = {
                "customer_id": customer_id,
                "first_contact_date": first_contact_date.isoformat() if first_contact_date else None,
                "last_contact_date": last_contact_date.isoformat() if last_contact_date else None,
                "competitor_used": analysis_result["competitor_used"],
                "shopper_customer": analysis_result["shopper_customer"],
                "lost_customer": True,
                "analysis_period": rule.detection_logic.get("analysis_period", "historical"),
                "permits_analyzed": len(matched_permits),
                "customer_name": str(customer_data.get("Customer Name", "")) if "Customer Name" in customer_data.index else "",
                "phone": str(customer_data.get("Phone", "")) if "Phone" in customer_data.index else "",
                "address": full_address
            }
            
            # Determine severity based on competitor usage and time since last contact
            severity = _determine_lost_customer_severity(last_contact_date, analysis_result["competitor_used"])
            
            event = EventResult(
                event_type="lost_customers",
                entity_type="customer",
                entity_id=customer_id,
                severity=severity,
                detected_at=datetime.now(UTC),
                details=event_details,
                rule_name=rule.name
            )
            events.append(event)
    
    # Log final processing statistics
    scanner.logger.log_processing_stats("lost_customers", "processing_complete", {
        "total_customers": len(all_customer_ids),
        "processed_count": processed_count,
        "skipped_no_row": skipped_no_row,
        "skipped_no_address": skipped_no_address,
        "skipped_no_calls": skipped_no_calls,
        "skipped_no_permits": skipped_no_permits,
        "analyzed_count": analyzed_count,
        "events_found": len(events),
        "processing_limit_applied": limited
    })
    
    # Store statistics for result calculation
    scanner._last_lost_customers_stats = {
        'total_examined': len(all_customer_ids),
        'processed': processed_count,
        'limit_applied': limited
    }
    
    return events


def _get_customer_contact_dates(calls_df: pd.DataFrame, customer_id: str):
    """Get first and last contact dates for a customer from calls data"""
    if calls_df is None or calls_df.empty:
        return None, None
    
    col_map = {col.lower(): col for col in calls_df.columns}
    
    # Find customer ID and call date columns
    customer_col = None
    date_col = None
    
    for possible_col in ['customer id', 'customerid', 'customer']:
        if possible_col in col_map:
            customer_col = col_map[possible_col]
            break
    
    for possible_col in ['call date', 'date', 'created date', 'contact date']:
        if possible_col in col_map:
            date_col = col_map[possible_col]
            break
    
    if not customer_col or not date_col:
        return None, None
    
    # Find all calls for this customer
    customer_calls = calls_df[calls_df[customer_col].astype(str) == customer_id]
    
    if customer_calls.empty:
        return None, None
    
    # Parse dates and find first/last
    call_dates = []
    for _, row in customer_calls.iterrows():
        date_value = row[date_col]
        if pd.notna(date_value):
            try:
                if isinstance(date_value, str):
                    parsed_date = pd.to_datetime(date_value, errors='coerce')
                else:
                    parsed_date = pd.to_datetime(date_value)
                
                if pd.notna(parsed_date):
                    call_dates.append(parsed_date.date())
            except Exception:
                continue
    
    if not call_dates:
        return None, None
    
    call_dates.sort()
    return call_dates[0], call_dates[-1]  # First and last dates


def _find_permits_for_address(permit_data: pd.DataFrame, customer_address: str):
    """Find permits matching customer address using optimized vectorized matching"""
    from .address_utils import normalize_address_street, extract_street_from_full_address
    
    if permit_data is None or permit_data.empty or not customer_address:
        return []
    
    # Normalize customer address
    try:
        customer_street = extract_street_from_full_address(customer_address)
        normalized_customer_address = normalize_address_street(customer_street)
    except Exception:
        return []
    
    if not normalized_customer_address:
        return []
    
    # Find address column in permit data
    col_map = {col.lower(): col for col in permit_data.columns}
    address_col = None
    
    for possible_col in ['location', 'original address 1', 'address', 'permit_location']:
        if possible_col in col_map:
            address_col = col_map[possible_col]
            break
    
    if not address_col:
        return []
    
    try:
        # Use vectorized operations for much better performance
        # First, get all non-null addresses
        permit_addresses = permit_data[address_col].dropna().astype(str)
        
        if permit_addresses.empty:
            return []
        
        # Normalize all permit addresses at once (more efficient)
        normalized_permits = []
        matching_indices = []
        
        for idx, permit_address in permit_addresses.items():
            try:
                permit_street = extract_street_from_full_address(permit_address)
                normalized_permit_address = normalize_address_street(permit_street)
                
                # Check if addresses match
                if normalized_customer_address == normalized_permit_address:
                    matching_indices.append(idx)
                    
            except Exception:
                continue
        
        # Return matching permits as dictionaries
        if matching_indices:
            matched_permits = permit_data.loc[matching_indices].to_dict('records')
            return matched_permits
        
    except Exception:
        # Fallback to slower row-by-row method if vectorized fails
        matched_permits = []
        for _, permit_row in permit_data.iterrows():
            permit_address = str(permit_row[address_col]) if pd.notna(permit_row[address_col]) else ""
            
            if not permit_address:
                continue
            
            try:
                permit_street = extract_street_from_full_address(permit_address)
                normalized_permit_address = normalize_address_street(permit_street)
                
                # Check if addresses match
                if normalized_customer_address == normalized_permit_address:
                    matched_permits.append(permit_row.to_dict())
                    
            except Exception:
                continue
        
        return matched_permits
    
    return []


def _analyze_customer_permits(permits: List[Dict], company_names: List[str], first_contact_date, last_contact_date):
    """Analyze permits to determine lost customer status"""
    result = {
        "lost_customer": False,
        "competitor_used": None,
        "shopper_customer": False
    }
    
    if not permits:
        return result
    
    # Find relevant permit columns
    sample_permit = permits[0]
    contractor_col = None
    applied_date_col = None
    issued_date_col = None
    
    # Look for contractor column
    for col in sample_permit.keys():
        if 'contractor' in col.lower() and 'company' in col.lower():
            contractor_col = col
            break
        elif 'contractor' in col.lower():
            contractor_col = col
            break
    
    # Look for date columns
    for col in sample_permit.keys():
        if 'applied' in col.lower() and 'date' in col.lower():
            applied_date_col = col
            break
    
    for col in sample_permit.keys():
        if 'issued' in col.lower() and 'date' in col.lower():
            issued_date_col = col
            break
    
    if not contractor_col:
        return result
    
    # Track permits by contractor and date
    company_permits = []
    competitor_permits = []
    
    for permit in permits:
        contractor = str(permit.get(contractor_col, "")).strip()
        
        if not contractor:
            continue
        
        # Get permit date (prefer applied date, fallback to issued date)
        permit_date = None
        date_value = permit.get(applied_date_col) if applied_date_col else None
        
        if not date_value and issued_date_col:
            date_value = permit.get(issued_date_col)
        
        if date_value:
            try:
                if isinstance(date_value, str):
                    permit_date = pd.to_datetime(date_value, errors='coerce')
                else:
                    permit_date = pd.to_datetime(date_value)
                
                if pd.notna(permit_date):
                    permit_date = permit_date.date()
            except Exception:
                continue
        
        if not permit_date:
            continue
        
        # Check if this is our company or a competitor
        is_our_company = any(company_name.lower() in contractor.lower() for company_name in company_names)
        
        permit_info = {
            "contractor": contractor,
            "date": permit_date,
            "is_our_company": is_our_company
        }
        
        if is_our_company:
            company_permits.append(permit_info)
        else:
            competitor_permits.append(permit_info)
    
    # Check for lost customer: competitor permit after our last contact
    for comp_permit in competitor_permits:
        if comp_permit["date"] > last_contact_date:
            result["lost_customer"] = True
            result["competitor_used"] = comp_permit["contractor"]
            break
    
    # Check for shopper customer: competitor work after first contact, but our most recent work
    if competitor_permits and company_permits:
        # Find most recent permits from each category
        latest_company = max(company_permits, key=lambda x: x["date"]) if company_permits else None
        latest_competitor = max(competitor_permits, key=lambda x: x["date"]) if competitor_permits else None
        
        # Customer had competitor work after first contact
        competitor_after_first_contact = any(
            comp_permit["date"] > first_contact_date for comp_permit in competitor_permits
        )
        
        # But our most recent permit is newer than competitor's most recent
        if (competitor_after_first_contact and latest_company and latest_competitor and 
            latest_company["date"] > latest_competitor["date"]):
            result["shopper_customer"] = True
    
    return result


def _determine_lost_customer_severity(last_contact_date, competitor_used) -> EventSeverity:
    """Determine severity based on how long since last contact and competitor usage"""
    if not last_contact_date:
        return EventSeverity.MEDIUM
    
    from datetime import date
    today = date.today()
    
    # Calculate months since last contact
    months_since = (today.year - last_contact_date.year) * 12 + (today.month - last_contact_date.month)
    if today.day < last_contact_date.day:
        months_since -= 1
    
    # Determine severity based on time and competitor usage
    if competitor_used:
        if months_since >= 24:  # 2+ years with competitor usage
            return EventSeverity.CRITICAL
        elif months_since >= 12:  # 1+ year with competitor usage
            return EventSeverity.HIGH
        else:
            return EventSeverity.MEDIUM
    else:
        # No specific competitor identified
        if months_since >= 36:  # 3+ years
            return EventSeverity.HIGH
        else:
            return EventSeverity.MEDIUM


def determine_aging_severity(age: int) -> EventSeverity:
    """Determine severity based on system age"""
    if age >= 25:  # 25+ years
        return EventSeverity.CRITICAL
    elif age >= 20:  # 20-24 years
        return EventSeverity.HIGH
    elif age >= 15:  # 15-19 years
        return EventSeverity.MEDIUM
    else:  # Under 15 years
        return EventSeverity.LOW
