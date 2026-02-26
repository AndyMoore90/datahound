"""Enhanced fuzzy address matching logic from legacy code"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set, Callable
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import rapidfuzz.fuzz as rf_fuzz
    import rapidfuzz.distance.Levenshtein as rf_lev
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

_RAPIDFUZZ_WARNED = False


def _warn_rapidfuzz_once() -> None:
    global _RAPIDFUZZ_WARNED
    if _RAPIDFUZZ_WARNED:
        return
    _RAPIDFUZZ_WARNED = True
    logger.warning(
        "rapidfuzz is not installed; fuzzy permit matching is disabled and simple matching will be used. "
        "Optional install: pip install rapidfuzz"
    )

from .address_utils import (
    normalize_address_street, 
    extract_house_number_token, 
    get_street_core_without_number,
    digits_only,
    nondigits_only,
    first_token
)


def build_enhanced_permit_index(permit_data: pd.DataFrame, max_edits: int = 2) -> Tuple[Dict, Dict, Dict, Dict]:
    """Build comprehensive permit index for fast fuzzy matching"""
    
    if not FUZZY_AVAILABLE:
        _warn_rapidfuzz_once()
        return {}, {}, {}, {}
    
    col_map = {col.lower(): col for col in permit_data.columns}
    
    # Find address column
    address_col = None
    for possible_col in ['original address 1', 'permit_location', 'original_address1', 'address', 'location']:
        if possible_col in col_map:
            address_col = col_map[possible_col]
            break
    
    if not address_col:
        return {}, {}, {}, {}
    
    # Initialize indices
    exact_map: Dict[Tuple[str, str], str] = {}  # (house, core) -> raw_address
    exact_count_map: Dict[Tuple[str, str], int] = {}
    
    # For fuzzy matching
    number_name_index: Dict[str, Dict[str, List[Tuple[str, str, str, str, Dict]]]] = defaultdict(lambda: defaultdict(list))
    number_initial_index: Dict[str, Dict[str, List[Tuple[str, str, str, str, Dict]]]] = defaultdict(lambda: defaultdict(list))
    number_all_index: Dict[str, List[Tuple[str, str, str, str, Dict]]] = defaultdict(list)
    
    # Process each permit
    for _, row in permit_data.iterrows():
        address = str(row[address_col]) if pd.notna(row[address_col]) else ""
        if not address:
            continue
        
        # Normalize address
        normalized = normalize_address_street(address)
        if not normalized:
            continue
        
        house = extract_house_number_token(normalized)
        core = get_street_core_without_number(normalized)
        
        if not house or not core:
            continue
        
        # Build exact index
        exact_key = (house, core)
        exact_map[exact_key] = address
        exact_count_map[exact_key] = exact_count_map.get(exact_key, 0) + 1
        
        # Build fuzzy indices
        core_digits = digits_only(core)
        core_nondigits = nondigits_only(core)
        first_tok = first_token(core_nondigits)
        initial = core_nondigits[0] if core_nondigits else ""
        
        # Create permit summary using dataset-friendly columns
        def _get_col(*names):
            for n in names:
                c = col_map.get(n)
                if c:
                    return c
            return None
        permit_num_col = _get_col('permit num', 'permit_id')
        issued_col = _get_col('issued date', 'issue_date')
        desc_col = _get_col('description')
        contractor_colname = _get_col('contractor company name', 'contractor')
        permit_summary = {
            'permit_id': str(row.get(permit_num_col, '')) if permit_num_col else '',
            'description': str(row.get(desc_col, '')) if desc_col else '',
            'contractor': str(row.get(contractor_colname, '')) if contractor_colname else '',
            'issue_date': str(row.get(issued_col, '')) if issued_col else ''
        }
        
        tuple_data = (core_nondigits, core_digits, core, address, permit_summary)
        
        # Add to fuzzy indices
        number_name_index[house][first_tok].append(tuple_data)
        number_initial_index[house][initial].append(tuple_data)
        number_all_index[house].append(tuple_data)
    
    return exact_map, exact_count_map, number_name_index, number_initial_index, number_all_index


def enhanced_address_match(location_address: str, exact_map: Dict, exact_count_map: Dict, 
                          number_name_index: Dict, number_initial_index: Dict, 
                          number_all_index: Dict, max_edits: int = 2,
                          debug: bool = False,
                          debug_cb: Optional[Callable[[str, Dict], None]] = None) -> Dict:
    """Perform enhanced fuzzy address matching using legacy logic"""
    
    if not FUZZY_AVAILABLE:
        _warn_rapidfuzz_once()
        return {"match_type": "NO_MATCH", "score": 0.0, "matched_permits": []}
    
    # Normalize location address (extract street first)
    from .address_utils import extract_street_from_full_address
    street = extract_street_from_full_address(location_address)
    normalized = normalize_address_street(street).lower()
    if not normalized:
        return {"match_type": "NO_MATCH", "score": 0.0, "matched_permits": []}
    
    loc_number = extract_house_number_token(normalized)
    loc_core = get_street_core_without_number(normalized)
    
    if not loc_number or not loc_core:
        return {"match_type": "NO_MATCH", "score": 0.0, "matched_permits": []}
    
    loc_core_digits = digits_only(loc_core)
    loc_core_nondigits = nondigits_only(loc_core)
    loc_first_tok = first_token(loc_core_nondigits)
    
    if debug and debug_cb:
        debug_cb("loc_norm", {
            "raw_address": location_address,
            "street": street,
            "normalized": normalized,
            "house": loc_number,
            "core": loc_core
        })

    # Try exact match first
    exact_key = (loc_number, loc_core)
    if exact_key in exact_map:
        if debug and debug_cb:
            debug_cb("exact_match", {"exact_key": exact_key, "address": exact_map[exact_key]})
        return {
            "match_type": "EXACT",
            "score": 1.0,
            "distance": 0,
            "matched_permits": [{"address": exact_map[exact_key]}],
            "match_count": exact_count_map.get(exact_key, 1),
            "best_address": exact_map[exact_key],
            "best_norm": loc_core.lower()
        }
    
    # Fuzzy matching
    candidate_lists: List[Tuple[str, str, str, str, Dict]] = []
    
    # Strategy 1: Match by house number + first token
    init = loc_core[0] if loc_core else ""
    c1 = number_name_index.get(loc_number, {}).get(loc_first_tok, [])
    candidate_lists.extend(c1)
    if debug and debug_cb:
        debug_cb("strategy_candidates", {"strategy": "house+first_token", "count": len(c1)})
    
    # Strategy 2: Match by house number + initial
    if not candidate_lists:
        c2 = number_initial_index.get(loc_number, {}).get(init, [])
        candidate_lists.extend(c2)
        if debug and debug_cb:
            debug_cb("strategy_candidates", {"strategy": "house+initial", "count": len(c2)})
    
    # Strategy 3: Match by house number only
    if not candidate_lists:
        c3 = number_all_index.get(loc_number, [])
        candidate_lists.extend(c3)
        if debug and debug_cb:
            debug_cb("strategy_candidates", {"strategy": "house_only", "count": len(c3)})
    
    # Find best fuzzy match
    best_dist: Optional[int] = None
    best_raw = ""
    best_norm = ""
    count_pass = 0
    passed_summaries: List[Dict] = []
    
    for cand_core_nondigits, cand_core_digits, cand_core, cand_raw, cand_summary in candidate_lists:
        # Digits must match exactly
        if loc_core_digits != cand_core_digits:
            if debug and debug_cb:
                debug_cb("candidate_skip_digits", {
                    "loc_core_digits": loc_core_digits,
                    "cand_core_digits": cand_core_digits,
                    "cand_raw": cand_raw,
                    "cand_core": cand_core
                })
            continue
        
        # Calculate edit distance for non-digit parts
        d = rf_lev.distance(loc_core_nondigits.lower(), cand_core_nondigits.lower(), score_cutoff=max_edits)
        if d is None or d > max_edits:
            if debug and debug_cb:
                debug_cb("candidate_skip_distance", {
                    "loc_core_nondigits": loc_core_nondigits,
                    "cand_core_nondigits": cand_core_nondigits,
                    "distance": d if d is not None else -1,
                    "max_edits": max_edits,
                    "cand_raw": cand_raw
                })
            continue
        
        # First token distance must be <= 1
        loc_first = first_token(loc_core_nondigits)
        cand_first = first_token(cand_core_nondigits)
        name_d = rf_lev.distance(loc_first.lower(), cand_first.lower(), score_cutoff=1)
        if name_d is None or name_d > 1:
            if debug and debug_cb:
                debug_cb("candidate_skip_first_token", {
                    "loc_first": loc_first,
                    "cand_first": cand_first,
                    "distance": name_d if name_d is not None else -1,
                    "cand_raw": cand_raw
                })
            continue
        
        count_pass += 1
        if debug and debug_cb:
            debug_cb("candidate_pass", {
                "cand_raw": cand_raw,
                "cand_core": cand_core,
                "distance": d,
                "first_token_distance": name_d
            })
        if cand_summary.get("permit_id"):
            passed_summaries.append(cand_summary)
        
        if best_dist is None or d < best_dist:
            best_dist = d
            best_raw = cand_raw
            best_norm = cand_core
            # Perfect match - stop searching
            if d == 0 and name_d == 0:
                break
    
    # Return result
    if best_dist is not None and best_dist <= max_edits:
        max_len = max(len(loc_core_nondigits), len(nondigits_only((best_norm or "").lower()))) or 1
        score = 1.0 - (best_dist / max_len)
        if debug and debug_cb:
            debug_cb("match_result", {
                "match_type": "FUZZY",
                "score": score,
                "distance": best_dist,
                "best_raw": best_raw,
                "best_norm": best_norm,
                "match_count": count_pass
            })
        
        return {
            "match_type": "FUZZY",
            "score": score,
            "distance": best_dist,
            "matched_permits": passed_summaries,
            "match_count": count_pass,
            "best_address": best_raw,
            "best_norm": best_norm
        }
    
    if debug and debug_cb:
        debug_cb("match_result", {
            "match_type": "NO_MATCH",
        })
    return {"match_type": "NO_MATCH", "score": 0.0, "matched_permits": []}


def enhanced_location_permit_matching(locations_df: pd.DataFrame, permit_data: pd.DataFrame, 
                                    max_edits: int = 2, processing_limit: Optional[int] = None) -> List[Dict]:
    """Enhanced location-to-permit matching using full legacy logic"""
    
    if not FUZZY_AVAILABLE:
        _warn_rapidfuzz_once()
        return []
    
    # Build comprehensive permit index
    exact_map, exact_count_map, number_name_index, number_initial_index, number_all_index = build_enhanced_permit_index(permit_data, max_edits)
    
    if not exact_map and not number_all_index:
        return []
    
    # Get location addresses
    location_addresses = prepare_location_addresses_enhanced(locations_df)
    
    # Apply processing limit
    if processing_limit is not None:
        location_items = list(location_addresses.items())[:processing_limit]
    else:
        location_items = list(location_addresses.items())
    
    results = []
    
    # Process each location
    for location_id, address_data in location_items:
        raw_address = address_data.get("full_address", "")
        
        match_result = enhanced_address_match(
            raw_address, exact_map, exact_count_map,
            number_name_index, number_initial_index, number_all_index, max_edits
        )
        
        if match_result.get("match_type") != "NO_MATCH":
            results.append({
                "location_id": location_id,
                "raw_address": raw_address,
                "match_type": match_result.get("match_type"),
                "score": match_result.get("score", 0.0),
                "distance": match_result.get("distance", 0),
                "permit_count": len(match_result.get("matched_permits", [])),
                "permits": match_result.get("matched_permits", [])
            })
    
    return results


def prepare_location_addresses_enhanced(locations_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Enhanced location address preparation"""
    from .address_utils import extract_street_from_full_address, normalize_address_street
    
    col_map = {col.lower(): col for col in locations_df.columns}
    
    # Find location ID and address columns
    location_id_col = None
    for possible_col in ['location id', 'locationid', 'id']:
        if possible_col in col_map:
            location_id_col = col_map[possible_col]
            break
    
    address_col = None
    for possible_col in ['customer address', 'full address', 'address', 'street address']:
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
            house = extract_house_number_token(normalized)
            core = get_street_core_without_number(normalized)
            
            location_addresses[location_id] = {
                "full_address": full_address,
                "street": street,
                "normalized": normalized,
                "house": house,
                "core": core,
                "core_digits": digits_only(core),
                "core_nondigits": nondigits_only(core),
                "first_token": first_token(nondigits_only(core))
            }
    
    return location_addresses
