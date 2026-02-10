"""Address normalization utilities adapted from legacy permit matching logic"""

import re
from typing import Optional, Dict, Set


# Address normalization mappings from legacy code
DIRECTIONAL_MAP = {
    "N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST",
    "NE": "NORTHEAST", "NW": "NORTHWEST", "SE": "SOUTHEAST", "SW": "SOUTHWEST",
    "NORTH": "NORTH", "SOUTH": "SOUTH", "EAST": "EAST", "WEST": "WEST",
    "NORTHEAST": "NORTHEAST", "NORTHWEST": "NORTHWEST", 
    "SOUTHEAST": "SOUTHEAST", "SOUTHWEST": "SOUTHWEST"
}

SUFFIX_MAP = {
    "ST": "STREET", "AVE": "AVENUE", "BLVD": "BOULEVARD", "DR": "DRIVE",
    "RD": "ROAD", "LN": "LANE", "CT": "COURT", "CIR": "CIRCLE",
    "PL": "PLACE", "WAY": "WAY", "PKWY": "PARKWAY", "TRL": "TRAIL",
    "STREET": "STREET", "AVENUE": "AVENUE", "BOULEVARD": "BOULEVARD",
    "DRIVE": "DRIVE", "ROAD": "ROAD", "LANE": "LANE", "COURT": "COURT",
    "CIRCLE": "CIRCLE", "PLACE": "PLACE", "PARKWAY": "PARKWAY", "TRAIL": "TRAIL"
}

UNIT_TOKENS = {
    "APT", "APARTMENT", "UNIT", "STE", "SUITE", "BLDG", "BUILDING", 
    "#", "LOT", "SPACE", "ROOM", "RM", "FLOOR", "FL"
}


def normalize_address_street(street: str) -> str:
    """Normalize street address using legacy logic"""
    if not isinstance(street, str):
        return ""
    
    base = street.strip().upper()
    base = base.replace(".", " ")
    base = re.sub(r"[^A-Z0-9# ]+", " ", base)
    tokens = [t for t in base.split() if t]
    
    # Remove unit information
    cut_idx = None
    for i, tok in enumerate(tokens):
        if tok in UNIT_TOKENS:
            cut_idx = i
            break
    if cut_idx is not None:
        tokens = tokens[:cut_idx]
    
    # Apply mappings
    tokens = [DIRECTIONAL_MAP.get(t, t) for t in tokens]
    tokens = [SUFFIX_MAP.get(t, t) for t in tokens]
    
    normalized = " ".join(tokens)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_street_from_full_address(full_address: str) -> str:
    """Extract street portion from full address"""
    if not isinstance(full_address, str):
        return ""
    
    # Simple extraction - take everything before city/state/zip pattern
    # This is a simplified version - the legacy code is more complex
    parts = full_address.strip().split(",")
    if parts:
        return parts[0].strip()
    return full_address.strip()


def extract_house_number_token(normalized_street: str) -> str:
    """Extract house number from normalized street"""
    if not normalized_street:
        return ""
    
    tokens = normalized_street.split()
    if tokens and re.match(r"^\d+$", tokens[0]):
        return tokens[0]
    return ""


def get_street_core_without_number(normalized_street: str) -> str:
    """Get street core without house number"""
    if not normalized_street:
        return ""
    
    tokens = normalized_street.split()
    if tokens and re.match(r"^\d+$", tokens[0]):
        return " ".join(tokens[1:])
    return normalized_street


def digits_only(text: str) -> str:
    """Extract only digits from text"""
    return re.sub(r"[^0-9]", "", text)


def nondigits_only(text: str) -> str:
    """Extract only non-digits from text"""
    return re.sub(r"[0-9]", "", text).strip()


def first_token(text: str) -> str:
    """Get first token from text"""
    tokens = text.split()
    return tokens[0] if tokens else ""


def is_mccullough(name: str) -> bool:
    """Check if company name is McCullough variant"""
    if not isinstance(name, str):
        return False
    
    s = name.strip().lower()
    return s in {
        'mccullough heating & air',
        'mccullough heating and air',
        'mccullough heating & air conditioning',
        'mccullough heating and air conditioning',
        'mccullough hvac',
        'mccullough'
    }
