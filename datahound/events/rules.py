from __future__ import annotations

from typing import Dict, List
from .types import EventRule, RuleType, EventSeverity


def get_default_overdue_maintenance_rule() -> EventRule:
    """Create the default overdue maintenance detection rule"""
    return EventRule(
        name="Overdue Maintenance Detection",
        description="Detects locations and customers with overdue maintenance based on job history",
        event_type="overdue_maintenance", 
        rule_type=RuleType.CROSS_TABLE,
        target_tables=["jobs", "locations", "customers"],
        detection_logic={
            "maintenance_criteria": {
                "job_class_values": ["maintenance"],
                "job_type_values": ["maintenance"],
                "exclude_summaries": ["test", "testing"]
            },
            "date_logic": {
                "preferred_date_column": "completion_date",
                "fallback_date_column": "created_date"
            },
            "threshold": {
                "default_months": 12,
                "severity_multipliers": {
                    "medium": 1.25,  # 15 months
                    "high": 1.5,     # 18 months  
                    "critical": 2.0   # 24 months
                }
            }
        },
        output_fields=[
            "entity_type",
            "entity_id", 
            "months_overdue",
            "last_maintenance_date",
            "name",
            "phone",
            "city",
            "state",
            "zip"
        ],
        severity=EventSeverity.MEDIUM,
        enabled=True,
        threshold_months=12
    )


def get_default_aging_systems_rule() -> EventRule:
    """Create the default aging systems detection rule"""
    return EventRule(
        name="Aging Systems Detection",
        description="Detects locations with aging HVAC systems based on LLM analysis of job histories",
        event_type="aging_systems",
        rule_type=RuleType.LLM_ANALYSIS,
        target_tables=["jobs", "locations"],
        detection_logic={
            "min_age_years": 15,
            "llm_analysis": {
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com",
                "max_tokens": 8192,
                "temperature": 0.0,
                "max_retries": 2
            },
            "job_history": {
                "max_jobs_to_analyze": 10,
                "prioritize_recent": True
            }
        },
        output_fields=[
            "entity_type",
            "entity_id",
            "system_age", 
            "description",
            "job_date",
            "text_snippet",
            "reasoning",
            "name",
            "phone",
            "city",
            "state"
        ],
        severity=EventSeverity.MEDIUM,
        enabled=True
    )


def get_default_canceled_jobs_rule() -> EventRule:
    """Create the default canceled jobs detection rule"""
    return EventRule(
        name="Canceled Jobs Detection",
        description="Detects jobs that have been canceled",
        event_type="canceled_jobs",
        rule_type=RuleType.SINGLE_TABLE,
        target_tables=["jobs"],
        detection_logic={
            "canceled_values": ["Canceled", "Cancelled"],
            "exclude_from_values": []
        },
        output_fields=[
            "entity_type",
            "entity_id",
            "status",
            "customer_id",
            "location_id",
            "summary",
            "created_date"
        ],
        severity=EventSeverity.MEDIUM,
        enabled=True
    )


def get_default_unsold_estimates_rule() -> EventRule:
    """Create the default unsold estimates detection rule"""
    return EventRule(
        name="Unsold Estimates Detection",
        description="Detects estimates that are dismissed or open (unsold)",
        event_type="unsold_estimates",
        rule_type=RuleType.SINGLE_TABLE,
        target_tables=["estimates"],
        detection_logic={
            "include_statuses": ["Dismissed", "Open"],
            "exclude_substrings": ["This is an empty"]
        },
        output_fields=[
            "entity_type",
            "entity_id",
            "status",
            "summary",
            "customer_id",
            "location_id",
            "total",
            "created_date"
        ],
        severity=EventSeverity.MEDIUM,
        enabled=True
    )


def get_default_permit_matching_rule() -> EventRule:
    """Create the default permit matching rule"""
    return EventRule(
        name="Permit Address Matching",
        description="Matches location addresses to mechanical permits using fuzzy address matching",
        event_type="permit_matches",
        rule_type=RuleType.ADDRESS_MATCHING,
        target_tables=["locations"],
        detection_logic={
            "match_types": ["EXACT", "FUZZY"],
            "min_score": 0.8,
            "max_edit_distance": 2
        },
        output_fields=[
            "entity_type",
            "entity_id",
            "match_type",
            "score",
            "permit_count",
            "permits"
        ],
        severity=EventSeverity.LOW,
        enabled=True
    )


def get_default_lost_customers_rule() -> EventRule:
    """Create the default lost customers detection rule"""
    return EventRule(
        name="Lost Customers Analysis",
        description="Identifies customers who used other contractors based on permit data analysis",
        event_type="lost_customers",
        rule_type=RuleType.ADDRESS_MATCHING,
        target_tables=["customers", "calls", "permits"],
        detection_logic={
            "exclude_contractors": ["McCullough Heating & Air", "McCullough Heating and Air"],
            "analysis_period": "historical",
            "address_matching": {
                "match_types": ["EXACT", "FUZZY"],
                "min_score": 0.8
            },
            "contact_analysis": {
                "required_tables": ["calls"],
                "date_columns": ["call_date", "created_date", "contact_date"]
            },
            "permit_analysis": {
                "contractor_column": "contractor_company_name",
                "date_columns": ["applied_date", "issued_date"],
                "address_column": "original_address_1"
            }
        },
        output_fields=[
            "entity_type",
            "entity_id",
            "customer_id",
            "first_contact_date",
            "last_contact_date", 
            "competitor_used",
            "shopper_customer",
            "lost_customer",
            "permits_analyzed",
            "customer_name",
            "phone",
            "address",
            "analysis_period"
        ],
        severity=EventSeverity.HIGH,
        enabled=True
    )


def get_default_market_share_rule() -> EventRule:
    """Create the default market share analysis rule"""
    return EventRule(
        name="Market Share Analysis",
        description="Analyzes McCullough's market share of mechanical permits by year",
        event_type="market_share",
        rule_type=RuleType.DATA_AGGREGATION,
        target_tables=["permits"],
        detection_logic={
            "years_back": 10,
            "min_market_share": 5.0,
            "contractor_variants": ["McCullough Heating & Air", "McCullough Heating and Air"]
        },
        output_fields=[
            "entity_type",
            "entity_id", 
            "year",
            "total_permits",
            "mcc_permits",
            "market_share_pct",
            "analysis_period"
        ],
        severity=EventSeverity.MEDIUM,
        enabled=True
    )


def get_default_system_age_audit_rule() -> EventRule:
    """Create the default system age audit rule"""
    return EventRule(
        name="System Age Audit",
        description="Audits locations with aging systems based on Current_System_Age column",
        event_type="system_age_audit",
        rule_type=RuleType.SINGLE_TABLE,
        target_tables=["locations"],
        detection_logic={
            "min_age": 15
        },
        output_fields=[
            "entity_type",
            "entity_id",
            "system_age",
            "customer_id",
            "name",
            "phone",
            "city",
            "state",
            "zip"
        ],
        severity=EventSeverity.MEDIUM,
        enabled=True
    )


def get_default_rules() -> List[EventRule]:
    """Get all default event rules"""
    return [
        get_default_overdue_maintenance_rule(),
        get_default_aging_systems_rule(),
        get_default_canceled_jobs_rule(),
        get_default_unsold_estimates_rule(),
        get_default_permit_matching_rule(),
        get_default_lost_customers_rule(),
        get_default_system_age_audit_rule(),
        # Future: Add other default rules here
        # get_default_permit_replacements_rule(),
    ]


def create_rule_from_config(rule_config: Dict) -> EventRule:
    """Create an EventRule from configuration dictionary"""
    return EventRule(
        name=rule_config.get("name", "Unnamed Rule"),
        description=rule_config.get("description", ""),
        event_type=rule_config.get("event_type", "custom"),
        rule_type=RuleType(rule_config.get("rule_type", "single_table")),
        target_tables=rule_config.get("target_tables", []),
        detection_logic=rule_config.get("detection_logic", {}),
        output_fields=rule_config.get("output_fields", []),
        severity=EventSeverity(rule_config.get("severity", "medium")),
        enabled=rule_config.get("enabled", True),
        threshold_months=rule_config.get("threshold_months")
    )


def rule_to_config(rule: EventRule) -> Dict:
    """Convert EventRule to configuration dictionary"""
    return {
        "name": rule.name,
        "description": rule.description,
        "event_type": rule.event_type,
        "rule_type": rule.rule_type.value,
        "target_tables": rule.target_tables,
        "detection_logic": rule.detection_logic,
        "output_fields": rule.output_fields,
        "severity": rule.severity.value,
        "enabled": rule.enabled,
        "threshold_months": rule.threshold_months
    }
