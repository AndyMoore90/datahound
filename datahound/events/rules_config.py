"""
Configurable rules system for event detection and recent events management
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from enum import Enum
import json
from pathlib import Path


class EventDetectionTrigger(Enum):
    """Types of triggers for event detection"""
    STATUS_CHANGE = "status_change"
    TIME_BASED = "time_based"
    VALUE_CHANGE = "value_change"
    COMBINATION = "combination"


class RemovalCondition(Enum):
    """Conditions for removing events from recent events"""
    AGE_BASED = "age_based"
    STATUS_RESOLUTION = "status_resolution"
    SUBSEQUENT_ACTIVITY = "subsequent_activity"
    MANUAL_RESOLUTION = "manual_resolution"
    INBOUND_CALL_ACTIVITY = "inbound_call_activity"
    SMS_ACTIVITY = "sms_activity"


@dataclass
class EventDetectionRule:
    """Configuration for how events are detected"""
    rule_id: str
    name: str
    description: str
    event_type: str
    trigger_type: EventDetectionTrigger
    source_table: str
    conditions: Dict[str, Any]
    enabled: bool = True
    priority: int = 1  # 1=High, 2=Medium, 3=Low


@dataclass
class RecentEventRemovalRule:
    """Configuration for how events are removed from recent events"""
    rule_id: str
    name: str
    description: str
    event_type: str
    removal_condition: RemovalCondition
    parameters: Dict[str, Any]
    enabled: bool = True
    order: int = 1  # Order of evaluation


class EventRulesManager:
    """Manages event detection and removal rules configuration"""
    
    def __init__(self, company: str, config_dir: Path):
        self.company = company
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.detection_rules_file = config_dir / f"{company}_event_detection_rules.json"
        self.removal_rules_file = config_dir / f"{company}_event_removal_rules.json"
        
        # Initialize with default rules if files don't exist
        if not self.detection_rules_file.exists():
            self._create_default_detection_rules()
        
        if not self.removal_rules_file.exists():
            self._create_default_removal_rules()
    
    def get_detection_rules(self) -> List[EventDetectionRule]:
        """Get all event detection rules"""
        
        try:
            with open(self.detection_rules_file, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            # Convert trigger_type strings back to enums
            for rule_data in rules_data:
                if isinstance(rule_data.get('trigger_type'), str):
                    rule_data['trigger_type'] = EventDetectionTrigger(rule_data['trigger_type'])
            
            return [EventDetectionRule(**rule) for rule in rules_data]
        except Exception as e:
            print(f"Error loading detection rules: {e}")
            return self._get_default_detection_rules()
    
    def get_removal_rules(self) -> List[RecentEventRemovalRule]:
        """Get all recent event removal rules"""
        
        try:
            with open(self.removal_rules_file, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            # Convert removal_condition strings back to enums
            for rule_data in rules_data:
                if isinstance(rule_data.get('removal_condition'), str):
                    rule_data['removal_condition'] = RemovalCondition(rule_data['removal_condition'])
            
            return [RecentEventRemovalRule(**rule) for rule in rules_data]
        except Exception as e:
            print(f"Error loading removal rules: {e}")
            return self._get_default_removal_rules()
    
    def save_detection_rules(self, rules: List[EventDetectionRule]):
        """Save event detection rules"""
        
        try:
            rules_data = [asdict(rule) for rule in rules]
            # Convert enums to strings
            for rule_data in rules_data:
                rule_data['trigger_type'] = rule_data['trigger_type'].value
            
            with open(self.detection_rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2)
        except Exception as e:
            print(f"Error saving detection rules: {e}")
    
    def save_removal_rules(self, rules: List[RecentEventRemovalRule]):
        """Save recent event removal rules"""
        
        try:
            rules_data = [asdict(rule) for rule in rules]
            # Convert enums to strings
            for rule_data in rules_data:
                rule_data['removal_condition'] = rule_data['removal_condition'].value
            
            with open(self.removal_rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2)
        except Exception as e:
            print(f"Error saving removal rules: {e}")
    
    def get_detection_rule_by_event_type(self, event_type: str) -> List[EventDetectionRule]:
        """Get detection rules for specific event type"""
        
        all_rules = self.get_detection_rules()
        return [rule for rule in all_rules if rule.event_type == event_type and rule.enabled]
    
    def get_removal_rules_by_event_type(self, event_type: str) -> List[RecentEventRemovalRule]:
        """Get removal rules for specific event type"""
        
        all_rules = self.get_removal_rules()
        return [rule for rule in all_rules if rule.event_type == event_type and rule.enabled]
    
    def _create_default_detection_rules(self):
        """Create default event detection rules"""
        
        default_rules = self._get_default_detection_rules()
        self.save_detection_rules(default_rules)
    
    def _create_default_removal_rules(self):
        """Create default event removal rules"""
        
        default_rules = self._get_default_removal_rules()
        self.save_removal_rules(default_rules)
    
    def _get_default_detection_rules(self) -> List[EventDetectionRule]:
        """Get default event detection rules"""
        
        return [
            # Job Cancellation Detection
            EventDetectionRule(
                rule_id="job_canceled_001",
                name="Job Status Changed to Canceled",
                description="Detects when a job status changes to 'canceled' or 'cancelled'",
                event_type="job_canceled",
                trigger_type=EventDetectionTrigger.STATUS_CHANGE,
                source_table="jobs",
                conditions={
                    "column": "Status",
                    "old_values": ["scheduled", "in progress", "pending"],
                    "new_values": ["canceled", "cancelled"],
                    "case_sensitive": False
                },
                enabled=True,
                priority=1
            ),
            
            # Job Completion Detection
            EventDetectionRule(
                rule_id="job_completed_001",
                name="Job Status Changed to Completed",
                description="Detects when a job status changes to 'completed'",
                event_type="job_completed",
                trigger_type=EventDetectionTrigger.STATUS_CHANGE,
                source_table="jobs",
                conditions={
                    "column": "Status",
                    "old_values": ["scheduled", "in progress", "pending", "canceled"],
                    "new_values": ["completed", "finished", "complete"],
                    "case_sensitive": False
                },
                enabled=True,
                priority=1
            ),
            
            # Job Rescheduled Detection
            EventDetectionRule(
                rule_id="job_rescheduled_001",
                name="Job Rescheduled After Cancellation",
                description="Detects when a canceled job is rescheduled",
                event_type="job_rescheduled",
                trigger_type=EventDetectionTrigger.STATUS_CHANGE,
                source_table="jobs",
                conditions={
                    "column": "Status",
                    "old_values": ["canceled", "cancelled"],
                    "new_values": ["scheduled", "in progress"],
                    "case_sensitive": False
                },
                enabled=True,
                priority=1
            ),
            
            # Estimate Dismissed Detection
            EventDetectionRule(
                rule_id="estimate_dismissed_001",
                name="Estimate Status Changed to Dismissed",
                description="Detects when an estimate is dismissed or declined",
                event_type="estimate_dismissed",
                trigger_type=EventDetectionTrigger.STATUS_CHANGE,
                source_table="estimates",
                conditions={
                    "column": "Estimate Status",
                    "old_values": ["pending", "open", "sent"],
                    "new_values": ["dismissed", "declined", "rejected"],
                    "case_sensitive": False,
                    "minimum_days_open": 7
                },
                enabled=True,
                priority=2
            ),
            
            # Estimate Sold Detection
            EventDetectionRule(
                rule_id="estimate_sold_001",
                name="Estimate Status Changed to Sold",
                description="Detects when an estimate is sold or accepted",
                event_type="estimate_sold",
                trigger_type=EventDetectionTrigger.STATUS_CHANGE,
                source_table="estimates",
                conditions={
                    "column": "Estimate Status",
                    "old_values": ["pending", "open", "sent", "dismissed"],
                    "new_values": ["sold", "accepted", "approved"],
                    "case_sensitive": False
                },
                enabled=True,
                priority=1
            ),
            
            # Overdue Maintenance Detection
            EventDetectionRule(
                rule_id="overdue_maintenance_001",
                name="Customer Overdue for Maintenance",
                description="Detects customers who haven't had maintenance in 12+ months",
                event_type="overdue_maintenance",
                trigger_type=EventDetectionTrigger.TIME_BASED,
                source_table="jobs",
                conditions={
                    "job_types": ["maintenance", "service", "repair", "tune", "clean"],
                    "date_column": "Completion Date",
                    "fallback_date_column": "Created Date",
                    "months_threshold": 12,
                    "severity_thresholds": {
                        "medium": 15,
                        "high": 24,
                        "critical": 36
                    }
                },
                enabled=True,
                priority=2
            ),
            
            # Customer Contact Detection
            EventDetectionRule(
                rule_id="customer_contact_001",
                name="Customer Contact Logged",
                description="Detects when customer contact is logged",
                event_type="customer_contact",
                trigger_type=EventDetectionTrigger.VALUE_CHANGE,
                source_table="calls",
                conditions={
                    "columns": ["Call Type", "Type", "Reason"],
                    "track_all_changes": True
                },
                enabled=True,
                priority=3
            )
        ]
    
    def _get_default_removal_rules(self) -> List[RecentEventRemovalRule]:
        """Get default recent event removal rules"""
        
        return [
            # Cancellation Removal Rules
            RecentEventRemovalRule(
                rule_id="cancel_remove_001",
                name="Remove Cancellations After 14 Days",
                description="Automatically remove cancellations that are older than 14 days",
                event_type="cancellations",
                removal_condition=RemovalCondition.AGE_BASED,
                parameters={
                    "max_days": 14,
                    "archive_reason": "aged_out_14_days"
                },
                enabled=True,
                order=1
            ),
            
            RecentEventRemovalRule(
                rule_id="cancel_remove_002",
                name="Remove Cancellations When Job Completed",
                description="Remove cancellations when customer completes any job",
                event_type="cancellations",
                removal_condition=RemovalCondition.SUBSEQUENT_ACTIVITY,
                parameters={
                    "activity_type": "job_completed",
                    "match_customer": True,
                    "match_entity": False,
                    "archive_reason": "job_completed_after_cancellation"
                },
                enabled=True,
                order=2
            ),
            
            RecentEventRemovalRule(
                rule_id="cancel_remove_003",
                name="Remove Cancellations When Job Rescheduled",
                description="Remove cancellations when the same job is rescheduled",
                event_type="cancellations",
                removal_condition=RemovalCondition.STATUS_RESOLUTION,
                parameters={
                    "resolution_event": "job_rescheduled",
                    "match_entity": True,
                    "archive_reason": "job_rescheduled"
                },
                enabled=True,
                order=3
            ),
            
            # Overdue Maintenance Removal Rules
            RecentEventRemovalRule(
                rule_id="maintenance_remove_001",
                name="Remove When Maintenance Completed",
                description="Remove overdue maintenance when customer completes maintenance",
                event_type="overdue_maintenance",
                removal_condition=RemovalCondition.SUBSEQUENT_ACTIVITY,
                parameters={
                    "activity_type": "maintenance_completed",
                    "job_types": ["maintenance", "service", "repair", "tune", "clean"],
                    "match_customer": True,
                    "archive_reason": "maintenance_completed"
                },
                enabled=True,
                order=1
            ),
            
            RecentEventRemovalRule(
                rule_id="maintenance_remove_002",
                name="Archive Critical Overdue (36+ Months)",
                description="Archive maintenance that's been overdue for more than 36 months",
                event_type="overdue_maintenance",
                removal_condition=RemovalCondition.AGE_BASED,
                parameters={
                    "condition_type": "months_overdue",
                    "threshold": 36,
                    "archive_reason": "critical_overdue_archive",
                    "preserve_in_critical_list": True
                },
                enabled=True,
                order=2
            ),
            
            # Unsold Estimates Removal Rules
            RecentEventRemovalRule(
                rule_id="estimate_remove_001",
                name="Remove When Estimate Sold",
                description="Remove unsold estimates when they are sold",
                event_type="unsold_estimates",
                removal_condition=RemovalCondition.STATUS_RESOLUTION,
                parameters={
                    "resolution_event": "estimate_sold",
                    "match_entity": True,
                    "archive_reason": "estimate_sold"
                },
                enabled=True,
                order=1
            ),
            
            RecentEventRemovalRule(
                rule_id="estimate_remove_002",
                name="Remove When Job Completed",
                description="Remove unsold estimates when customer completes any job",
                event_type="unsold_estimates",
                removal_condition=RemovalCondition.SUBSEQUENT_ACTIVITY,
                parameters={
                    "activity_type": "job_completed",
                    "match_customer": True,
                    "match_entity": False,
                    "archive_reason": "subsequent_job_activity"
                },
                enabled=True,
                order=2
            ),
            
            RecentEventRemovalRule(
                rule_id="estimate_remove_003",
                name="Remove Old Unsold Estimates",
                description="Remove unsold estimates older than 60 days",
                event_type="unsold_estimates",
                removal_condition=RemovalCondition.AGE_BASED,
                parameters={
                    "max_days": 60,
                    "archive_reason": "aged_out_60_days"
                },
                enabled=True,
                order=3
            ),
            
            # Lost Customers Removal Rules
            RecentEventRemovalRule(
                rule_id="lost_remove_001",
                name="Remove When Customer Returns",
                description="Remove lost customer status when they make contact or book service",
                event_type="lost_customers",
                removal_condition=RemovalCondition.SUBSEQUENT_ACTIVITY,
                parameters={
                    "activity_types": ["customer_contact", "job_scheduled", "estimate_created"],
                    "match_customer": True,
                    "archive_reason": "customer_returned"
                },
                enabled=True,
                order=1
            ),
            
            RecentEventRemovalRule(
                rule_id="lost_remove_002",
                name="Archive Old Lost Customers",
                description="Archive lost customers after 90 days",
                event_type="lost_customers",
                removal_condition=RemovalCondition.AGE_BASED,
                parameters={
                    "max_days": 90,
                    "archive_reason": "aged_out_90_days"
                },
                enabled=True,
                order=2
            )
        ]
    
    def validate_detection_rule(self, rule: EventDetectionRule) -> List[str]:
        """Validate a detection rule and return list of errors"""
        
        errors = []
        
        if not rule.rule_id:
            errors.append("Rule ID is required")
        
        if not rule.name:
            errors.append("Rule name is required")
        
        if not rule.event_type:
            errors.append("Event type is required")
        
        if not rule.source_table:
            errors.append("Source table is required")
        
        if not rule.conditions:
            errors.append("Conditions are required")
        
        # Validate conditions based on trigger type
        if rule.trigger_type == EventDetectionTrigger.STATUS_CHANGE:
            if "column" not in rule.conditions:
                errors.append("Status change rules must specify a column")
            if "new_values" not in rule.conditions:
                errors.append("Status change rules must specify new_values")
        
        elif rule.trigger_type == EventDetectionTrigger.TIME_BASED:
            if "months_threshold" not in rule.conditions:
                errors.append("Time-based rules must specify months_threshold")
        
        return errors
    
    def validate_removal_rule(self, rule: RecentEventRemovalRule) -> List[str]:
        """Validate a removal rule and return list of errors"""
        
        errors = []
        
        if not rule.rule_id:
            errors.append("Rule ID is required")
        
        if not rule.name:
            errors.append("Rule name is required")
        
        if not rule.event_type:
            errors.append("Event type is required")
        
        if not rule.parameters:
            errors.append("Parameters are required")
        
        # Validate parameters based on removal condition
        if rule.removal_condition == RemovalCondition.AGE_BASED:
            if "max_days" not in rule.parameters and "threshold" not in rule.parameters:
                errors.append("Age-based rules must specify max_days or threshold")
        elif rule.removal_condition == RemovalCondition.STATUS_RESOLUTION:
            if "resolution_event" not in rule.parameters:
                errors.append("Status resolution rules must specify resolution_event")
        elif rule.removal_condition == RemovalCondition.SUBSEQUENT_ACTIVITY:
            if "activity_type" not in rule.parameters and "activity_types" not in rule.parameters:
                errors.append("Subsequent activity rules must specify activity_type or activity_types")
        elif rule.removal_condition == RemovalCondition.INBOUND_CALL_ACTIVITY:
            required_keys = ["phone_number_column", "caller_phone_column", "direction_column", "call_date_column", "detection_field", "archive_reason"]
            for key in required_keys:
                if key not in rule.parameters:
                    errors.append(f"Inbound call rules must specify {key}")
        elif rule.removal_condition == RemovalCondition.SMS_ACTIVITY:
            required_keys = ["phone_number_column", "sms_phone_column", "sms_created_column", "detection_field", "archive_reason"]
            for key in required_keys:
                if key not in rule.parameters:
                    errors.append(f"SMS activity rules must specify {key}")
        
        return errors
