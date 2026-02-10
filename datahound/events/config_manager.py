"""Configuration management for event system"""

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar, List
import pandas as pd

from .types import (
    EventSystemConfig, EventTypeConfig, MaintenanceEventConfig, 
    AgingSystemsConfig, CanceledJobsConfig, UnsoldEstimatesConfig,
    PermitMatchingConfig, LostCustomersConfig, MarketShareConfig,
    SystemAgeAuditConfig, PermitReplacementsConfig, EventSeverity
)

T = TypeVar('T', bound=EventTypeConfig)


class EventConfigManager:
    """Manages loading, saving, and validation of event configurations"""
    
    def __init__(self, company: str, config_dir: Path):
        self.company = company
        self.config_dir = config_dir
        self.config_file = config_dir / f"{company}_event_config.json"
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> EventSystemConfig:
        """Load event system configuration from file"""
        
        if not self.config_file.exists():
            return EventSystemConfig()  # Return default config
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            return self._dict_to_config(config_data)
            
        except Exception:
            # Return default config if loading fails
            return EventSystemConfig()
    
    def save_config(self, config: EventSystemConfig) -> bool:
        """Save event system configuration to file"""
        
        try:
            config_dict = self._config_to_dict(config)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception:
            return False
    
    def _config_to_dict(self, config: EventSystemConfig) -> Dict[str, Any]:
        """Convert EventSystemConfig to dictionary"""
        
        config_dict = {}
        
        # Convert each event type config
        event_types = [
            "overdue_maintenance", "aging_systems", "canceled_jobs",
            "unsold_estimates", "permit_matches", "lost_customers",
            "market_share", "system_age_audit", "permit_replacements"
        ]
        
        for event_type in event_types:
            event_config = getattr(config, event_type)
            config_dict[event_type] = asdict(event_config)
            
            # Convert enums to strings
            if "severity_threshold" in config_dict[event_type]:
                config_dict[event_type]["severity_threshold"] = config_dict[event_type]["severity_threshold"].value
        
        # Add global settings
        config_dict["global_settings"] = {
            "global_processing_limit": config.global_processing_limit,
            "show_progress": config.show_progress,
            "chunk_size": config.chunk_size,
            "backup_before_scan": config.backup_before_scan
        }
        
        return config_dict
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> EventSystemConfig:
        """Convert dictionary to EventSystemConfig"""
        
        config = EventSystemConfig()
        
        # Map of event types to their config classes
        config_classes = {
            "overdue_maintenance": MaintenanceEventConfig,
            "aging_systems": AgingSystemsConfig,
            "canceled_jobs": CanceledJobsConfig,
            "unsold_estimates": UnsoldEstimatesConfig,
            "permit_matches": PermitMatchingConfig,
            "lost_customers": LostCustomersConfig,
            "market_share": MarketShareConfig,
            "system_age_audit": SystemAgeAuditConfig,
            "permit_replacements": PermitReplacementsConfig
        }
        
        # Load each event type config
        for event_type, config_class in config_classes.items():
            if event_type in config_dict:
                event_config_dict = config_dict[event_type].copy()
                
                # Convert severity string back to enum
                if "severity_threshold" in event_config_dict:
                    severity_str = event_config_dict["severity_threshold"]
                    event_config_dict["severity_threshold"] = EventSeverity(severity_str)
                
                # Create config instance
                try:
                    event_config = config_class(**event_config_dict)
                    setattr(config, event_type, event_config)
                except Exception:
                    # Use default if creation fails
                    setattr(config, event_type, config_class())
        
        # Load global settings
        if "global_settings" in config_dict:
            global_settings = config_dict["global_settings"]
            config.global_processing_limit = global_settings.get("global_processing_limit")
            config.show_progress = global_settings.get("show_progress", True)
            config.chunk_size = global_settings.get("chunk_size", 100)
            config.backup_before_scan = global_settings.get("backup_before_scan", True)
        
        return config
    
    def get_table_schemas(self, parquet_dir: Path) -> Dict[str, List[str]]:
        """Get available columns for each table"""
        
        schemas = {}
        
        if not parquet_dir.exists():
            return schemas
        
        for parquet_file in parquet_dir.glob("*.parquet"):
            table_name = parquet_file.stem.lower()
            
            try:
                df = pd.read_parquet(parquet_file, columns=[])  # Read only schema
                schemas[table_name] = list(df.columns)
            except Exception:
                schemas[table_name] = []
        
        return schemas
    
    def validate_config(self, config: EventSystemConfig) -> Dict[str, List[str]]:
        """Validate configuration and return any issues"""
        
        issues = {}
        
        # Validate each event type config
        event_types = [
            "overdue_maintenance", "aging_systems", "canceled_jobs",
            "unsold_estimates", "permit_matches", "lost_customers", 
            "market_share", "system_age_audit", "permit_replacements"
        ]
        
        for event_type in event_types:
            event_config = getattr(config, event_type)
            event_issues = self._validate_event_config(event_type, event_config)
            
            if event_issues:
                issues[event_type] = event_issues
        
        # Validate global settings
        global_issues = []
        
        if config.chunk_size <= 0:
            global_issues.append("Chunk size must be positive")
        
        if config.global_processing_limit is not None and config.global_processing_limit <= 0:
            global_issues.append("Global processing limit must be positive")
        
        if global_issues:
            issues["global_settings"] = global_issues
        
        return issues
    
    def _validate_event_config(self, event_type: str, config: EventTypeConfig) -> List[str]:
        """Validate individual event type configuration"""
        
        issues = []
        
        # Common validations
        if config.processing_limit is not None and config.processing_limit <= 0:
            issues.append("Processing limit must be positive")
        
        # Event-specific validations
        if event_type == "overdue_maintenance" and isinstance(config, MaintenanceEventConfig):
            if config.months_threshold_min <= 0:
                issues.append("Minimum months threshold must be positive")
            
            if config.months_threshold_max is not None and config.months_threshold_max <= config.months_threshold_min:
                issues.append("Maximum months threshold must be greater than minimum")
        
        elif event_type == "aging_systems" and isinstance(config, AgingSystemsConfig):
            if config.min_age_years <= 0:
                issues.append("Minimum age years must be positive")
            
            if config.max_jobs_to_analyze <= 0:
                issues.append("Max jobs to analyze must be positive")
        
        elif event_type == "canceled_jobs" and isinstance(config, CanceledJobsConfig):
            if config.hours_back <= 0:
                issues.append("Hours back must be positive")
            
            if config.min_job_value is not None and config.min_job_value < 0:
                issues.append("Minimum job value cannot be negative")
        
        elif event_type == "unsold_estimates" and isinstance(config, UnsoldEstimatesConfig):
            if config.days_back <= 0:
                issues.append("Days back must be positive")
            
            if config.min_estimate_value is not None and config.min_estimate_value < 0:
                issues.append("Minimum estimate value cannot be negative")
        
        elif event_type == "permit_matches" and isinstance(config, PermitMatchingConfig):
            if config.match_radius_miles <= 0:
                issues.append("Match radius must be positive")
            
            if not (0 <= config.min_match_score <= 1):
                issues.append("Match score must be between 0 and 1")
        
        return issues
    
    def create_template_config(self) -> EventSystemConfig:
        """Create a template configuration with reasonable defaults"""
        
        config = EventSystemConfig()
        
        # Enable some common event types by default
        config.overdue_maintenance.enabled = True
        config.canceled_jobs.enabled = True
        config.unsold_estimates.enabled = True
        
        # Set reasonable defaults
        config.overdue_maintenance.months_threshold_min = 12
        config.overdue_maintenance.months_threshold_max = 36
        
        config.canceled_jobs.hours_back = 48
        config.unsold_estimates.days_back = 14
        
        return config
    
    def export_config_summary(self, config: EventSystemConfig) -> Dict[str, Any]:
        """Export a summary of the configuration for display"""
        
        summary = {
            "enabled_event_types": config.get_enabled_event_types(),
            "total_event_types": 9,
            "global_settings": {
                "processing_limit": config.global_processing_limit,
                "show_progress": config.show_progress,
                "chunk_size": config.chunk_size,
                "backup_enabled": config.backup_before_scan
            },
            "event_details": {}
        }
        
        # Add details for enabled events
        for event_type in summary["enabled_event_types"]:
            event_config = config.get_config_for_event_type(event_type)
            if event_config:
                summary["event_details"][event_type] = {
                    "severity_threshold": event_config.severity_threshold.value,
                    "detection_mode": event_config.detection_mode,
                    "processing_limit": event_config.processing_limit,
                    "selected_columns_count": len(event_config.selected_columns),
                    "enrichment_enabled": any(event_config.enrichment.values())
                }
        
        return summary
