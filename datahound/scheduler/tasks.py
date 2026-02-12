"""Task definitions and configuration for DataHound Pro scheduler"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


class TaskType(Enum):
    DOWNLOAD = "download"
    PREPARE = "prepare"
    INTEGRATED_UPSERT = "integrated_upsert"
    HISTORICAL_EVENT_SCAN = "historical_event_scan"
    CUSTOM_EXTRACTION = "custom_extraction"
    CREATE_CORE_DATA = "create_core_data"
    REFRESH_CORE_DATA = "refresh_core_data"
    TRANSCRIPT_PIPELINE = "transcript_pipeline"
    EVENT_UPLOAD = "event_upload"
    SMS_SHEET_SYNC = "sms_sheet_sync"


class ScheduleType(Enum):
    """Types of scheduling patterns"""
    INTERVAL = "interval"  # Every X minutes
    DAILY_TIME = "daily_time"  # Daily at specific time
    WEEKLY_SCHEDULE = "weekly_schedule"  # Specific days and times
    

class TaskStatus(Enum):
    """Status of scheduled tasks"""
    ACTIVE = "active"
    PAUSED = "paused"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


class DayOfWeek(Enum):
    """Days of the week for scheduling"""
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"
    
    @classmethod
    def get_weekday_number(cls, day: 'DayOfWeek') -> int:
        """Get Python weekday number (0=Monday, 6=Sunday)"""
        mapping = {
            cls.MONDAY: 0,
            cls.TUESDAY: 1,
            cls.WEDNESDAY: 2,
            cls.THURSDAY: 3,
            cls.FRIDAY: 4,
            cls.SATURDAY: 5,
            cls.SUNDAY: 6
        }
        return mapping[day]


@dataclass
class TimeWindow:
    """Time window for scheduling"""
    start_time: time  # Start time (e.g., 09:00)
    end_time: time    # End time (e.g., 17:00)
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'TimeWindow':
        return cls(
            start_time=time.fromisoformat(data["start_time"]),
            end_time=time.fromisoformat(data["end_time"])
        )


@dataclass
class WeeklySchedule:
    """Weekly schedule configuration"""
    days: List[DayOfWeek] = field(default_factory=list)
    time_window: Optional[TimeWindow] = None
    interval_minutes: int = 60  # Interval within the time window
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "days": [day.value for day in self.days],
            "time_window": self.time_window.to_dict() if self.time_window else None,
            "interval_minutes": self.interval_minutes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeeklySchedule':
        return cls(
            days=[DayOfWeek(day) for day in data.get("days", [])],
            time_window=TimeWindow.from_dict(data["time_window"]) if data.get("time_window") else None,
            interval_minutes=data.get("interval_minutes", 60)
        )


@dataclass
class TaskConfiguration:
    """Configuration specific to each task type"""
    task_type: TaskType
    company: str
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Download specific
    file_types: List[str] = field(default_factory=list)
    archive_existing: bool = False
    dedup_after: bool = False
    mark_as_read: bool = True
    
    # Prepare specific
    prepare_types: List[str] = field(default_factory=list)
    write_parquet: bool = True
    write_csv: bool = False
    
    # Upsert specific
    backup_files: bool = True
    dry_run: bool = False
    write_mode: str = "inplace"
    
    # Event scan specific
    event_type: Optional[str] = None  # For individual event scans
    scan_all_events: bool = False  # For scanning all events
    
    # Extraction specific
    execute_all_enabled: bool = True
    extraction_configs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Core data specific
    include_rfm: bool = True
    include_demographics: bool = True
    include_permits: bool = True
    include_marketable: bool = True
    include_segments: bool = True
    processing_limit: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "company": self.company,
            "settings": self.settings,
            "file_types": self.file_types,
            "archive_existing": self.archive_existing,
            "dedup_after": self.dedup_after,
            "mark_as_read": self.mark_as_read,
            "prepare_types": self.prepare_types,
            "write_parquet": self.write_parquet,
            "write_csv": self.write_csv,
            "backup_files": self.backup_files,
            "dry_run": self.dry_run,
            "write_mode": self.write_mode,
            "event_type": self.event_type,
            "scan_all_events": self.scan_all_events,
            "execute_all_enabled": self.execute_all_enabled,
            "extraction_configs": self.extraction_configs,
            "include_rfm": self.include_rfm,
            "include_demographics": self.include_demographics,
            "include_permits": self.include_permits,
            "include_marketable": self.include_marketable,
            "include_segments": self.include_segments,
            "processing_limit": self.processing_limit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskConfiguration':
        return cls(
            task_type=TaskType(data["task_type"]),
            company=data["company"],
            settings=data.get("settings", {}),
            file_types=data.get("file_types", []),
            archive_existing=data.get("archive_existing", False),
            dedup_after=data.get("dedup_after", False),
            mark_as_read=data.get("mark_as_read", True),
            prepare_types=data.get("prepare_types", []),
            write_parquet=data.get("write_parquet", True),
            write_csv=data.get("write_csv", False),
            backup_files=data.get("backup_files", True),
            dry_run=data.get("dry_run", False),
            write_mode=data.get("write_mode", "inplace"),
            event_type=data.get("event_type"),
            scan_all_events=data.get("scan_all_events", False),
            execute_all_enabled=data.get("execute_all_enabled", True),
            extraction_configs=data.get("extraction_configs", []),
            include_rfm=data.get("include_rfm", True),
            include_demographics=data.get("include_demographics", True),
            include_permits=data.get("include_permits", True),
            include_marketable=data.get("include_marketable", True),
            include_segments=data.get("include_segments", True),
            processing_limit=data.get("processing_limit")
        )


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    task_id: str
    name: str
    description: str
    task_config: TaskConfiguration
    schedule_type: ScheduleType
    status: TaskStatus = TaskStatus.ACTIVE
    
    # Scheduling configuration
    interval_minutes: Optional[int] = None  # For INTERVAL type
    daily_time: Optional[time] = None  # For DAILY_TIME type
    weekly_schedule: Optional[WeeklySchedule] = None  # For WEEKLY_SCHEDULE type
    
    # Execution tracking
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    last_error: Optional[str] = None
    run_count: int = 0
    consecutive_errors: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "task_config": self.task_config.to_dict(),
            "schedule_type": self.schedule_type.value,
            "status": self.status.value,
            "interval_minutes": self.interval_minutes,
            "daily_time": self.daily_time.isoformat() if self.daily_time else None,
            "weekly_schedule": self.weekly_schedule.to_dict() if self.weekly_schedule else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_error": self.last_error,
            "run_count": self.run_count,
            "consecutive_errors": self.consecutive_errors,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledTask':
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            description=data["description"],
            task_config=TaskConfiguration.from_dict(data["task_config"]),
            schedule_type=ScheduleType(data["schedule_type"]),
            status=TaskStatus(data["status"]),
            interval_minutes=data.get("interval_minutes"),
            daily_time=time.fromisoformat(data["daily_time"]) if data.get("daily_time") else None,
            weekly_schedule=WeeklySchedule.from_dict(data["weekly_schedule"]) if data.get("weekly_schedule") else None,
            last_run=datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None,
            next_run=datetime.fromisoformat(data["next_run"]) if data.get("next_run") else None,
            last_error=data.get("last_error"),
            run_count=data.get("run_count", 0),
            consecutive_errors=data.get("consecutive_errors", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now()
        )

