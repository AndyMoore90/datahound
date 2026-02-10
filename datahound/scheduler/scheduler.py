"""Main scheduler service for DataHound Pro"""

import time
import threading
import pytz
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import traceback
import sys

from .tasks import ScheduledTask, TaskType, TaskStatus, ScheduleType, DayOfWeek
from .persistence import SchedulerPersistence
from .executor import TaskExecutor


class DataHoundScheduler:
    """Main scheduler service that manages and executes scheduled tasks"""
    
    def __init__(self, base_dir: Path, timezone: str = "US/Pacific"):
        """Initialize the scheduler
        
        Args:
            base_dir: Base directory for data storage
            timezone: Timezone for scheduling (default: US/Pacific)
        """
        self.base_dir = Path(base_dir)
        self.timezone = pytz.timezone(timezone)
        self.persistence = SchedulerPersistence(self.base_dir / "data")
        self.executor = TaskExecutor(self.base_dir)
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Task execution callbacks
        self.on_task_start: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        self.on_task_error: Optional[Callable] = None
    
    def start(self, daemon: bool = True):
        """Start the scheduler service"""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        # Create heartbeat file to indicate scheduler is running
        self._create_heartbeat_file()
        
        self._thread = threading.Thread(target=self._run_scheduler, daemon=daemon)
        self._thread.start()
        
        print(f"Scheduler started (timezone: {self.timezone.zone})")
    
    def stop(self, timeout: int = 10):
        """Stop the scheduler service"""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        
        # Remove heartbeat file
        self._remove_heartbeat_file()
        
        print("Scheduler stopped")
    
    def _create_heartbeat_file(self):
        """Create heartbeat file to indicate scheduler is running"""
        try:
            heartbeat_file = self.base_dir / "data" / "scheduler" / ".heartbeat"
            heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
            heartbeat_file.write_text(str(datetime.now().timestamp()))
        except Exception:
            pass  # Don't let heartbeat errors break the scheduler
    
    def _remove_heartbeat_file(self):
        """Remove heartbeat file when scheduler stops"""
        try:
            heartbeat_file = self.base_dir / "data" / "scheduler" / ".heartbeat"
            if heartbeat_file.exists():
                heartbeat_file.unlink()
        except Exception:
            pass
    
    def _update_heartbeat(self):
        """Update heartbeat timestamp"""
        try:
            heartbeat_file = self.base_dir / "data" / "scheduler" / ".heartbeat"
            if heartbeat_file.exists():
                heartbeat_file.write_text(str(datetime.now().timestamp()))
        except Exception:
            pass
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        while self._running:
            try:
                # Update heartbeat
                self._update_heartbeat()
                
                # Get current time in configured timezone
                now = datetime.now(self.timezone)
                
                # Load and check tasks
                tasks = self.persistence.get_active_tasks()
                
                # Sort tasks by execution priority
                tasks = self._sort_tasks_by_priority(tasks)
                
                for task in tasks:
                    if self._should_run_task(task, now):
                        self._execute_task(task)
                
                # Sleep for a short interval before next check
                self._stop_event.wait(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                traceback.print_exc()
                self._stop_event.wait(60)  # Wait longer on error
    
    def _sort_tasks_by_priority(self, tasks: List[ScheduledTask]) -> List[ScheduledTask]:
        """Sort tasks by execution priority"""
        priority_order = {
            TaskType.DOWNLOAD: 1,
            TaskType.PREPARE: 2,
            TaskType.INTEGRATED_UPSERT: 3,
            TaskType.HISTORICAL_EVENT_SCAN: 4,
            TaskType.CUSTOM_EXTRACTION: 8,
            TaskType.REFRESH_CORE_DATA: 9,
            TaskType.CREATE_CORE_DATA: 10,
        }
        
        event_type_priority = {
            "unsold_estimates": 4,
            "canceled_jobs": 5,
            "overdue_maintenance": 6,
            "lost_customers": 7,
        }
        
        def get_task_priority(task: ScheduledTask) -> int:
            base_priority = priority_order.get(task.task_config.task_type, 99)
            
            if task.task_config.task_type == TaskType.HISTORICAL_EVENT_SCAN:
                event_type = task.task_config.event_type
                if event_type:
                    return event_type_priority.get(event_type, base_priority)
            
            return base_priority
        
        return sorted(tasks, key=get_task_priority)
    
    def _should_run_task(self, task: ScheduledTask, now: datetime) -> bool:
        """Check if a task should run now"""
        # Skip if task is not active or currently running
        if task.status != TaskStatus.ACTIVE:
            return False
        
        # Calculate next run time if not set
        if task.next_run is None:
            task.next_run = self._calculate_next_run(task, now)
            self.persistence.update_task(task)
            return False
        
        # Check if it's time to run
        next_run_aware = task.next_run
        if next_run_aware.tzinfo is None:
            next_run_aware = self.timezone.localize(task.next_run)
        
        return now >= next_run_aware
    
    def _calculate_next_run(self, task: ScheduledTask, from_time: datetime) -> datetime:
        """Calculate the next run time for a task"""
        # Ensure from_time is timezone-aware
        if from_time.tzinfo is None:
            from_time = self.timezone.localize(from_time)
        
        if task.schedule_type == ScheduleType.INTERVAL:
            # Simple interval scheduling
            if task.last_run:
                last_run_aware = task.last_run
                if last_run_aware.tzinfo is None:
                    last_run_aware = self.timezone.localize(task.last_run)
                return last_run_aware + timedelta(minutes=task.interval_minutes)
            else:
                return from_time
        
        elif task.schedule_type == ScheduleType.DAILY_TIME:
            # Daily at specific time
            if task.daily_time:
                next_run = from_time.replace(
                    hour=task.daily_time.hour,
                    minute=task.daily_time.minute,
                    second=0,
                    microsecond=0
                )
                if next_run <= from_time:
                    next_run += timedelta(days=1)
                return next_run
        
        elif task.schedule_type == ScheduleType.WEEKLY_SCHEDULE:
            # Weekly schedule with time windows
            if task.weekly_schedule and task.weekly_schedule.days:
                schedule = task.weekly_schedule
                
                # Find next valid day and time
                current_weekday = from_time.weekday()
                
                for days_ahead in range(7):
                    check_date = from_time + timedelta(days=days_ahead)
                    check_weekday = check_date.weekday()
                    
                    # Check if this day is scheduled
                    for day in schedule.days:
                        if DayOfWeek.get_weekday_number(day) == check_weekday:
                            # Calculate time within window
                            if schedule.time_window:
                                start_time = check_date.replace(
                                    hour=schedule.time_window.start_time.hour,
                                    minute=schedule.time_window.start_time.minute,
                                    second=0,
                                    microsecond=0
                                )
                                
                                # If we're past the window today, skip to next scheduled day
                                end_time = check_date.replace(
                                    hour=schedule.time_window.end_time.hour,
                                    minute=schedule.time_window.end_time.minute,
                                    second=0,
                                    microsecond=0
                                )
                                
                                if days_ahead == 0:
                                    # Today - check if we're still in the window
                                    if from_time >= end_time:
                                        continue  # Past window, try next day
                                    
                                    # Calculate next interval time
                                    if task.last_run:
                                        last_run_aware = task.last_run
                                        if last_run_aware.tzinfo is None:
                                            last_run_aware = self.timezone.localize(task.last_run)
                                        
                                        next_interval = last_run_aware + timedelta(minutes=schedule.interval_minutes)
                                        if next_interval > from_time and next_interval < end_time:
                                            return next_interval
                                    
                                    # First run today or past interval
                                    if from_time < start_time:
                                        return start_time
                                    else:
                                        # Next interval within today's window
                                        minutes_since_start = (from_time - start_time).total_seconds() / 60
                                        intervals_passed = int(minutes_since_start / schedule.interval_minutes)
                                        next_interval_time = start_time + timedelta(minutes=(intervals_passed + 1) * schedule.interval_minutes)
                                        
                                        if next_interval_time < end_time:
                                            return next_interval_time
                                        else:
                                            continue  # No more intervals today
                                else:
                                    # Future day - start at beginning of window
                                    return start_time
        
        # Default: run in 1 hour
        return from_time + timedelta(hours=1)
    
    def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task"""
        # Update task status
        task.status = TaskStatus.RUNNING
        self.persistence.update_task(task)
        
        # Notify start
        if self.on_task_start:
            self.on_task_start(task)
        
        start_time = time.time()
        success = False
        error_message = ""
        
        try:
            # Execute the task
            print(f"Executing task: {task.name} ({task.task_id})")
            result = self.executor.execute_task(task)
            success = result.get("success", False)
            error_message = result.get("error", "")
            
            if success:
                task.consecutive_errors = 0
                print(f"Task completed successfully: {task.name}")
            else:
                task.consecutive_errors += 1
                task.last_error = error_message
                print(f"Task failed: {task.name} - {error_message}")
                
        except Exception as e:
            success = False
            error_message = str(e)
            task.consecutive_errors += 1
            task.last_error = error_message
            print(f"Task execution error: {task.name} - {e}")
            traceback.print_exc()
        
        finally:
            # Update task after execution
            duration = time.time() - start_time
            now = datetime.now(self.timezone)
            
            task.last_run = now
            task.run_count += 1
            task.next_run = self._calculate_next_run(task, now)
            
            # Update status based on errors
            if task.consecutive_errors >= 5:
                task.status = TaskStatus.ERROR
            else:
                task.status = TaskStatus.ACTIVE
            
            self.persistence.update_task(task)
            
            # Log execution
            self.persistence.log_task_execution(
                task.task_id,
                success,
                error_message,
                duration
            )
            
            # Notify completion
            if success and self.on_task_complete:
                self.on_task_complete(task)
            elif not success and self.on_task_error:
                self.on_task_error(task, error_message)
    
    def add_task(self, task: ScheduledTask) -> bool:
        """Add a new scheduled task"""
        # Calculate initial next run time
        now = datetime.now(self.timezone)
        task.next_run = self._calculate_next_run(task, now)
        
        return self.persistence.add_task(task)
    
    def update_task(self, task: ScheduledTask) -> bool:
        """Update an existing task"""
        # Recalculate next run if schedule changed
        now = datetime.now(self.timezone)
        task.next_run = self._calculate_next_run(task, now)
        
        return self.persistence.update_task(task)
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a scheduled task"""
        return self.persistence.delete_task(task_id)
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a specific task"""
        return self.persistence.get_task(task_id)
    
    def get_all_tasks(self) -> List[ScheduledTask]:
        """Get all scheduled tasks"""
        return self.persistence.load_tasks()
    
    def get_tasks_by_company(self, company: str) -> List[ScheduledTask]:
        """Get all tasks for a specific company"""
        return self.persistence.get_tasks_by_company(company)
    
    def pause_task(self, task_id: str) -> bool:
        """Pause a scheduled task"""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.PAUSED
            return self.persistence.update_task(task)
        return False
    
    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task"""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.ACTIVE
            task.consecutive_errors = 0  # Reset error count
            
            # Recalculate next run
            now = datetime.now(self.timezone)
            task.next_run = self._calculate_next_run(task, now)
            
            return self.persistence.update_task(task)
        return False
    
    def run_task_now(self, task_id: str) -> Dict[str, Any]:
        """Execute a task immediately"""
        task = self.get_task(task_id)
        if not task:
            return {"success": False, "error": "Task not found"}
        
        # Execute in current thread
        self._execute_task(task)
        
        return {"success": True}

