"""Persistence layer for scheduled tasks"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from contextlib import contextmanager

# fcntl is Unix-only, handle Windows separately
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from .tasks import ScheduledTask, TaskType, TaskStatus

from central_logging.config import scheduler_dir


class SchedulerPersistence:
    """Handles persistence of scheduled tasks"""
    
    def __init__(self, data_dir: Path):
        """Initialize persistence with data directory"""
        self.data_dir = Path(data_dir)
        self.scheduler_dir = self.data_dir / "scheduler"
        self.tasks_file = self.scheduler_dir / "scheduled_tasks.json"
        self.history_file = scheduler_dir() / "task_history.jsonl"
        self.lock_file = self.scheduler_dir / ".scheduler.lock"
        
        # Ensure directories exist
        self.scheduler_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _file_lock(self, timeout: int = 10):
        """Context manager for file locking"""
        lock_fd = None
        try:
            # Windows or no fcntl support - use a simple file-based lock
            if not HAS_FCNTL or os.name == 'nt':
                import time
                start_time = time.time()
                while self.lock_file.exists():
                    if time.time() - start_time > timeout:
                        raise TimeoutError("Failed to acquire scheduler lock")
                    time.sleep(0.1)
                self.lock_file.touch()
                yield
            else:
                # Unix-based systems with fcntl
                lock_fd = open(self.lock_file, 'w')
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield
        finally:
            if not HAS_FCNTL or os.name == 'nt':
                if self.lock_file.exists():
                    try:
                        self.lock_file.unlink()
                    except:
                        pass
            elif lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
    
    def load_tasks(self) -> List[ScheduledTask]:
        """Load all scheduled tasks from storage"""
        if not self.tasks_file.exists():
            return []
        
        try:
            with self._file_lock():
                with open(self.tasks_file, 'r') as f:
                    data = json.load(f)
                    return [ScheduledTask.from_dict(task_data) for task_data in data]
        except Exception as e:
            print(f"Error loading scheduled tasks: {e}")
            return []
    
    def save_tasks(self, tasks: List[ScheduledTask]) -> bool:
        """Save all scheduled tasks to storage"""
        try:
            with self._file_lock():
                data = [task.to_dict() for task in tasks]
                with open(self.tasks_file, 'w') as f:
                    json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving scheduled tasks: {e}")
            return False
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a specific task by ID"""
        tasks = self.load_tasks()
        for task in tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def add_task(self, task: ScheduledTask) -> bool:
        """Add a new scheduled task"""
        tasks = self.load_tasks()
        
        # Check for duplicate ID
        for existing in tasks:
            if existing.task_id == task.task_id:
                return False
        
        tasks.append(task)
        return self.save_tasks(tasks)
    
    def update_task(self, task: ScheduledTask) -> bool:
        """Update an existing scheduled task"""
        tasks = self.load_tasks()
        
        for i, existing in enumerate(tasks):
            if existing.task_id == task.task_id:
                task.updated_at = datetime.now()
                tasks[i] = task
                return self.save_tasks(tasks)
        
        return False
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a scheduled task"""
        tasks = self.load_tasks()
        original_count = len(tasks)
        
        tasks = [task for task in tasks if task.task_id != task_id]
        
        if len(tasks) < original_count:
            return self.save_tasks(tasks)
        return False
    
    def get_active_tasks(self) -> List[ScheduledTask]:
        """Get all active scheduled tasks"""
        tasks = self.load_tasks()
        return [task for task in tasks if task.status == TaskStatus.ACTIVE]
    
    def get_tasks_by_type(self, task_type: TaskType) -> List[ScheduledTask]:
        """Get all tasks of a specific type"""
        tasks = self.load_tasks()
        return [task for task in tasks if task.task_config.task_type == task_type]
    
    def get_tasks_by_company(self, company: str) -> List[ScheduledTask]:
        """Get all tasks for a specific company"""
        tasks = self.load_tasks()
        return [task for task in tasks if task.task_config.company == company]
    
    def log_task_execution(self, task_id: str, success: bool, message: str = "",
                           duration_seconds: float = 0, task_name: str = "",
                           task_type: str = "") -> None:
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "task_name": task_name,
                "task_type": task_type,
                "success": success,
                "message": message,
                "duration_seconds": duration_seconds,
            }
            
            with self._file_lock():
                with open(self.history_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Error logging task execution: {e}")
    
    def get_task_history(self, task_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get task execution history"""
        if not self.history_file.exists():
            return []
        
        history = []
        try:
            with self._file_lock():
                with open(self.history_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        try:
                            entry = json.loads(line.strip())
                            if task_id is None or entry.get("task_id") == task_id:
                                history.append(entry)
                        except:
                            continue
        except Exception as e:
            print(f"Error reading task history: {e}")
        
        return history
    
    def clear_old_history(self, days_to_keep: int = 30):
        """Clear old task history entries"""
        if not self.history_file.exists():
            return
        
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            new_entries = []
            
            with self._file_lock():
                with open(self.history_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                            if entry_time > cutoff_date:
                                new_entries.append(line)
                        except:
                            continue
                
                with open(self.history_file, 'w') as f:
                    f.writelines(new_entries)
        except Exception as e:
            print(f"Error clearing old history: {e}")

