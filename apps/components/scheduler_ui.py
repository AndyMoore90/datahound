import streamlit as st
import uuid
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from datahound.scheduler import (
    DataHoundScheduler, ScheduledTask, TaskType, TaskStatus, 
    ScheduleType, TaskConfiguration, DayOfWeek, TimeWindow, WeeklySchedule
)


def get_scheduler_instance() -> DataHoundScheduler:
    if 'scheduler' not in st.session_state:
        base_dir = Path.cwd()
        st.session_state.scheduler = DataHoundScheduler(base_dir, timezone="US/Pacific")
    return st.session_state.scheduler


def render_schedule_config(
    key_prefix: str,
    schedule_type: Optional[ScheduleType] = None,
    default_interval_minutes: int = 60,
    default_daily_time: Optional[time] = None,
    default_days: Optional[List[DayOfWeek]] = None,
    default_time_window: Optional[Tuple[time, time]] = None,
) -> Dict[str, Any]:
    schedule_types = {
        "Interval (Every X minutes)": ScheduleType.INTERVAL,
        "Daily (Specific time)": ScheduleType.DAILY_TIME,
        "Weekly (Specific days and times)": ScheduleType.WEEKLY_SCHEDULE
    }
    
    selected_type_label = st.selectbox(
        "Schedule Type",
        options=list(schedule_types.keys()),
        key=f"{key_prefix}_schedule_type"
    )
    selected_type = schedule_types[selected_type_label]
    
    config = {"schedule_type": selected_type}
    
    if selected_type == ScheduleType.INTERVAL:
        # Interval configuration
        config["interval_minutes"] = st.number_input(
            "Run every (minutes)",
            min_value=5,
            max_value=1440,
            value=default_interval_minutes,
            step=5,
            key=f"{key_prefix}_interval"
        )
        
    elif selected_type == ScheduleType.DAILY_TIME:
        # Daily time configuration
        default_time = default_daily_time or time(1, 0)  # Default 1:00 AM
        
        col1, col2 = st.columns(2)
        with col1:
            hour = st.selectbox(
                "Hour",
                options=list(range(24)),
                format_func=lambda x: f"{x:02d}",
                index=default_time.hour,
                key=f"{key_prefix}_hour"
            )
        with col2:
            minute = st.selectbox(
                "Minute",
                options=[0, 15, 30, 45],
                format_func=lambda x: f"{x:02d}",
                index=[0, 15, 30, 45].index(default_time.minute if default_time.minute in [0, 15, 30, 45] else 0),
                key=f"{key_prefix}_minute"
            )
        
        config["daily_time"] = time(hour, minute)
        st.info(f"🕐 Will run daily at {hour:02d}:{minute:02d} Pacific Time")
        
    elif selected_type == ScheduleType.WEEKLY_SCHEDULE:
        # Weekly schedule configuration
        st.markdown("**Select Days to Run**")
        
        day_cols = st.columns(7)
        days_map = [
            ("Mon", DayOfWeek.MONDAY),
            ("Tue", DayOfWeek.TUESDAY),
            ("Wed", DayOfWeek.WEDNESDAY),
            ("Thu", DayOfWeek.THURSDAY),
            ("Fri", DayOfWeek.FRIDAY),
            ("Sat", DayOfWeek.SATURDAY),
            ("Sun", DayOfWeek.SUNDAY)
        ]
        
        selected_days = []
        for i, (day_name, day_enum) in enumerate(days_map):
            with day_cols[i]:
                if st.checkbox(
                    day_name,
                    value=(default_days and day_enum in default_days) or False,
                    key=f"{key_prefix}_day_{day_name}"
                ):
                    selected_days.append(day_enum)
        
        if not selected_days:
            st.warning("⚠️ Please select at least one day")
        
        st.markdown("**Time Window**")
        col1, col2, col3 = st.columns(3)
        
        default_start = default_time_window[0] if default_time_window else time(9, 0)
        default_end = default_time_window[1] if default_time_window else time(17, 0)
        
        with col1:
            start_hour = st.selectbox(
                "Start Hour",
                options=list(range(24)),
                format_func=lambda x: f"{x:02d}:00",
                index=default_start.hour,
                key=f"{key_prefix}_start_hour"
            )
        
        with col2:
            end_hour = st.selectbox(
                "End Hour",
                options=list(range(24)),
                format_func=lambda x: f"{x:02d}:00",
                index=default_end.hour,
                key=f"{key_prefix}_end_hour"
            )
        
        with col3:
            interval = st.number_input(
                "Interval (minutes)",
                min_value=5,
                max_value=480,
                value=default_interval_minutes,
                step=5,
                key=f"{key_prefix}_weekly_interval"
            )
        
        config["weekly_schedule"] = WeeklySchedule(
            days=selected_days,
            time_window=TimeWindow(
                start_time=time(start_hour, 0),
                end_time=time(end_hour, 0)
            ),
            interval_minutes=interval
        )
        
        if selected_days:
            st.info(f"🕐 Will run every {interval} minutes between {start_hour:02d}:00-{end_hour:02d}:00 PT on {', '.join([d.value.capitalize() for d in selected_days])}")
    
    return config


def render_task_manager(
    task_type: TaskType,
    company: str,
    task_name: str,
    task_description: str,
    key_context: str = "",
) -> None:
    scheduler = get_scheduler_instance()
    ctx = f"_{key_context}" if key_context else ""
    key_pfx = f"{task_type.value}_{company}{ctx}"

    all_tasks = scheduler.get_all_tasks()
    relevant_tasks = [
        task for task in all_tasks
        if task.task_config.task_type == task_type
        and task.task_config.company == company
    ]

    if relevant_tasks:
        for task in relevant_tasks:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

            with col1:
                status_icon = {
                    TaskStatus.ACTIVE: "✅",
                    TaskStatus.PAUSED: "⏸️",
                    TaskStatus.RUNNING: "🔄",
                    TaskStatus.ERROR: "❌",
                    TaskStatus.DISABLED: "🚫",
                }.get(task.status, "❓")
                st.write(f"{status_icon} **{task.name}**")
                st.caption(task.description)

            with col2:
                if task.last_run:
                    st.caption(f"Last: {task.last_run.strftime('%m/%d %H:%M')}")
                else:
                    st.caption("Never run")
                if task.next_run:
                    st.caption(f"Next: {task.next_run.strftime('%m/%d %H:%M')}")

            with col3:
                st.caption(f"Runs: {task.run_count}")
                if task.consecutive_errors > 0:
                    st.error(f"Errors: {task.consecutive_errors}")

            with col4:
                a1, a2 = st.columns(2)
                with a1:
                    if task.status == TaskStatus.ACTIVE:
                        if st.button("⏸️", key=f"pause_{key_pfx}_{task.task_id}", help="Pause"):
                            scheduler.pause_task(task.task_id)
                            st.rerun()
                    elif task.status == TaskStatus.PAUSED:
                        if st.button("▶️", key=f"resume_{key_pfx}_{task.task_id}", help="Resume"):
                            scheduler.resume_task(task.task_id)
                            st.rerun()
                with a2:
                    if st.button("🗑️", key=f"delete_{key_pfx}_{task.task_id}", help="Delete"):
                        scheduler.delete_task(task.task_id)
                        st.rerun()

        if st.button("Run Now", key=f"run_now_{key_pfx}"):
            task = relevant_tasks[0]
            with st.spinner(f"Running {task.name}..."):
                result = scheduler.run_task_now(task.task_id)
                if result["success"]:
                    st.success("Task executed successfully!")
                else:
                    st.error(f"Task failed: {result.get('error', 'Unknown error')}")
    else:
        st.info("No scheduled tasks for this operation.")


def create_scheduled_task(
    task_type: TaskType,
    company: str,
    task_name: str,
    task_description: str,
    schedule_config: Dict[str, Any],
    task_config_overrides: Optional[Dict[str, Any]] = None,
) -> bool:
    try:
        scheduler = get_scheduler_instance()
        
        # Create task configuration
        task_config = TaskConfiguration(
            task_type=task_type,
            company=company,
            settings=task_config_overrides or {}
        )
        
        # Apply overrides
        if task_config_overrides:
            for key, value in task_config_overrides.items():
                if hasattr(task_config, key):
                    setattr(task_config, key, value)
        
        # Create scheduled task
        task = ScheduledTask(
            task_id=str(uuid.uuid4()),
            name=task_name,
            description=task_description,
            task_config=task_config,
            schedule_type=schedule_config["schedule_type"],
            status=TaskStatus.ACTIVE
        )
        
        # Apply schedule configuration
        if task.schedule_type == ScheduleType.INTERVAL:
            task.interval_minutes = schedule_config["interval_minutes"]
        elif task.schedule_type == ScheduleType.DAILY_TIME:
            task.daily_time = schedule_config["daily_time"]
        elif task.schedule_type == ScheduleType.WEEKLY_SCHEDULE:
            task.weekly_schedule = schedule_config["weekly_schedule"]
        
        # Add the task
        success = scheduler.add_task(task)
        
        if success:
            print(f"✅ Successfully created task: {task_name}")
        else:
            print(f"❌ Failed to create task: {task_name}")
            
        return success
        
    except Exception as e:
        print(f"❌ Error creating scheduled task: {e}")
        import traceback
        traceback.print_exc()
        return False


def render_scheduler_status(key_context: str = "") -> None:
    scheduler = get_scheduler_instance()
    sfx = f"_{key_context}" if key_context else ""

    col1, col2, col3 = st.columns(3)

    with col1:
        if scheduler._running:
            st.success("Scheduler Running")
            if st.button("Stop Scheduler", key=f"stop_sched{sfx}"):
                scheduler.stop()
                st.rerun()
        else:
            st.warning("Scheduler Stopped")
            if st.button("Start Scheduler", key=f"start_sched{sfx}"):
                scheduler.start(daemon=True)
                st.success("Scheduler started!")
                st.rerun()

    with col2:
        all_tasks = scheduler.get_all_tasks()
        active_tasks = [t for t in all_tasks if t.status == TaskStatus.ACTIVE]
        st.metric("Active Tasks", len(active_tasks))

    with col3:
        error_tasks = [t for t in all_tasks if t.status == TaskStatus.ERROR]
        if error_tasks:
            st.error(f"Tasks with Errors: {len(error_tasks)}")
        else:
            st.success("No Errors")

