# DataHound Pro Automation Guide

## Overview

DataHound Pro now includes comprehensive automation capabilities that allow you to schedule various data processing tasks. All scheduling is based on **Pacific Standard Time (PST/PDT)**.

## Features

### Scheduling Options

Each page that supports automation offers three types of scheduling:

1. **Interval Scheduling**: Run tasks every X minutes
2. **Daily Scheduling**: Run tasks at a specific time each day
3. **Weekly Scheduling**: Run tasks on specific days within defined time windows

### Supported Operations

#### 1. Download Page
- **Location**: Pages > Download > Automation Tab
- **Features**:
  - Schedule automatic email downloads from Gmail
  - Configure file types, archiving, deduplication, and read marking
  - Set interval, daily, or weekly schedules

#### 2. Prepare Page
- **Location**: Pages > Prepare > Automation Tab
- **Features**:
  - Schedule automatic file preparation
  - Configure which file types to prepare
  - Choose output formats (Parquet/CSV)

#### 3. Integrated Upsert Page
- **Location**: Pages > Integrated Upsert > Automation Tab
- **Features**:
  - Schedule master data updates
  - Configure backup and write modes
  - Automate the upsert process

#### 4. Historical Events Page
- **Location**: Pages > Historical Events > Automation Tab
- **Features**:
  - Schedule individual event type scans
  - Different schedules for different event types
  - Configure scan parameters per event type

#### 5. Customer Extraction Page
- **Location**: Pages > Customer Extraction > Execute Tab
- **Features**:
  - Schedule "Execute All Enabled" operations
  - Simple interval-based scheduling (e.g., every 60 minutes)

#### 6. Create Core Data Page
- **Location**: Pages > Create Core Data > Automation Tab
- **Features**:
  - Schedule core data generation
  - Daily scheduling (e.g., every day at 1:00 AM PST)
  - Configure RFM, demographics, and segmentation options

## How to Use Automation

### Step 1: Configure Your Task

1. Navigate to the desired page (Download, Prepare, etc.)
2. Go to the **Manual** tab first
3. Configure all settings as you want them for automation
4. These settings will be used when the task runs automatically

### Step 2: Set Up Schedule

1. Switch to the **Automation** tab
2. Review the current settings that will be used
3. Choose your schedule type:
   - **Interval**: Select minutes between runs
   - **Daily**: Choose the time of day
   - **Weekly**: Select days and time windows

### Step 3: Create Scheduled Task

1. Click the "📅 Create Scheduled Task" button
2. The task will be added to the scheduler
3. You'll see it appear in the task list below

### Step 4: Manage Tasks

Each scheduled task shows:
- Status (Active ✅, Paused ⏸️, Running 🔄, Error ❌)
- Last run time
- Next scheduled run
- Run count and error count

You can:
- **Pause/Resume**: Temporarily stop a task
- **Delete**: Remove a task permanently
- **Run Now**: Execute immediately for testing

## Running the Scheduler Service

### Option 1: Start from UI

1. Go to any page with automation support
2. Click the **Scheduler Status** tab
3. Click "Start Scheduler" to begin the service

### Option 2: Command Line Service

Run the scheduler as a standalone service:

```bash
# Basic usage
python scheduler_service.py

# With options
python scheduler_service.py --timezone US/Pacific --daemon

# View help
python scheduler_service.py --help
```

Options:
- `--timezone`: Set timezone (default: US/Pacific)
- `--data-dir`: Base directory for data
- `--daemon`: Run in background mode

### Option 3: Windows Service

Create a batch file (`start_scheduler.bat`):
```batch
@echo off
cd /d C:\Users\YourUser\Desktop\datahound_pro
python scheduler_service.py
pause
```

### Option 4: Linux/Mac Service

Create a systemd service or use screen/tmux:
```bash
# Using screen
screen -S datahound-scheduler
python scheduler_service.py
# Press Ctrl+A then D to detach

# Reattach later
screen -r datahound-scheduler
```

## Example Automation Workflows

### Daily Data Processing Pipeline

1. **2:00 AM**: Download new files from Gmail
2. **2:30 AM**: Prepare downloaded files
3. **3:00 AM**: Update master data (upsert)
4. **3:30 AM**: Scan for historical events
5. **4:00 AM**: Extract custom data
6. **4:30 AM**: Create core customer data

### Business Hours Processing

- **Download**: Every 30 minutes, 8 AM - 6 PM, Monday-Friday
- **Prepare**: Every hour, 8 AM - 6 PM, Monday-Friday
- **Events**: Every 2 hours, 9 AM - 5 PM, Monday-Friday

### Weekend Batch Processing

- **Saturday 1:00 AM**: Full download and prepare
- **Saturday 2:00 AM**: Complete master data update
- **Saturday 3:00 AM**: All event scans
- **Sunday 1:00 AM**: Generate core data and reports

## Monitoring and Troubleshooting

### View Scheduler Status

1. Check the **Scheduler Status** tab on any automation page
2. See total active tasks and error counts
3. Start/stop the scheduler service

### Check Task History

Task execution history is stored in:
```
data/scheduler/task_history.jsonl
```

### View Logs

Each task type creates its own logs:
- Download: `data/{company}/logs/`
- Prepare: `data/{company}/logs/pipeline/`
- Events: `data/{company}/logs/event_scan_log.jsonl`
- Extraction: `data/logs/custom_extraction_log.jsonl`

### Common Issues

1. **Scheduler Not Running**
   - Check the Scheduler Status tab
   - Ensure the service is started
   - Check for Python errors in console

2. **Tasks Not Executing**
   - Verify task is Active (not Paused or Error)
   - Check Next Run time is in the past
   - Review task configuration

3. **Authentication Errors**
   - Ensure Gmail tokens are valid
   - Check credentials in company config
   - Re-authenticate if needed

4. **Time Zone Issues**
   - All times are in Pacific Time (PST/PDT)
   - Scheduler handles DST automatically
   - Check system time is correct

## Best Practices

1. **Start Small**: Test with one task before adding many
2. **Stagger Schedules**: Avoid running everything at once
3. **Monitor Initially**: Watch first few runs for issues
4. **Use Appropriate Intervals**: Don't over-schedule tasks
5. **Regular Maintenance**: Clear old logs periodically
6. **Backup First**: Always backup before major automated updates

## Security Considerations

1. **Credentials**: Store securely in `secrets/` directory
2. **Access Control**: Limit who can modify schedules
3. **Data Protection**: Ensure backups are enabled
4. **Error Handling**: Monitor for repeated failures
5. **Resource Usage**: Watch disk space and memory

## Advanced Configuration

### Custom Task Parameters

Each task type supports specific parameters:

```python
# Download Task
task_config = {
    'file_types': ['invoices', 'estimates'],
    'archive_existing': True,
    'dedup_after': True,
    'mark_as_read': True
}

# Prepare Task
task_config = {
    'prepare_types': ['invoices', 'estimates'],
    'write_parquet': True,
    'write_csv': False
}

# Event Scan Task
task_config = {
    'event_type': 'unsold_estimates',
    'scan_all_events': False
}
```

### Programmatic Task Creation

```python
from datahound.scheduler import (
    DataHoundScheduler, ScheduledTask, 
    TaskType, TaskConfiguration, ScheduleType
)
from datetime import time

scheduler = DataHoundScheduler(Path.cwd())

task_config = TaskConfiguration(
    task_type=TaskType.DOWNLOAD,
    company="MyCompany",
    file_types=['invoices']
)

task = ScheduledTask(
    task_id="daily-download",
    name="Daily Invoice Download",
    description="Download invoices every day at 2 AM",
    task_config=task_config,
    schedule_type=ScheduleType.DAILY_TIME,
    daily_time=time(2, 0)  # 2:00 AM
)

scheduler.add_task(task)
```

## Support

For issues or questions:
1. Check this documentation first
2. Review the logs for error messages
3. Ensure all dependencies are installed
4. Verify configuration files are correct

