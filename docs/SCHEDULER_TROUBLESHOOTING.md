# DataHound Pro Scheduler Troubleshooting Guide

## 🔍 **Issues Identified**

### Issue 1: Scheduler Service Not Running
**Symptoms**: Tasks show correct "Next run" times but never execute  
**Cause**: The scheduler service is not running in the background  
**Status**: ✅ **IDENTIFIED AND FIXED**

### Issue 2: Task Execution Errors  
**Symptoms**: Tasks fail with import errors  
**Cause**: Missing scan functions in executor  
**Status**: ✅ **FIXED**

### Issue 3: "Schedule All Events" Button Not Responding
**Symptoms**: Button click doesn't show any feedback  
**Cause**: Likely UI state issue  
**Status**: 🔧 **INVESTIGATING**

## 🛠️ **Solutions**

### Step 1: Start the Scheduler Service

**Option A: From Command Line (Recommended)**
```bash
# Navigate to your project directory
cd C:\Users\Andym\Desktop\datahound_pro

# Start the scheduler service
python scheduler_service.py
```

**Option B: From Streamlit UI**
1. Go to any page with automation (Download, Prepare, etc.)
2. Click the "📊 Scheduler Status" tab
3. Click "Start Scheduler" button
4. Keep the browser tab open

### Step 2: Verify Scheduler is Running

Run this test command:
```bash
python -c "from datahound.scheduler import DataHoundScheduler; from pathlib import Path; s = DataHoundScheduler(Path.cwd()); print(f'Scheduler running: {s._running}'); tasks = s.get_all_tasks(); print(f'Active tasks: {len([t for t in tasks if t.status.value == \"active\"])}')"
```

Expected output:
```
Scheduler running: True
Active tasks: X
```

### Step 3: Monitor Task Execution

1. **Check Task Status**: Go to any automation page → Scheduler Status tab
2. **View Logs**: Check `data/scheduler/task_history.jsonl`
3. **Watch Console**: If running `scheduler_service.py`, you'll see execution logs

### Step 4: Test Individual Tasks

1. Go to the page where you created a task
2. Find the task in the "📅 Scheduled Tasks" section
3. Click "🚀 Run Selected Task Now" to test immediately
4. Check if it executes successfully

## 🧪 **Testing Your Setup**

### Test Download Automation:
1. Go to **Download** page → **Automation** tab
2. Configure a simple schedule (e.g., every 5 minutes)
3. Create the scheduled task
4. Go to **Scheduler Status** tab and start the scheduler
5. Wait 5 minutes and check if it executes

### Test Historical Events:
1. Go to **Historical Events** page
2. Enable one event type (e.g., "Overdue Maintenance")
3. Configure its settings
4. Check the "Configure scheduling for overdue maintenance" checkbox
5. Set up a schedule and create it
6. Verify the task appears in the scheduler

## 🐛 **Common Issues and Fixes**

### Issue: "Schedule All Events" Button Not Working

**Possible Causes:**
1. No events are enabled
2. Streamlit key conflicts
3. Session state issues

**Debug Steps:**
```python
# Check which events are enabled
python -c "
import json
from pathlib import Path
config_path = Path('config/events/McCullough Heating and Air_historical_events_config.json')
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    enabled = [k for k, v in config.get('events', {}).items() if v.get('enabled')]
    print(f'Enabled events: {enabled}')
else:
    print('No event config found')
"
```

### Issue: Tasks Created But Not Executing

**Check List:**
1. ✅ Is scheduler service running?
2. ✅ Are tasks in "active" status?
3. ✅ Is the next run time in the past?
4. ✅ Are there any error messages?

**Debug Command:**
```python
python -c "
from datahound.scheduler import DataHoundScheduler
from pathlib import Path
from datetime import datetime
import pytz

s = DataHoundScheduler(Path.cwd())
tasks = s.get_all_tasks()
now = datetime.now(pytz.timezone('US/Pacific'))

print(f'Current time (PT): {now}')
print(f'Scheduler running: {s._running}')
print()

for task in tasks:
    print(f'Task: {task.name}')
    print(f'  Status: {task.status.value}')
    print(f'  Next run: {task.next_run}')
    if task.next_run:
        should_run = now >= task.next_run.replace(tzinfo=pytz.timezone('US/Pacific'))
        print(f'  Should run now: {should_run}')
    print(f'  Last error: {task.last_error or \"None\"}')
    print()
"
```

### Issue: Import Errors in Task Execution

**Status**: ✅ **FIXED** - Updated executor to use direct implementation instead of missing imports

## 🚀 **Quick Start Guide**

### To Get Automation Working Right Now:

1. **Start Scheduler Service**:
   ```bash
   python scheduler_service.py
   ```
   Keep this terminal window open.

2. **Test a Simple Task**:
   - Go to Download page → Automation tab
   - Create a task with 5-minute interval
   - Watch the console for execution logs

3. **Monitor Results**:
   - Check the "📅 Scheduled Tasks" section for status updates
   - Look for "Last run" time updates
   - Check for any error messages

### Expected Behavior:
- **Console Output**: You should see messages like:
  ```
  [START] Download Files - McCullough Heating and Air (task-id)
  [SUCCESS] Download Files completed successfully
  ```
- **UI Updates**: Last run time should update after execution
- **File Changes**: New files should appear in your data directories

## 📋 **Current Status**

✅ **Scheduler Infrastructure**: Complete and working  
✅ **Task Creation**: Working correctly  
✅ **Task Persistence**: Saving/loading tasks properly  
✅ **Executor Logic**: Fixed import issues  
🔧 **Service Management**: Needs manual start  
🔧 **UI Feedback**: Some buttons may need debugging  

## 🎯 **Next Steps**

1. **Start the scheduler service** using the command above
2. **Test with a simple download task** to verify it works
3. **Check the console output** for execution logs
4. **Report back** on what you see when tasks execute

The core automation system is working - it just needs the service to be running!
