#!/usr/bin/env python
"""Enhanced DataHound Pro Automation Startup Script"""

import sys
import signal
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datahound.env import load_env_fallback
from datahound.scheduler import DataHoundScheduler

# Ensure local .env is loaded for non-exported shell environments
load_env_fallback(ROOT)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Received shutdown signal, stopping automation...")
    if 'scheduler' in globals():
        scheduler.stop()
    sys.exit(0)

def show_startup_banner():
    """Show startup banner with system info"""
    print("🤖 DATAHOUND PRO AUTOMATION SERVICE")
    print("=" * 60)
    print("🚀 Starting automated data processing pipeline...")
    print("📍 All times shown in Pacific Time (PT)")
    print("⌨️  Press Ctrl+C to stop")
    print("=" * 60)

def main():
    """Enhanced main function with detailed monitoring"""
    
    parser = argparse.ArgumentParser(description="DataHound Pro Enhanced Automation Service")
    parser.add_argument('--timezone', default='US/Pacific', help='Timezone for scheduling')
    parser.add_argument('--data-dir', type=Path, default=Path.cwd(), help='Base directory for data')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon in background')
    parser.add_argument('--status-only', action='store_true', help='Show status and exit')
    parser.add_argument('--quiet', action='store_true', help='Minimal output mode')
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    global scheduler
    scheduler = DataHoundScheduler(args.data_dir, args.timezone)
    
    if args.status_only:
        tasks = scheduler.get_all_tasks()
        print(f"Tasks: {len(tasks)}")
        for t in tasks:
            print(f"  {t.name} [{t.status.value}] runs={t.run_count}")
        return
    
    if not args.quiet:
        show_startup_banner()
        
        # Show initial task overview
        tasks = scheduler.get_all_tasks()
        if tasks:
            print(f"\n📋 Found {len(tasks)} scheduled tasks:")
            task_types = {}
            for task in tasks:
                task_type = task.task_config.task_type.value.replace('_', ' ').title()
                if task_type not in task_types:
                    task_types[task_type] = 0
                task_types[task_type] += 1
            
            for task_type, count in task_types.items():
                print(f"   📊 {task_type}: {count} tasks")
        else:
            print("\n⚠️  No scheduled tasks found!")
            print("   Create tasks using the DataHound Pro UI")
        
        print("\n🔄 Starting scheduler service...")
    
    # Enhanced callbacks for detailed logging
    def on_task_start(task):
        now = datetime.now(scheduler.timezone)
        task_type = task.task_config.task_type.value.replace('_', ' ').title()
        
        print(f"\n🚀 [{now.strftime('%H:%M:%S')}] EXECUTING: {task_type}")
        print(f"   📋 {task.name}")
        print(f"   🏢 {task.task_config.company}")
        
        # Show task-specific details
        if task.task_config.task_type.value == 'historical_event_scan':
            if hasattr(task.task_config, 'event_type') and task.task_config.event_type:
                print(f"   🎯 Event: {task.task_config.event_type.replace('_', ' ').title()}")
        elif task.task_config.task_type.value == 'download':
            if hasattr(task.task_config, 'file_types') and task.task_config.file_types:
                print(f"   📁 Files: {', '.join(task.task_config.file_types[:3])}{'...' if len(task.task_config.file_types) > 3 else ''}")
        elif task.task_config.task_type.value == 'custom_extraction':
            print(f"   🔄 Executing all enabled extractions")
        elif task.task_config.task_type.value == 'create_core_data':
            print(f"   👥 Building customer profiles with RFM analysis")
    
    def on_task_complete(task):
        now = datetime.now(scheduler.timezone)
        task_type = task.task_config.task_type.value.replace('_', ' ').title()
        
        print(f"✅ [{now.strftime('%H:%M:%S')}] COMPLETED: {task_type}")
        print(f"   📋 {task.name}")
        print(f"   📊 Run #{task.run_count}")
        
        # Calculate and show next run
        if task.next_run:
            next_run = task.next_run.replace(tzinfo=scheduler.timezone) if task.next_run.tzinfo is None else task.next_run
            time_until = next_run - now
            
            if time_until.total_seconds() > 0:
                if time_until.days > 0:
                    next_str = f"{time_until.days}d {time_until.seconds//3600}h"
                elif time_until.seconds > 3600:
                    next_str = f"{time_until.seconds//3600}h {(time_until.seconds%3600)//60}m"
                else:
                    next_str = f"{time_until.seconds//60}m"
                print(f"   ⏰ Next execution in {next_str} at {next_run.strftime('%H:%M')}")
    
    def on_task_error(task, error):
        now = datetime.now(scheduler.timezone)
        task_type = task.task_config.task_type.value.replace('_', ' ').title()
        
        print(f"❌ [{now.strftime('%H:%M:%S')}] FAILED: {task_type}")
        print(f"   📋 {task.name}")
        print(f"   🔴 Error: {error}")
        print(f"   🔢 Consecutive Failures: {task.consecutive_errors}")
        
        if task.consecutive_errors >= 5:
            print(f"   🚨 Task automatically disabled after 5 failures")
    
    # Set callbacks
    scheduler.on_task_start = on_task_start
    scheduler.on_task_complete = on_task_complete
    scheduler.on_task_error = on_task_error
    
    # Start the scheduler
    scheduler.start(daemon=args.daemon)
    
    if not args.quiet:
        print("✅ Automation service started successfully!")
        
        # Show initial task schedule
        tasks = scheduler.get_all_tasks()
        now = datetime.now(scheduler.timezone)
        active_tasks = [t for t in tasks if t.status.value == 'active']
        
        if active_tasks:
            print("\n" + "=" * 60)
            print("📅 SCHEDULED TASKS - NEXT RUN TIMES")
            print("=" * 60)
            
            # Sort by next run time
            sorted_tasks = []
            for task in active_tasks:
                if task.next_run:
                    next_run = task.next_run.replace(tzinfo=scheduler.timezone) if task.next_run.tzinfo is None else task.next_run
                    time_until = next_run - now
                    sorted_tasks.append((task, next_run, time_until))
                else:
                    # Task doesn't have next_run set yet - will be calculated by scheduler
                    # Use a far future date for sorting purposes
                    next_run = now + timedelta(days=365)
                    time_until = timedelta(days=365)
                    sorted_tasks.append((task, next_run, time_until))
            
            sorted_tasks.sort(key=lambda x: x[1])
            
            for task, next_run, time_until in sorted_tasks[:10]:  # Show next 10
                task_type = task.task_config.task_type.value.replace('_', ' ').title()
                
                if time_until.total_seconds() <= 0:
                    status = "🟢 DUE NOW"
                elif time_until.total_seconds() <= 300:
                    status = "🟡 SOON"
                else:
                    status = "⚪ SCHEDULED"
                
                if time_until.total_seconds() <= 0:
                    time_str = "NOW"
                elif time_until.days > 0:
                    time_str = f"{time_until.days}d {time_until.seconds//3600}h"
                elif time_until.seconds >= 3600:
                    hours = time_until.seconds // 3600
                    mins = (time_until.seconds % 3600) // 60
                    time_str = f"{hours}h {mins}m"
                else:
                    mins = time_until.seconds // 60
                    secs = time_until.seconds % 60
                    time_str = f"{mins}m {secs}s"
                
                print(f"{status} {task_type:25} → {next_run.strftime('%H:%M:%S')} ({time_str})")
                print(f"      {task.name[:50]}")
            
            if len(sorted_tasks) > 10:
                print(f"\n   ... and {len(sorted_tasks) - 10} more tasks")
        
        print("\n📊 Live monitoring active (updates every 60 seconds)")
        print("=" * 60)
    
    # Enhanced monitoring loop
    try:
        update_counter = 0
        while True:
            time.sleep(60)  # Update every minute
            update_counter += 1
            
            tasks = scheduler.get_all_tasks()
            now = datetime.now(scheduler.timezone)
            
            # Show detailed status every minute
            if not args.quiet:
                # Every 10 minutes, show full status
                if update_counter % 10 == 0:
                    print("\n" + "="*60)
                    print("📊 AUTOMATION HEALTH CHECK")
                    print("="*60)
                    print(f"⏱️  Uptime: {update_counter} minutes")
                    print(f"📋 Total Tasks: {len(tasks)}")
                    
                    active_count = len([t for t in tasks if t.status.value == 'active'])
                    running_count = len([t for t in tasks if t.status.value == 'running'])
                    error_count = len([t for t in tasks if t.status.value == 'error'])
                    
                    print(f"✅ Active: {active_count} | 🏃 Running: {running_count} | ❌ Errors: {error_count}")
                    print("="*60)
                
                # Show immediate upcoming tasks (next 5 minutes)
                upcoming_soon = []
                for task in tasks:
                    if task.status.value == 'active' and task.next_run:
                        next_run = task.next_run.replace(tzinfo=scheduler.timezone) if task.next_run.tzinfo is None else task.next_run
                        time_until = next_run - now
                        if 0 <= time_until.total_seconds() <= 300:  # Next 5 minutes
                            upcoming_soon.append((task, time_until))
                
                if upcoming_soon:
                    print(f"\n🕐 {now.strftime('%H:%M:%S')} PT | 🔥 {len(upcoming_soon)} task(s) executing soon:")
                    for task, time_until in sorted(upcoming_soon, key=lambda x: x[1]):
                        mins = int(time_until.total_seconds() // 60)
                        secs = int(time_until.total_seconds() % 60)
                        task_type = task.task_config.task_type.value.replace('_', ' ').title()
                        if time_until.total_seconds() <= 0:
                            print(f"   ⚡ NOW - {task_type}: {task.name[:45]}")
                        else:
                            print(f"   ⏰ {mins}m {secs}s - {task_type}: {task.name[:45]}")
                else:
                    # Show next upcoming task
                    active_tasks = [t for t in tasks if t.status.value == 'active' and t.next_run]
                    if active_tasks:
                        next_task = min(active_tasks, key=lambda x: x.next_run.replace(tzinfo=scheduler.timezone) if x.next_run.tzinfo is None else x.next_run)
                        if next_task.next_run:
                            next_run = next_task.next_run.replace(tzinfo=scheduler.timezone) if next_task.next_run.tzinfo is None else next_task.next_run
                            time_until = next_run - now
                            
                            if time_until.total_seconds() > 0:
                                if time_until.days > 0:
                                    next_str = f"{time_until.days}d {time_until.seconds//3600}h"
                                elif time_until.seconds >= 3600:
                                    hours = time_until.seconds // 3600
                                    mins = (time_until.seconds % 3600) // 60
                                    next_str = f"{hours}h {mins}m"
                                else:
                                    mins = time_until.seconds // 60
                                    secs = time_until.seconds % 60
                                    next_str = f"{mins}m {secs}s"
                                
                                task_type = next_task.task_config.task_type.value.replace('_', ' ').title()
                                print(f"\n🕐 {now.strftime('%H:%M:%S')} PT | ⏰ Next: {task_type} in {next_str} ({next_run.strftime('%H:%M:%S')})")
                                print(f"   📋 {next_task.name[:55]}")
                    else:
                        print(f"\n🕐 {now.strftime('%H:%M:%S')} PT | ⏸️  Waiting for scheduled tasks...")
                
                # Show errors if any
                error_tasks = [t for t in tasks if t.status.value == 'error']
                if error_tasks:
                    print(f"\n❌ {len(error_tasks)} task(s) have errors:")
                    for task in error_tasks[:3]:
                        print(f"   🔴 {task.name[:50]}: {task.last_error[:50] if task.last_error else 'Unknown error'}")
                
                # Show running tasks
                running_tasks = [t for t in tasks if t.status.value == 'running']
                if running_tasks:
                    print(f"\n🏃 {len(running_tasks)} task(s) currently running:")
                    for task in running_tasks:
                        task_type = task.task_config.task_type.value.replace('_', ' ').title()
                        print(f"   ⚙️  {task_type}: {task.name[:45]}")
            
    except KeyboardInterrupt:
        if not args.quiet:
            print("\n🛑 Shutdown requested...")
    finally:
        scheduler.stop()
        if not args.quiet:
            print("✅ DataHound Pro Automation Service stopped")

if __name__ == "__main__":
    main()
