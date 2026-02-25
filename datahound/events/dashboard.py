"""Dashboard analytics and visualization for event system"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from central_logging.config import event_detection_dir


def load_event_analytics(logs_dir: Path | None = None, company: str | None = None, days_back: int = 30) -> Dict[str, Any]:
    cutoff_date = datetime.now() - timedelta(days=days_back)
    analytics = {
        "scan_summary": {},
        "event_trends": {},
        "performance_metrics": {},
        "error_analysis": {},
        "llm_usage": {}
    }
    if logs_dir is None and company:
        logs_dir = event_detection_dir(company)
    elif logs_dir is None:
        return analytics
    log_files = {
        "scans": logs_dir / "scan.jsonl",
        "detections": logs_dir / "event_detection.jsonl",
        "llm": logs_dir / "llm_analysis.jsonl",
        "errors": logs_dir / "errors.jsonl"
    }
    
    # Analyze scan operations
    if log_files["scans"].exists():
        scan_entries = load_recent_jsonl(log_files["scans"], cutoff_date)
        analytics["scan_summary"] = analyze_scan_operations(scan_entries)
        analytics["performance_metrics"] = analyze_performance_metrics(scan_entries)
    
    # Analyze event detections
    if log_files["detections"].exists():
        detection_entries = load_recent_jsonl(log_files["detections"], cutoff_date)
        analytics["event_trends"] = analyze_event_trends(detection_entries)
    
    # Analyze LLM usage
    if log_files["llm"].exists():
        llm_entries = load_recent_jsonl(log_files["llm"], cutoff_date)
        analytics["llm_usage"] = analyze_llm_usage(llm_entries)
    
    # Analyze errors
    if log_files["errors"].exists():
        error_entries = load_recent_jsonl(log_files["errors"], cutoff_date)
        analytics["error_analysis"] = analyze_errors(error_entries)
    
    return analytics


def load_recent_jsonl(file_path: Path, cutoff_date: datetime) -> List[Dict]:
    """Load JSONL entries newer than cutoff date"""
    if not file_path.exists():
        return []
    
    entries = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entry_date = datetime.fromisoformat(entry.get("ts", "").replace("Z", "+00:00"))
                    if entry_date >= cutoff_date:
                        entries.append(entry)
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        pass
    
    return entries


def analyze_scan_operations(scan_entries: List[Dict]) -> Dict[str, Any]:
    """Analyze scan operation patterns"""
    
    if not scan_entries:
        return {}
    
    # Group by rule name and action
    by_rule = defaultdict(lambda: {"starts": 0, "completes": 0, "errors": 0})
    
    for entry in scan_entries:
        rule_name = entry.get("rule_name", "unknown")
        action = entry.get("action", "")
        
        if action == "scan_start":
            by_rule[rule_name]["starts"] += 1
        elif action == "scan_complete":
            by_rule[rule_name]["completes"] += 1
            by_rule[rule_name]["total_events"] = by_rule[rule_name].get("total_events", 0) + entry.get("total_events", 0)
            by_rule[rule_name]["total_processed"] = by_rule[rule_name].get("total_processed", 0) + entry.get("entities_processed", 0)
    
    return dict(by_rule)


def analyze_event_trends(detection_entries: List[Dict]) -> Dict[str, Any]:
    """Analyze event detection trends"""
    
    if not detection_entries:
        return {}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(detection_entries)
    
    if df.empty:
        return {}
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['ts'], errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    
    trends = {}
    
    # Events by type
    trends["by_type"] = df['event_type'].value_counts().to_dict()
    
    # Events by severity
    trends["by_severity"] = df['severity'].value_counts().to_dict()
    
    # Events by day
    daily = df.groupby('date').size()
    trends["daily_counts"] = daily.to_dict()
    
    # Events by hour
    hourly = df.groupby('hour').size()
    trends["hourly_pattern"] = hourly.to_dict()
    
    # Top entities with most events
    entity_counts = df.groupby(['entity_type', 'entity_id']).size().sort_values(ascending=False).head(10)
    trends["top_entities"] = entity_counts.to_dict()
    
    return trends


def analyze_performance_metrics(scan_entries: List[Dict]) -> Dict[str, Any]:
    """Analyze performance metrics from scan operations"""
    
    complete_scans = [e for e in scan_entries if e.get("action") == "scan_complete"]
    
    if not complete_scans:
        return {}
    
    metrics = {}
    
    # Average duration by rule type
    by_rule = defaultdict(list)
    for entry in complete_scans:
        rule_name = entry.get("rule_name", "unknown")
        duration = entry.get("duration_ms", 0)
        by_rule[rule_name].append(duration)
    
    metrics["avg_duration_by_rule"] = {
        rule: sum(durations) / len(durations) for rule, durations in by_rule.items()
    }
    
    # Processing efficiency
    total_examined = sum(e.get("entities_examined", 0) for e in complete_scans)
    total_processed = sum(e.get("entities_processed", 0) for e in complete_scans)
    total_events = sum(e.get("total_events", 0) for e in complete_scans)
    
    metrics["efficiency"] = {
        "total_examined": total_examined,
        "total_processed": total_processed,
        "total_events": total_events,
        "event_rate": (total_events / max(total_processed, 1)) * 100
    }
    
    return metrics


def analyze_llm_usage(llm_entries: List[Dict]) -> Dict[str, Any]:
    """Analyze LLM API usage patterns"""
    
    if not llm_entries:
        return {}
    
    usage = {}
    
    # Success rate
    total_calls = len(llm_entries)
    successful_calls = len([e for e in llm_entries if e.get("success", False)])
    usage["success_rate"] = (successful_calls / max(total_calls, 1)) * 100
    
    # Average duration
    durations = [e.get("duration_ms", 0) for e in llm_entries if e.get("duration_ms")]
    usage["avg_duration_ms"] = sum(durations) / len(durations) if durations else 0
    
    # Usage by analysis type
    by_type = defaultdict(int)
    for entry in llm_entries:
        analysis_type = entry.get("analysis_type", "unknown")
        by_type[analysis_type] += 1
    
    usage["by_analysis_type"] = dict(by_type)
    
    # Cost estimation (rough)
    # Assume ~1000 tokens per call, $0.14 per 1M tokens for DeepSeek
    estimated_tokens = total_calls * 1000
    estimated_cost = (estimated_tokens / 1_000_000) * 0.14
    usage["estimated_cost_usd"] = round(estimated_cost, 4)
    
    return usage


def analyze_errors(error_entries: List[Dict]) -> Dict[str, Any]:
    """Analyze error patterns"""
    
    if not error_entries:
        return {}
    
    error_analysis = {}
    
    # Errors by rule
    by_rule = defaultdict(int)
    for entry in error_entries:
        rule_name = entry.get("rule_name", "unknown")
        by_rule[rule_name] += 1
    
    error_analysis["by_rule"] = dict(by_rule)
    
    # Common error patterns
    error_patterns = defaultdict(int)
    for entry in error_entries:
        error_msg = entry.get("error", "")
        # Extract error type from message
        if "API" in error_msg.upper():
            error_patterns["API Errors"] += 1
        elif "COLUMN" in error_msg.upper() or "MISSING" in error_msg.upper():
            error_patterns["Data Schema Errors"] += 1
        elif "MEMORY" in error_msg.upper() or "SIZE" in error_msg.upper():
            error_patterns["Memory/Size Errors"] += 1
        else:
            error_patterns["Other Errors"] += 1
    
    error_analysis["by_pattern"] = dict(error_patterns)
    
    return error_analysis


def create_dashboard_charts(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Create chart data for Streamlit dashboard"""
    
    charts = {}
    
    # Event trends chart data
    if "event_trends" in analytics and analytics["event_trends"]:
        trends = analytics["event_trends"]
        
        # Daily events chart
        if "daily_counts" in trends:
            daily_data = []
            for date_str, count in trends["daily_counts"].items():
                daily_data.append({"date": str(date_str), "events": count})
            charts["daily_events"] = daily_data
        
        # Event type distribution
        if "by_type" in trends:
            type_data = []
            for event_type, count in trends["by_type"].items():
                type_data.append({"type": event_type.replace("_", " ").title(), "count": count})
            charts["event_types"] = type_data
        
        # Severity distribution  
        if "by_severity" in trends:
            severity_data = []
            for severity, count in trends["by_severity"].items():
                severity_data.append({"severity": severity.title(), "count": count})
            charts["severity_dist"] = severity_data
    
    # Performance metrics chart
    if "performance_metrics" in analytics and analytics["performance_metrics"]:
        perf = analytics["performance_metrics"]
        
        if "avg_duration_by_rule" in perf:
            duration_data = []
            for rule, avg_ms in perf["avg_duration_by_rule"].items():
                duration_data.append({"rule": rule, "avg_duration_sec": avg_ms / 1000})
            charts["performance"] = duration_data
    
    return charts
