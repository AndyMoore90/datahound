"""Comprehensive logging utilities for event operations"""

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, Optional, List


class EventLogger:
    """Centralized logger for all event operations following JSONL pattern"""
    
    def __init__(self, company: str, data_dir: Path):
        self.company = company
        self.logs_dir = data_dir.parent / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Different log files for different operations
        self.event_scan_log = self.logs_dir / "event_scan_log.jsonl"
        self.llm_analysis_log = self.logs_dir / "llm_analysis_log.jsonl" 
        self.event_detection_log = self.logs_dir / "event_detection_log.jsonl"
        self.event_errors_log = self.logs_dir / "event_errors_log.jsonl"
    
    def _write_log(self, log_file: Path, record: Dict[str, Any]) -> None:
        """Write a single log record to specified file"""
        record["ts"] = datetime.now(UTC).isoformat()
        record["company"] = self.company
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    
    def log_scan_start(self, rule_name: str, rule_type: str, config: Dict[str, Any]) -> None:
        """Log the start of an event scan"""
        self._write_log(self.event_scan_log, {
            "action": "scan_start",
            "rule_name": rule_name,
            "rule_type": rule_type,
            "config": config,
            "status": "started"
        })
    
    def log_scan_complete(self, rule_name: str, total_events: int, 
                         entities_examined: int, entities_processed: int,
                         duration_ms: int, limit_applied: bool) -> None:
        """Log the completion of an event scan"""
        self._write_log(self.event_scan_log, {
            "action": "scan_complete",
            "rule_name": rule_name,
            "total_events": total_events,
            "entities_examined": entities_examined,
            "entities_processed": entities_processed,
            "duration_ms": duration_ms,
            "limit_applied": limit_applied,
            "status": "completed"
        })
    
    def log_scan_error(self, rule_name: str, error: str, context: Dict[str, Any] = None) -> None:
        """Log an error during event scanning"""
        record = {
            "action": "scan_error",
            "rule_name": rule_name,
            "error": error,
            "status": "error"
        }
        if context:
            record["context"] = context
        
        self._write_log(self.event_errors_log, record)
    
    def log_llm_analysis(self, location_id: str, analysis_type: str, 
                        input_data: Dict[str, Any], result: Dict[str, Any],
                        duration_ms: int, success: bool) -> None:
        """Log LLM analysis operations"""
        self._write_log(self.llm_analysis_log, {
            "action": "llm_analysis",
            "entity_id": location_id,
            "analysis_type": analysis_type,
            "input_summary": {
                "job_count": input_data.get("job_count", 0),
                "permit_count": input_data.get("permit_count", 0),
                "text_length": len(str(input_data.get("text", "")))
            },
            "result": result,
            "duration_ms": duration_ms,
            "success": success,
            "status": "completed" if success else "failed"
        })
    
    def log_event_detected(self, event_type: str, entity_type: str, entity_id: str,
                          severity: str, details: Dict[str, Any], rule_name: str) -> None:
        """Log when an event is detected"""
        self._write_log(self.event_detection_log, {
            "action": "event_detected", 
            "event_type": event_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "severity": severity,
            "details": details,
            "rule_name": rule_name,
            "status": "detected"
        })
    
    def log_processing_stats(self, rule_name: str, stage: str, stats: Dict[str, Any]) -> None:
        """Log processing statistics at various stages"""
        self._write_log(self.event_scan_log, {
            "action": "processing_stats",
            "rule_name": rule_name,
            "stage": stage,
            "stats": stats,
            "status": "info"
        })
    
    def log_table_load(self, table_name: str, row_count: int, columns: List[str], 
                      duration_ms: int, success: bool) -> None:
        """Log table loading operations"""
        self._write_log(self.event_scan_log, {
            "action": "table_load",
            "table_name": table_name,
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns[:10],  # First 10 columns only
            "duration_ms": duration_ms,
            "success": success,
            "status": "completed" if success else "failed"
        })
