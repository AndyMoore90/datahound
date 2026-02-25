"""
Simplified Integrated Upsert Engine - Master Data Updates Only
"""

import json
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import pandas as pd

from .engine import upsert_type as _core_upsert_type
from .types import UpsertResult, AuditChange


class IntegratedUpsertResult:
    """Result from simplified upsert operation - master data updates only"""
    
    def __init__(self):
        # Core upsert results by type
        self.upsert_results: Dict[str, UpsertResult] = {}
        
        # Change tracking
        self.total_changes: int = 0
        self.total_new_records: int = 0
        self.total_updated_records: int = 0
        self.affected_customer_ids: Set[str] = set()
        
        # Performance metrics
        self.total_duration_seconds: float = 0
        self.phase_durations: Dict[str, float] = {}
        
        # Errors and warnings
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and display"""
        return {
            "total_changes": self.total_changes,
            "total_new_records": self.total_new_records,
            "total_updated_records": self.total_updated_records,
            "affected_customers": len(self.affected_customer_ids),
            "total_duration_seconds": self.total_duration_seconds,
            "phase_durations": self.phase_durations,
            "errors": self.errors,
            "warnings": self.warnings,
            "upsert_summary": {
                ftype: {
                    "examined": result.examined_rows,
                    "updated": result.updated_rows,
                    "new": result.new_rows,
                    "changes": len(result.audit_changes)
                } for ftype, result in self.upsert_results.items()
            }
        }


class IntegratedUpsertEngine:
    """
    Simplified upsert engine that handles:
    1. Upsert prepared data to master files
    2. Track all changes in detailed logs
    """
    
    def __init__(self, company: str, data_dir: Path, tables_dir: Path):
        self.company = company
        self.data_dir = data_dir
        self.tables_dir = tables_dir
        self.parquet_dir = tables_dir
        from central_logging.config import pipeline_dir
        company = data_dir.name
        self.log_file = pipeline_dir(company) / "integrated_upsert.jsonl"
    
    def process_prepared_files(self, 
                              prepared_files: Dict[str, Path],
                              config: Dict[str, Any],
                              progress_callback: Optional[callable] = None) -> IntegratedUpsertResult:
        """
        Process prepared files to update master data and track changes
        
        Args:
            prepared_files: Dict of {file_type: prepared_file_path}
            config: Configuration dict with upsert settings
            progress_callback: Optional callback for progress updates
        """
        
        start_time = time.perf_counter()
        result = IntegratedUpsertResult()
        
        self._log_event("info", "Starting master data update", {
            "files_to_process": list(prepared_files.keys()),
            "total_files": len(prepared_files)
        })
        
        try:
            # Phase 1: Upsert all prepared data to master files
            phase1_start = time.perf_counter()
            if progress_callback:
                progress_callback("Phase 1: Upserting data to master files...", 0.1)
            
            self._process_upserts(prepared_files, config, result, progress_callback)
            result.phase_durations["upsert"] = time.perf_counter() - phase1_start
            
            # Phase 2: Process changes and create logs
            phase2_start = time.perf_counter()
            if progress_callback:
                progress_callback("Phase 2: Processing changes and creating logs...", 0.7)
            
            self._process_changes(result, progress_callback)
            result.phase_durations["change_processing"] = time.perf_counter() - phase2_start
            
            # Complete
            result.total_duration_seconds = time.perf_counter() - start_time
            
            if progress_callback:
                progress_callback("Master files updated - ready for Historical Events processing", 1.0)
            
            self._log_event("info", "Master data update completed", result.to_dict())
            
        except Exception as e:
            result.errors.append(f"Master data update failed: {str(e)}")
            self._log_event("error", f"Master data update failed: {e}")
            raise
        
        return result
    
    def _process_upserts(self, prepared_files: Dict[str, Path], config: Dict[str, Any], 
                        result: IntegratedUpsertResult, progress_callback: Optional[callable]):
        """Phase 1: Process all upsert operations - with smart optimization to skip files with no changes"""
        
        total_files = len(prepared_files)
        files_processed = 0
        files_skipped = 0
        
        for i, (ftype, prepared_path) in enumerate(prepared_files.items()):
            try:
                self._log_event("info", f"Checking {ftype} for changes", {
                    "file_type": ftype,
                    "prepared_file": str(prepared_path)
                })
                
                # Get configuration for this file type
                id_col = config.get('id_columns', {}).get(ftype)
                master_filename = config.get('master_filenames', {}).get(ftype)
                
                if not id_col:
                    id_col = f"{ftype.rstrip('s').capitalize()} ID"
                
                # Pre-check if file has any changes (optimization)
                has_changes = self._check_file_has_changes(prepared_path, ftype, id_col, master_filename)
                
                if not has_changes:
                    files_skipped += 1
                    self._log_event("info", f"Skipping {ftype} - no changes detected", {
                        "file_type": ftype,
                        "optimization": "no_changes_skip"
                    })
                    
                    if progress_callback:
                        phase1_progress = 0.1 + (0.3 * (i + 1) / total_files)
                        progress_callback(f"Skipped {ftype}: No changes", phase1_progress)
                    continue
                
                files_processed += 1
                self._log_event("info", f"Processing upsert for {ftype}", {
                    "file_type": ftype,
                    "prepared_file": str(prepared_path),
                    "changes_detected": True
                })
                
                # Run core upsert
                upsert_result = _core_upsert_type(
                    company=self.company,
                    data_dir=self.data_dir,
                    tables_dir=self.tables_dir,
                    ftype=ftype,
                    prepared_path=prepared_path,
                    id_col=id_col,
                    event_rules=config.get('event_rules'),
                    master_filename=master_filename,
                    backup=config.get('backup', True),
                    backup_dir=config.get('backup_dir'),
                    write_mode=config.get('write_mode', 'inplace'),
                    dry_run=config.get('dry_run', False),
                    maintain_store=config.get('maintain_store', False),
                    prefer_parquet=config.get('prefer_parquet', True)
                )
                
                # Store result and aggregate metrics
                result.upsert_results[ftype] = upsert_result
                result.total_changes += len(upsert_result.audit_changes)
                result.total_new_records += upsert_result.new_rows
                result.total_updated_records += upsert_result.updated_rows
                
                # Track affected customer IDs
                if ftype == 'customers':
                    for change in upsert_result.audit_changes:
                        result.affected_customer_ids.add(change.id_value)
                    if upsert_result.new_rows > 0:
                        try:
                            prepared_df = pd.read_parquet(prepared_path)
                            if id_col in prepared_df.columns:
                                new_ids = prepared_df[id_col].astype(str).unique()
                                result.affected_customer_ids.update(new_ids)
                        except Exception as e:
                            result.warnings.append(f"Could not extract new customer IDs: {e}")
                
                # Update progress
                if progress_callback:
                    phase1_progress = 0.1 + (0.3 * (i + 1) / total_files)
                    progress_callback(f"Processed {ftype}: {len(upsert_result.audit_changes)} changes", phase1_progress)
                
            except Exception as e:
                error_msg = f"Upsert failed for {ftype}: {str(e)}"
                result.errors.append(error_msg)
                self._log_event("error", error_msg)
        
        # Log optimization results
        self._log_event("info", f"Upsert processing completed", {
            "total_files": total_files,
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "optimization_saved": f"{files_skipped}/{total_files} files skipped"
        })
    
    def _check_file_has_changes(self, prepared_path: Path, ftype: str, id_col: str, master_filename: Optional[str]) -> bool:
        """Pre-check if a file has any changes before processing it - optimization to skip empty updates"""
        
        try:
            from .engine import _canonicalize_series_for_compare
            
            # Find master file
            parquet_dir = self.parquet_dir
            master_files = list(parquet_dir.glob(f"{ftype.capitalize()}.parquet"))
            
            if not master_files:
                return True  # New master file - always has changes
            
            # Load data
            prepared_df = pd.read_parquet(prepared_path)
            master_df = pd.read_parquet(master_files[0])
            
            # Check ID column exists
            if id_col not in prepared_df.columns or id_col not in master_df.columns:
                return True  # Can't compare - assume changes
            
            # Normalize IDs same way as upsert engine
            master_df[id_col] = master_df[id_col].astype("string")
            prepared_df[id_col] = prepared_df[id_col].astype("string") 
            master_df[id_col] = _canonicalize_series_for_compare(master_df[id_col], id_col, id_col)
            prepared_df[id_col] = _canonicalize_series_for_compare(prepared_df[id_col], id_col, id_col)
            
            # Merge to find changes
            merged = master_df.merge(prepared_df, how="outer", on=id_col, indicator=True, suffixes=("_x", "_y"))
            
            # Check for new records
            new_records = len(merged[merged["_merge"] == "right_only"])
            if new_records > 0:
                return True
            
            # Check for changes in existing records
            both_df = merged[merged["_merge"] == "both"]
            if len(both_df) == 0:
                return False
            
            for col in master_df.columns:
                if col == id_col:
                    continue
                master_col = f"{col}_x"
                prepared_col = f"{col}_y"
                if master_col in both_df.columns and prepared_col in both_df.columns:
                    lhs = _canonicalize_series_for_compare(both_df[master_col], col, id_col)
                    rhs = _canonicalize_series_for_compare(both_df[prepared_col], col, id_col)
                    if (lhs != rhs).any():
                        return True
            
            return False  # No changes detected
            
        except Exception as e:
            self._log_event("warning", f"Change check failed for {ftype}, will process anyway: {e}")
            return True  # Error - assume changes to be safe
    
    def _process_changes(self, result: IntegratedUpsertResult, progress_callback: Optional[callable]):
        """Phase 2: Log change summary for tracking"""
        
        try:
            total_changes = sum(len(upsert_result.audit_changes) for upsert_result in result.upsert_results.values())
            
            self._log_event("info", f"Master files updated with {total_changes} changes - ready for Historical Events processing", {
                "total_changes": total_changes,
                "total_new_records": result.total_new_records,
                "total_updated_records": result.total_updated_records,
                "files_processed": list(result.upsert_results.keys())
            })
            
        except Exception as e:
            error_msg = f"Change logging failed: {str(e)}"
            result.errors.append(error_msg)
            self._log_event("error", error_msg)
    
    def _log_event(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log event to JSONL file"""
        
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
            "company": self.company,
            "component": "integrated_upsert_engine"
        }
        
        if details:
            log_entry.update(details)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass


def find_prepared_files(downloads_dir: Path) -> Dict[str, Path]:
    """Find all prepared parquet files in the downloads directory"""
    
    prepared_files = {}
    
    # Find prepared parquet files only
    for file_path in downloads_dir.glob("prepared_*.parquet"):
        filename = file_path.name.lower()
        
        # Pattern matching for file types
        if 'customer' in filename:
            prepared_files['customers'] = file_path
        elif 'call' in filename:
            prepared_files['calls'] = file_path
        elif 'job' in filename:
            prepared_files['jobs'] = file_path
        elif 'estimate' in filename:
            prepared_files['estimates'] = file_path
        elif 'invoice' in filename:
            prepared_files['invoices'] = file_path
        elif 'location' in filename:
            prepared_files['locations'] = file_path
        elif 'membership' in filename:
            prepared_files['memberships'] = file_path
    
    return prepared_files


def create_upsert_config(company_config: Any) -> Dict[str, Any]:
    """Create upsert configuration from company config"""
    
    config = {
        'backup': True,
        'backup_dir': None,
        'write_mode': 'inplace',
        'dry_run': False,
        'maintain_store': False,
        'prefer_parquet': True,
        'id_columns': {},
        'master_filenames': {},
        'event_rules': None
    }
    
    # Extract from company config if available
    if hasattr(company_config, 'upsert'):
        if hasattr(company_config.upsert, 'id_column_by_type'):
            config['id_columns'] = company_config.upsert.id_column_by_type
        if hasattr(company_config.upsert, 'event_rules'):
            config['event_rules'] = company_config.upsert.event_rules
    
    if hasattr(company_config, 'prepare'):
        if hasattr(company_config.prepare, 'file_type_to_master'):
            config['master_filenames'] = company_config.prepare.file_type_to_master
    
    return config
