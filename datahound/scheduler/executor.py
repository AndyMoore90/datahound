"""Task executor for scheduled tasks"""

import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path to ensure imports work
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .tasks import ScheduledTask, TaskType


class TaskExecutor:
    """Executes scheduled tasks based on their type and configuration"""
    
    def __init__(self, base_dir: Path):
        """Initialize the task executor
        
        Args:
            base_dir: Base directory for the application
        """
        self.base_dir = Path(base_dir)
    
    def execute_task(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute a scheduled task
        
        Args:
            task: The scheduled task to execute
            
        Returns:
            Dictionary with success status and any error messages
        """
        try:
            task_type = task.task_config.task_type
            
            if task_type == TaskType.DOWNLOAD:
                return self._execute_download(task)
            elif task_type == TaskType.PREPARE:
                return self._execute_prepare(task)
            elif task_type == TaskType.INTEGRATED_UPSERT:
                return self._execute_upsert(task)
            elif task_type == TaskType.HISTORICAL_EVENT_SCAN:
                return self._execute_event_scan(task)
            elif task_type == TaskType.CUSTOM_EXTRACTION:
                return self._execute_extraction(task)
            elif task_type == TaskType.CREATE_CORE_DATA:
                return self._execute_core_data(task)
            elif task_type == TaskType.REFRESH_CORE_DATA:
                return self._execute_refresh_core_data(task)
            else:
                return {"success": False, "error": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def _execute_download(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute download task"""
        try:
            from datahound.download.types import load_config
            from datahound.download.gmail import GmailDownloader
            
            config = task.task_config
            
            # Load company configuration
            cfg_path = Path("companies") / config.company / "config.json"
            if not cfg_path.exists():
                return {"success": False, "error": f"Config not found: {cfg_path}"}
            
            cfg = load_config(cfg_path)
            
            # Apply task settings
            cfg.mark_as_read = config.mark_as_read
            
            # Create downloader
            downloader = GmailDownloader(cfg)
            
            # Archive if requested
            if config.archive_existing:
                downloader.archive_existing_files()
            
            # Run download
            results = downloader.run(config.file_types or [])
            
            # Dedup if requested
            if config.dedup_after and results:
                import hashlib
                base = Path(cfg.data_dir)
                removed = []
                
                for t, files in results.items():
                    if not files:
                        continue
                    seen: Dict[str, Path] = {}
                    stamped = sorted(
                        [base / f for f in files if (base / f).exists()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True
                    )
                    for p in stamped:
                        try:
                            h = hashlib.sha256(p.read_bytes()).hexdigest()
                            if h in seen:
                                p.unlink()
                                removed.append(str(p.name))
                            else:
                                seen[h] = p
                        except:
                            continue
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_prepare(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute prepare task"""
        try:
            from datahound.download.types import load_config
            from datahound.prepare.engine import prepare_latest_files
            
            config = task.task_config
            
            # Load company configuration
            cfg_path = Path("companies") / config.company / "config.json"
            if not cfg_path.exists():
                return {"success": False, "error": f"Config not found: {cfg_path}"}
            
            cfg = load_config(cfg_path)
            
            if not cfg.prepare:
                return {"success": False, "error": "Prepare configuration not found"}
            
            # Run prepare
            results = prepare_latest_files(
                cfg,
                selected_types=config.prepare_types or [],
                write_csv=config.write_csv,
                write_parquet=config.write_parquet
            )
            
            return {"success": True, "results": {k: str(v) for k, v in results.items()}}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_upsert(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute integrated upsert task"""
        try:
            from datahound.upsert.integrated_engine import (
                IntegratedUpsertEngine, find_prepared_files, create_upsert_config
            )
            from datahound.download.types import load_config
            
            config = task.task_config
            
            # Load company configuration
            cfg_path = Path("companies") / config.company / "config.json"
            if not cfg_path.exists():
                return {"success": False, "error": f"Config not found: {cfg_path}"}
            
            cfg = load_config(cfg_path)
            
            # Get paths
            data_dir = Path("data") / config.company
            downloads_dir = data_dir / "downloads"
            parquet_dir = Path("companies") / config.company / "parquet"
            
            # Get prepared files
            prepared_files = find_prepared_files(downloads_dir) if downloads_dir.exists() else {}
            
            if not prepared_files:
                return {"success": False, "error": "No prepared files found"}
            
            # Initialize engine
            engine = IntegratedUpsertEngine(config.company, data_dir, parquet_dir)
            
            # Create upsert configuration
            upsert_config = create_upsert_config(cfg)
            upsert_config.update({
                'backup': config.backup_files,
                'dry_run': config.dry_run,
                'write_mode': config.write_mode
            })
            
            # Run upsert
            result = engine.process_prepared_files(
                prepared_files=prepared_files,
                config=upsert_config
            )
            
            return {
                "success": not result.errors,
                "total_changes": result.total_changes,
                "new_records": result.total_new_records,
                "updated_records": result.total_updated_records,
                "errors": result.errors
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_event_scan(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute historical event scan task"""
        try:
            import json
            import pandas as pd
            from pathlib import Path
            from datetime import datetime, UTC
            
            config = task.task_config
            company = config.company
            
            # Load event configuration
            config_path = Path("config/events") / f"{company}_historical_events_config.json"
            if not config_path.exists():
                return {"success": False, "error": f"Event config not found: {config_path}"}
            
            with open(config_path, 'r') as f:
                full_config = json.load(f)
            
            # Extract events configuration
            event_config = full_config.get("events", {})
            
            # Determine which events to scan
            if config.scan_all_events:
                event_types = list(event_config.keys())
            elif config.event_type:
                event_types = [config.event_type]
            else:
                return {"success": False, "error": "No event type specified"}
            
            # Use the same scanning logic as the Historical Events page
            results = {}
            errors = []
            
            # Get paths
            parquet_dir = Path("companies") / company / "parquet"
            events_master_default_dir = Path("data") / company / "events" / "master_files"
            events_master_default_dir.mkdir(parents=True, exist_ok=True)
            
            for event_type in event_types:
                if event_type not in event_config:
                    errors.append(f"Event type not configured: {event_type}")
                    continue
                
                try:
                    type_config = event_config[event_type]
                    
                    # Execute the specific event type using the same logic as the UI
                    if event_type == "overdue_maintenance":
                        result = self._scan_overdue_maintenance(company, type_config, parquet_dir, events_master_default_dir)
                    elif event_type == "unsold_estimates":
                        result = self._scan_unsold_estimates(company, type_config, parquet_dir, events_master_default_dir)
                    elif event_type == "canceled_jobs":
                        result = self._scan_canceled_jobs(company, type_config, parquet_dir, events_master_default_dir)
                    elif event_type == "aging_systems":
                        result = self._scan_aging_systems(company, type_config, parquet_dir, events_master_default_dir)
                    elif event_type == "lost_customers":
                        result = self._scan_lost_customers(company, type_config, parquet_dir, events_master_default_dir)
                    else:
                        errors.append(f"Unknown event type: {event_type}")
                        continue
                    
                    results[event_type] = result
                    
                except Exception as e:
                    errors.append(f"Error scanning {event_type}: {str(e)}")
            
            return {
                "success": len(errors) == 0,
                "results": results,
                "errors": errors
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _scan_overdue_maintenance(self, company: str, cfg: Dict[str, Any], parquet_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Scan for overdue maintenance using the same logic as the UI"""
        try:
            import pandas as pd
            from datetime import datetime, UTC
            
            # Load source file
            src = parquet_dir / cfg.get("file", "Jobs.parquet")
            if not src.exists():
                return {"success": False, "error": f"Source file not found: {src}"}
            
            df = pd.read_parquet(src, engine="pyarrow")
            
            # Apply filters (same logic as UI)
            fcol = cfg.get("filter_column", "Job Type")
            match_values = cfg.get("filter_match_values", [])
            if not isinstance(match_values, list):
                match_values = []
            match_values_lower = [str(v).lower().strip() for v in match_values if v]
            status_col = cfg.get("status_column", "Status")
            status_value = str(cfg.get("status_value", "Completed")).strip()
            date_col = cfg.get("date_column", "Completion Date")
            id_col = cfg.get("id_column", "Job ID")
            months_threshold = int(cfg.get("months_threshold", 12))
            
            # Filter by exact match (case-insensitive) against list of values
            if fcol in df.columns:
                if match_values_lower:
                    col_values_lower = df[fcol].astype(str).str.lower().str.strip()
                    filtered = df[col_values_lower.isin(match_values_lower)].copy()
                else:
                    filtered = df.copy()
            else:
                filtered = df.copy()
            
            # Filter by Status column
            if status_col in filtered.columns:
                status_mask = filtered[status_col].astype(str).str.strip() == status_value
                filtered = filtered[status_mask].copy()
            
            # Calculate months since completion
            if date_col in filtered.columns:
                dt = pd.to_datetime(filtered[date_col], errors="coerce")
                now = pd.Timestamp.utcnow().tz_localize(None)
                months_since = ((now.year - dt.dt.year) * 12 + (now.month - dt.dt.month)).astype("Int64")
                filtered["months_since"] = months_since
                overdue = filtered[filtered["months_since"] >= months_threshold].copy()
            else:
                overdue = filtered.iloc[0:0]
            
            # Apply invalidation rules (same as UI)
            ex_col = cfg.get("exclude_column", "")
            ex_eq = cfg.get("exclude_equals", "")
            ex_contains = cfg.get("exclude_contains", "")
            
            if ex_col and ex_col in overdue.columns and not overdue.empty:
                mask = pd.Series([True] * len(overdue), index=overdue.index)
                
                if ex_eq:
                    eq_mask = overdue[ex_col].astype(str) != ex_eq
                    mask &= eq_mask
                
                if ex_contains:
                    cont_mask = ~overdue[ex_col].astype(str).str.lower().str.contains(str(ex_contains).lower(), na=False)
                    mask &= cont_mask
                
                overdue = overdue[mask]
            
            # Apply deduplication (same as UI)
            if cfg.get("dedup_enabled", False) and not overdue.empty:
                dedup_cust_col = cfg.get("dedup_customer_column", "Customer ID")
                dedup_loc_col = cfg.get("dedup_location_column", "Location ID")
                
                if dedup_cust_col in overdue.columns and dedup_loc_col in overdue.columns:
                    # Group by customer + location, keep most recent completion date
                    completion_dates = pd.to_datetime(overdue[date_col], errors="coerce")
                    dedup_key = overdue[dedup_cust_col].astype(str) + "||" + overdue[dedup_loc_col].astype(str)
                    
                    dedup_df = pd.DataFrame({
                        "key": dedup_key,
                        "completion_date": completion_dates,
                    }, index=overdue.index)
                    
                    dedup_df = dedup_df.sort_values(["key", "completion_date"], ascending=[True, False])
                    keep_indices = dedup_df.groupby("key").head(1).index.tolist()
                    overdue = overdue.loc[keep_indices]
            
            # Apply processing limit
            limit = int(cfg.get("processing_limit", 0) or 0)
            if limit > 0 and not overdue.empty:
                overdue = overdue.head(limit)
            
            # Build events
            if not overdue.empty:
                payload_cols = [c for c in (cfg.get("payload_columns") or []) if c in overdue.columns]
                base_cols = {
                    "event_type": "overdue_maintenance",
                    "entity_type": "job",
                    "entity_id": overdue[id_col].astype(str) if id_col in overdue.columns else overdue.index.astype(str),
                    "detected_at": datetime.now(UTC).isoformat(),
                    "months_overdue": overdue["months_since"] if "months_since" in overdue.columns else pd.Series([], dtype="Int64"),
                    "job_class": overdue[fcol].astype(str) if fcol in overdue.columns else "",
                    "completion_date": overdue[date_col].astype(str) if date_col in overdue.columns else "",
                }
                events = pd.DataFrame(base_cols)
                for col in payload_cols:
                    events[col] = overdue[col].astype(str)
                
                # Save to output file
                output_file = output_dir / cfg.get("output_filename", "overdue_maintenance_master.parquet")
                events.to_parquet(output_file, index=False)
                
                return {
                    "success": True,
                    "events_found": len(events),
                    "output_file": str(output_file)
                }
            else:
                return {
                    "success": True,
                    "events_found": 0,
                    "message": "No overdue maintenance found"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _scan_unsold_estimates(self, company: str, cfg: Dict[str, Any], parquet_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Scan for unsold estimates using the same logic as the UI"""
        try:
            import pandas as pd
            from datetime import datetime, UTC
            
            # Load source file
            src = parquet_dir / cfg.get("file", "Estimates.parquet")
            if not src.exists():
                return {"success": False, "error": f"Source file not found: {src}"}
            
            df = pd.read_parquet(src, engine="pyarrow")
            
            # Get configuration
            status_col = cfg.get("status_column", "Estimate Status")
            include_vals = [str(v) for v in (cfg.get("status_include_values") or ["Dismissed", "Open"])]
            creation_col = cfg.get("creation_date_column", "Creation Date")
            months_back = int(cfg.get("months_back", 24))
            opp_col = cfg.get("opportunity_status_column", "Opportunity Status")
            opp_exclude = str(cfg.get("opportunity_exclude_value", "Won"))
            id_col = cfg.get("id_column", "Estimate ID")
            
            # Filter by status
            if status_col in df.columns:
                filtered = df[df[status_col].astype(str).isin(include_vals)].copy()
            else:
                filtered = df.copy()
            
            # Filter by creation date (within months_back)
            if creation_col in filtered.columns:
                dt = pd.to_datetime(filtered[creation_col], errors="coerce")
                now = pd.Timestamp.utcnow().tz_localize(None)
                months_since = ((now.year - dt.dt.year) * 12 + (now.month - dt.dt.month)).astype("Int64")
                recent = filtered[months_since <= months_back].copy()
            else:
                recent = filtered.copy()
            
            # Filter by opportunity status (exclude Won)
            if opp_col in recent.columns:
                recent = recent[recent[opp_col].astype(str) != opp_exclude].copy()
            
            # Apply invalidation logic (same as UI)
            if not recent.empty:
                # Invalidation A: Summary contains invalidation text
                sum_col = cfg.get("summary_column", "Estimate Summary")
                sum_contains = str(cfg.get("summary_invalidate_contains", "Test Estimate"))
                if sum_col in recent.columns and sum_contains:
                    before_sum = len(recent)
                    mask_sum = ~recent[sum_col].astype(str).str.contains(sum_contains, na=False, case=False)
                    recent = recent[mask_sum]
                
                # Invalidation B: Same location has recent completion
                loc_col = cfg.get("location_column", "Location ID")
                est_comp_col = cfg.get("estimate_completion_date_column", "Completion Date")
                recent_est_days = int(cfg.get("recent_estimate_days", 30))
                
                if loc_col in recent.columns and est_comp_col in df.columns:
                    # Find estimates with recent completions at same location
                    est_comp_dates = pd.to_datetime(df[est_comp_col], errors="coerce")
                    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=recent_est_days)
                    recent_completions = (est_comp_dates >= cutoff) & (~est_comp_dates.isna())
                    
                    # Group by location to find which locations have recent completions
                    has_recent_completion = df.groupby(df[loc_col].astype(str))[est_comp_col].apply(
                        lambda g: recent_completions.loc[g.index].any()
                    )
                    
                    # Filter out estimates at locations with recent completions
                    recent_loc = recent[loc_col].astype(str)
                    flag = recent_loc.map(has_recent_completion).fillna(False)
                    recent = recent[~flag]
                
                # Invalidation C: Same customer has job within ±recent_job_days
                jobs_file = cfg.get("jobs_file", "Jobs.parquet")
                cust_col = cfg.get("customer_column", "Customer ID")
                job_created_col = cfg.get("job_created_date_column", "Created Date")
                job_scheduled_col = cfg.get("job_scheduled_date_column", "Scheduled Date")
                recent_job_days = int(cfg.get("recent_job_days", 21))
                
                jobs_path = parquet_dir / jobs_file
                if jobs_path.exists() and cust_col in recent.columns:
                    try:
                        jobs_df = pd.read_parquet(jobs_path, engine="pyarrow")
                        
                        # For each estimate, check if customer has recent job
                        estimates_to_keep = []
                        for idx, est_row in recent.iterrows():
                            customer_id = est_row[cust_col]
                            est_creation = pd.to_datetime(est_row[creation_col], errors="coerce")
                            
                            if pd.isna(est_creation):
                                estimates_to_keep.append(idx)
                                continue
                            
                            # Find jobs for this customer
                            if cust_col in jobs_df.columns:
                                customer_jobs = jobs_df[jobs_df[cust_col].astype(str) == str(customer_id)]
                                
                                has_conflicting_job = False
                                for job_date_col in [job_created_col, job_scheduled_col]:
                                    if job_date_col in customer_jobs.columns:
                                        job_dates = pd.to_datetime(customer_jobs[job_date_col], errors="coerce")
                                        valid_job_dates = job_dates[~job_dates.isna()]
                                        
                                        for job_date in valid_job_dates:
                                            days_diff = abs((est_creation - job_date).days)
                                            if days_diff <= recent_job_days:
                                                has_conflicting_job = True
                                                break
                                    
                                    if has_conflicting_job:
                                        break
                                
                                if not has_conflicting_job:
                                    estimates_to_keep.append(idx)
                            else:
                                estimates_to_keep.append(idx)
                        
                        recent = recent.loc[estimates_to_keep]
                    except Exception:
                        pass  # Keep all estimates if job checking fails
                
                # Deduplication (same as UI)
                if cfg.get("dedup_enabled", False) and not recent.empty:
                    dedup_cust_col = cfg.get("dedup_customer_column", "Customer ID")
                    dedup_loc_col = cfg.get("dedup_location_column", "Location ID")
                    
                    if dedup_cust_col in recent.columns and dedup_loc_col in recent.columns:
                        # Group by customer + location, keep most recent creation date
                        creation_dates = pd.to_datetime(recent[creation_col], errors="coerce")
                        dedup_key = recent[dedup_cust_col].astype(str) + "||" + recent[dedup_loc_col].astype(str)
                        
                        dedup_df = pd.DataFrame({
                            "key": dedup_key,
                            "creation_date": creation_dates,
                        }, index=recent.index)
                        
                        dedup_df = dedup_df.sort_values(["key", "creation_date"], ascending=[True, False])
                        keep_indices = dedup_df.groupby("key").head(1).index.tolist()
                        recent = recent.loc[keep_indices]
            
            # Apply processing limit
            limit = int(cfg.get("processing_limit", 0) or 0)
            if limit > 0 and not recent.empty:
                recent = recent.head(limit)
            
            # Build events
            if not recent.empty:
                payload_cols = [c for c in (cfg.get("payload_columns") or []) if c in recent.columns]
                base_cols = {
                    "event_type": "unsold_estimates",
                    "entity_type": "estimate",
                    "entity_id": recent[id_col].astype(str) if id_col in recent.columns else recent.index.astype(str),
                    "detected_at": datetime.now(UTC).isoformat(),
                    "estimate_status": recent[status_col].astype(str) if status_col in recent.columns else "",
                    "creation_date": recent[creation_col].astype(str) if creation_col in recent.columns else "",
                    "opportunity_status": recent[opp_col].astype(str) if opp_col in recent.columns else "",
                    "location_id": recent[loc_col].astype(str) if loc_col in recent.columns else "",
                }
                events = pd.DataFrame(base_cols)
                for col in payload_cols:
                    events[col] = recent[col].astype(str)
                
                # Save to output file
                output_file = output_dir / cfg.get("output_filename", "unsold_estimates_master.parquet")
                events.to_parquet(output_file, index=False)
                
                return {
                    "success": True,
                    "events_found": len(events),
                    "output_file": str(output_file)
                }
            else:
                return {
                    "success": True,
                    "events_found": 0,
                    "message": "No unsold estimates found"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _scan_canceled_jobs(self, company: str, cfg: Dict[str, Any], parquet_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Scan for canceled jobs using the same logic as the UI"""
        try:
            import pandas as pd
            from datetime import datetime, UTC
            
            # Load source file
            src = parquet_dir / cfg.get("file", "Jobs.parquet")
            if not src.exists():
                return {"success": False, "error": f"Source file not found: {src}"}
            
            df = pd.read_parquet(src, engine="pyarrow")
            
            # Get configuration
            status_col = cfg.get("status_column", "Status")
            canceled_val = str(cfg.get("status_canceled_value", "Canceled"))
            date_col = cfg.get("date_column", "Completion Date")
            id_col = cfg.get("id_column", "Job ID")
            months_back = int(cfg.get("months_back", 24))
            
            # Filter by canceled status
            if status_col in df.columns:
                filtered = df[df[status_col].astype(str) == canceled_val].copy()
            else:
                filtered = df.iloc[0:0]  # Empty if no status column
            
            # Filter by date (within months_back)
            if date_col in filtered.columns and not filtered.empty:
                dt = pd.to_datetime(filtered[date_col], errors="coerce")
                now = pd.Timestamp.utcnow().tz_localize(None)
                months_since = ((now.year - dt.dt.year) * 12 + (now.month - dt.dt.month)).astype("Int64")
                recent = filtered[months_since <= months_back].copy()
                try:
                    recent["cancellation_age_months"] = months_since.loc[recent.index]
                except Exception:
                    pass
            else:
                recent = filtered.iloc[0:0]
            
            # Apply invalidation logic (same as UI)
            # First: Exclude if summary contains invalidation text
            sum_col = cfg.get("summary_column", "")
            sum_contains = cfg.get("summary_invalidate_contains", "")
            if sum_col and sum_col in recent.columns and sum_contains and not recent.empty:
                before_sum = len(recent)
                mask_sum = ~recent[sum_col].astype(str).str.lower().str.contains(sum_contains.lower(), na=False)
                recent = recent[mask_sum]
            
            # Second: Exclude canceled jobs if same location has non-canceled job within recent days
            loc_col = cfg.get("location_column", "Location ID")
            inv_date_col = cfg.get("invalidation_date_column", "Completion Date")
            inv_days = int(cfg.get("invalidation_days_recent", 30))
            
            if not recent.empty and loc_col in df.columns and inv_date_col in df.columns and status_col in df.columns:
                df_status_norm = df[status_col].astype(str).str.strip().str.lower()
                df_dates = pd.to_datetime(df[inv_date_col], errors="coerce")
                cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=inv_days)
                non_canceled_recent = (df_status_norm != canceled_val.strip().lower()) & (df_dates >= cutoff)
                
                # Map location -> has_non_canceled_recent
                has_recent_map = df.groupby(df[loc_col].astype(str))[inv_date_col].apply(
                    lambda g: bool(non_canceled_recent.loc[g.index].any())
                )
                
                # Filter out canceled jobs at locations with recent non-canceled jobs
                recent_loc = recent[loc_col].astype(str)
                flag = recent_loc.map(has_recent_map).fillna(False)
                recent = recent[~flag]
            
            # Apply deduplication if enabled
            if cfg.get("dedup_enabled", False) and not recent.empty:
                dedup_cust_col = cfg.get("dedup_customer_column", "Customer ID")
                dedup_loc_col = cfg.get("dedup_location_column", "Location ID")
                
                if dedup_cust_col in recent.columns and dedup_loc_col in recent.columns:
                    # Group by customer + location, keep most recent completion date
                    completion_dates = pd.to_datetime(recent[date_col], errors="coerce")
                    dedup_key = recent[dedup_cust_col].astype(str) + "||" + recent[dedup_loc_col].astype(str)
                    
                    dedup_df = pd.DataFrame({
                        "key": dedup_key,
                        "completion_date": completion_dates,
                    }, index=recent.index)
                    
                    dedup_df = dedup_df.sort_values(["key", "completion_date"], ascending=[True, False])
                    keep_indices = dedup_df.groupby("key").head(1).index.tolist()
                    recent = recent.loc[keep_indices]
            
            # Apply processing limit
            limit = int(cfg.get("processing_limit", 0) or 0)
            if limit > 0 and not recent.empty:
                recent = recent.head(limit)
            
            # Build events
            if not recent.empty:
                payload_cols = [c for c in (cfg.get("payload_columns") or []) if c in recent.columns]
                base_cols = {
                    "event_type": "canceled_jobs",
                    "entity_type": "job", 
                    "entity_id": recent[id_col].astype(str) if id_col in recent.columns else recent.index.astype(str),
                    "detected_at": datetime.now(UTC).isoformat(),
                    "status": recent[status_col].astype(str) if status_col in recent.columns else "",
                    "completion_date": recent[date_col].astype(str) if date_col in recent.columns else "",
                    "location_id": recent[loc_col].astype(str) if loc_col in recent.columns else "",
                    "cancellation_age_months": recent.get("cancellation_age_months", pd.Series([], dtype="Int64")),
                }
                events = pd.DataFrame(base_cols)
                for col in payload_cols:
                    events[col] = recent[col].astype(str)
                
                # Save to output file
                output_file = output_dir / cfg.get("output_filename", "canceled_jobs_master.parquet")
                events.to_parquet(output_file, index=False)
                
                return {
                    "success": True,
                    "events_found": len(events),
                    "output_file": str(output_file)
                }
            else:
                return {
                    "success": True,
                    "events_found": 0,
                    "message": "No canceled jobs found"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _scan_aging_systems(self, company: str, cfg: Dict[str, Any], parquet_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Placeholder for aging systems scanning"""
        return {"success": True, "message": "Aging systems scan not yet implemented in executor"}
    
    def _scan_lost_customers(self, company: str, cfg: Dict[str, Any], parquet_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Placeholder for lost customers scanning"""
        return {"success": True, "message": "Lost customers scan not yet implemented in executor"}
    
    def _execute_extraction(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute custom extraction task"""
        try:
            from datahound.extract import CustomExtractionEngine, ExtractionBatch
            
            config = task.task_config
            company = config.company
            
            # Initialize extraction engine
            data_dir = Path("data") / company
            parquet_dir = Path("companies") / company / "parquet"
            
            engine = CustomExtractionEngine(company, data_dir, parquet_dir)
            
            # Execute all enabled extractions using the same logic as the UI
            if config.execute_all_enabled:
                # Try to load saved extraction configuration first
                try:
                    # Import the load function from the UI page
                    sys.path.insert(0, str(Path("apps/pages")))
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("custom_extraction", Path("apps/pages/13_Custom_Extraction.py"))
                    custom_extraction_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(custom_extraction_module)
                    
                    # Load saved configurations
                    saved_configs = custom_extraction_module.load_extraction_configuration(company)
                    
                    if saved_configs:
                        enabled_configs = [c for c in saved_configs if c.enabled]
                    else:
                        # Fall back to default templates
                        available_configs = engine.get_available_extractions()
                        enabled_configs = [c for c in available_configs if c.enabled]
                        
                except Exception as load_error:
                    # Fall back to default templates if loading fails
                    available_configs = engine.get_available_extractions()
                    enabled_configs = [c for c in available_configs if c.enabled]
                
                if not enabled_configs:
                    return {"success": False, "error": "No enabled extraction configurations found"}
                
                # Execute each extraction
                results = []
                errors = []
                
                for extraction_config in enabled_configs:
                    try:
                        result = engine.extract_single(extraction_config)
                        results.append(result)
                        
                        if not result.success:
                            errors.append(f"{extraction_config.name}: {result.error_message}")
                            
                    except Exception as e:
                        errors.append(f"{extraction_config.name}: {str(e)}")
                
                # Calculate totals
                total_extracted = sum(r.records_found for r in results if r.success)
                total_saved = sum(r.records_saved for r in results if r.success)
                successful_extractions = len([r for r in results if r.success])
                
                return {
                    "success": len(errors) == 0,
                    "total_extracted": total_extracted,
                    "total_saved": total_saved,
                    "extractions_run": len(results),
                    "successful_extractions": successful_extractions,
                    "errors": errors
                }
            else:
                return {"success": False, "error": "Extraction not configured"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_core_data(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute create core data task"""
        try:
            from datahound.profiles.enhanced_core_data import EnhancedCustomerCoreDataBuilder
            from datahound.profiles.types import ProfileBuildConfig, ProfileBuildMode
            
            config = task.task_config
            company = config.company
            
            # Get paths
            parquet_dir = Path("companies") / company / "parquet"
            data_dir = Path("data") / company
            
            # Create builder
            builder = EnhancedCustomerCoreDataBuilder(
                company=company,
                parquet_dir=parquet_dir,
                data_dir=data_dir
            )
            
            # Build configuration
            build_config = ProfileBuildConfig(
                mode=ProfileBuildMode.ALL_CUSTOMERS,
                processing_limit=config.processing_limit,
                include_rfm=config.include_rfm,
                include_demographics=config.include_demographics,
                include_permits=config.include_permits,
                include_marketable=config.include_marketable,
                include_segments=config.include_segments
            )
            
            # Build core data
            result = builder.build_enhanced_customer_profiles(build_config)
            
            return {
                "success": result.errors_encountered == 0,
                "customers_processed": result.total_customers_processed,
                "new_profiles": result.new_profiles_created,
                "updated_profiles": result.existing_profiles_updated,
                "errors_encountered": result.errors_encountered,
                "duration_seconds": result.total_duration_seconds,
                "error_details": result.error_details
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_refresh_core_data(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute refresh core data task (new customers only)"""
        try:
            from datahound.profiles.enhanced_core_data import EnhancedCustomerCoreDataBuilder
            from datahound.profiles.types import ProfileBuildConfig, ProfileBuildMode
            
            config = task.task_config
            company = config.company
            
            parquet_dir = Path("companies") / company / "parquet"
            data_dir = Path("data") / company
            
            builder = EnhancedCustomerCoreDataBuilder(
                company=company,
                parquet_dir=parquet_dir,
                data_dir=data_dir
            )
            
            build_config = ProfileBuildConfig(
                mode=ProfileBuildMode.NEW_CUSTOMERS_ONLY,
                processing_limit=config.processing_limit,
                include_rfm=config.include_rfm,
                include_demographics=config.include_demographics,
                include_permits=config.include_permits,
                include_marketable=config.include_marketable,
                include_segments=config.include_segments
            )
            
            result = builder.build_enhanced_customer_profiles(build_config)
            
            return {
                "success": result.errors_encountered == 0,
                "customers_processed": result.total_customers_processed,
                "new_profiles": result.new_profiles_created,
                "updated_profiles": result.existing_profiles_updated,
                "errors_encountered": result.errors_encountered,
                "duration_seconds": result.total_duration_seconds,
                "error_details": result.error_details
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

