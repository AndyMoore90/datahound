"""Integrated Upsert Management Page"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List

import streamlit as st
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._shared import ensure_root_on_path, select_company_config
from apps.components.ui_components import (
    inject_custom_css, dh_page_header, dh_alert, dh_breadcrumbs
)
from apps.components.scheduler_ui import (
    render_schedule_config, render_task_manager, 
    create_scheduled_task, render_scheduler_status
)
from datahound.scheduler import TaskType

ensure_root_on_path()

from datahound.upsert.integrated_engine import (
    IntegratedUpsertEngine, find_prepared_files, create_upsert_config
)


def are_values_semantically_different(master_series: pd.Series, prepared_series: pd.Series, column_name: str) -> pd.Series:
    """
    Compare two pandas Series semantically, accounting for data type coercion issues
    Returns a boolean Series indicating which values are truly different
    """
    
    # Handle dates
    if 'date' in column_name.lower():
        return compare_dates_semantically(master_series, prepared_series)
    
    # Handle numeric values (expanded detection)
    numeric_keywords = [
        'revenue', 'total', 'rate', 'count', 'completed', 'payment', 'sales', 'locations',
        'jobs', 'invoices', 'opportunities', 'converted', 'canceled', 'hold', 'warranty',
        'recall', 'cancellation', '%', 'percent', 'booked', 'unconverted'
    ]
    if any(keyword in column_name.lower() for keyword in numeric_keywords):
        return compare_numbers_semantically(master_series, prepared_series)
    
    # Handle boolean values (expanded detection)
    boolean_keywords = ['do not', 'mail', 'service', 'lead', 'active', 'enabled', 'flag']
    if any(keyword in column_name.lower() for keyword in boolean_keywords):
        return compare_booleans_semantically(master_series, prepared_series)
    
    # Default: string comparison with normalization
    return compare_strings_semantically(master_series, prepared_series)


def compare_dates_semantically(master_series: pd.Series, prepared_series: pd.Series) -> pd.Series:
    """Compare dates semantically, handling different formats"""
    
    try:
        # Handle empty/null values first
        master_empty = master_series.astype(str).str.strip().isin(['', 'nan', 'NaT', 'None', 'NULL'])
        prepared_empty = prepared_series.astype(str).str.strip().isin(['', 'nan', 'NaT', 'None', 'NULL'])
        both_empty = master_empty & prepared_empty
        
        # Convert both to datetime, handling various formats (suppress warnings)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format")
            master_dates = pd.to_datetime(master_series, errors='coerce')
            prepared_dates = pd.to_datetime(prepared_series, errors='coerce')
        
        # Both NaT/NaN should be considered the same
        both_nat = master_dates.isna() & prepared_dates.isna()
        
        # Compare actual dates - only if both parsed successfully
        both_valid = ~master_dates.isna() & ~prepared_dates.isna()
        dates_different = (master_dates != prepared_dates) & both_valid
        
        # Combine: different unless both empty or both same date
        return dates_different & ~both_empty & ~both_nat
        
    except Exception:
        # Fallback to string comparison
        return compare_strings_semantically(master_series, prepared_series)


def compare_numbers_semantically(master_series: pd.Series, prepared_series: pd.Series) -> pd.Series:
    """Compare numbers semantically, handling int/float differences"""
    
    try:
        # Convert to numeric, handling empty strings as 0
        master_nums = pd.to_numeric(master_series.replace('', '0'), errors='coerce')
        prepared_nums = pd.to_numeric(prepared_series.replace('', '0'), errors='coerce')
        
        # Handle null values - both null should be same
        master_is_null = master_nums.isna() | (master_series.astype(str).str.strip() == '')
        prepared_is_null = prepared_nums.isna() | (prepared_series.astype(str).str.strip() == '')
        both_null = master_is_null & prepared_is_null
        
        # Handle zero values - treat empty string as 0
        master_nums = master_nums.fillna(0)
        prepared_nums = prepared_nums.fillna(0)
        
        # Compare numeric values (0 == 0.0, empty string == 0)
        nums_different = master_nums != prepared_nums
        
        # Don't count null-to-null or zero-to-zero as different
        return nums_different & ~both_null
        
    except Exception:
        # Fallback to string comparison
        return compare_strings_semantically(master_series, prepared_series)


def compare_booleans_semantically(master_series: pd.Series, prepared_series: pd.Series) -> pd.Series:
    """Compare boolean values semantically, handling different representations"""
    
    try:
        # Normalize boolean representations
        def normalize_boolean(series):
            normalized = series.astype(str).str.upper().str.strip()  # Use upper() to handle TRUE/FALSE
            # Map various representations to standard boolean
            bool_map = {
                'TRUE': True, '1': True, '1.0': True, 'YES': True, 'Y': True,
                'FALSE': False, '0': False, '0.0': False, 'NO': False, 'N': False,
                'NAN': None, 'NAT': None, '': None, 'NONE': None, 'NULL': None
            }
            return normalized.map(bool_map)
        
        master_bools = normalize_boolean(master_series)
        prepared_bools = normalize_boolean(prepared_series)
        
        # Both None should be considered the same
        both_null = master_bools.isna() & prepared_bools.isna()
        
        # Compare boolean values
        bools_different = master_bools != prepared_bools
        
        # Don't count null-to-null as different
        return bools_different & ~both_null
        
    except Exception:
        # Fallback to string comparison
        return compare_strings_semantically(master_series, prepared_series)


def compare_strings_semantically(master_series: pd.Series, prepared_series: pd.Series) -> pd.Series:
    """Compare strings semantically with normalization"""
    
    # Normalize strings
    master_normalized = master_series.astype(str).fillna("").str.strip()
    prepared_normalized = prepared_series.astype(str).fillna("").str.strip()
    
    # Replace common null representations with empty string
    null_values = ["nan", "None", "NULL", "N/A", "NaT", "nat", "0.0", "0"]
    for null_val in null_values:
        master_normalized = master_normalized.replace(null_val, "")
        prepared_normalized = prepared_normalized.replace(null_val, "")
    
    # Additional normalization for numeric-looking strings
    # If both can be converted to the same number, consider them equal
    try:
        master_as_num = pd.to_numeric(master_series, errors='coerce')
        prepared_as_num = pd.to_numeric(prepared_series, errors='coerce')
        
        # If both are numeric and equal, they're the same
        both_numeric = ~master_as_num.isna() & ~prepared_as_num.isna()
        numeric_same = master_as_num == prepared_as_num
        
        # For non-numeric or different numeric values, use string comparison
        string_different = master_normalized != prepared_normalized
        
        # Return: different strings unless they're the same number
        return string_different & ~(both_numeric & numeric_same)
        
    except Exception:
        # Simple string comparison fallback
        return master_normalized != prepared_normalized


def find_date_columns(df: pd.DataFrame) -> List[str]:
    """Find all date columns in a DataFrame using multiple detection methods"""
    
    date_columns = []
    
    for col in df.columns:
        is_date_column = False
        
        # Exclude obvious non-date columns first
        non_date_indicators = ['id', 'name', 'phone', 'email', 'total', 'amount', 'balance', 'fee', 'number', '#', 'status', 'type', 'summary']
        if any(indicator in col.lower() for indicator in non_date_indicators):
            continue
        
        # Method 1: Column name contains date-related keywords (exclude time-only columns)
        date_keywords = ['date', 'created', 'completed', 'dispatch', 'due', 'paid', 'issued', 'sold', 'from', 'to', 'start', 'end', 'expire']
        time_only_keywords = ['time', 'duration', 'length', 'elapsed']
        
        # Exclude time-only columns
        if any(time_keyword in col.lower() for time_keyword in time_only_keywords):
            continue  # Skip time-only columns
        
        if any(keyword in col.lower() for keyword in date_keywords):
            is_date_column = True
        
        # Method 2: Sample values look like dates (precise detection)
        if not is_date_column:
            sample_values = df[col].dropna().head(10).astype(str).tolist()
            if sample_values:
                # Check if values contain proper date patterns (exclude time-only patterns)
                date_patterns = [
                    r'^\d{1,2}/\d{1,2}/\d{4}$',  # Exact MM/DD/YYYY
                    r'^\d{4}-\d{1,2}-\d{1,2}$',  # Exact YYYY-MM-DD
                    r'^\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}:\d{1,2}$',  # ISO datetime
                ]
                
                # Time-only patterns to exclude
                time_only_patterns = [
                    r'^\d{1,2}:\d{2} (AM|PM)$',  # 7:48 PM
                    r'^\d{1,2}:\d{2}:\d{2}$',   # 14:30:25
                    r'^\d{1,2}:\d{2}$',         # 14:30
                ]
                
                import re
                date_like_count = 0
                time_only_count = 0
                total_non_empty = 0
                
                for val in sample_values[:5]:
                    val_str = str(val).strip()
                    if val_str and val_str not in ['', 'nan', 'None', 'False', 'True']:
                        total_non_empty += 1
                        
                        # Check if it's a time-only value
                        if any(re.match(pattern, val_str) for pattern in time_only_patterns):
                            time_only_count += 1
                        # Check if it's a date value
                        elif any(re.match(pattern, val_str) for pattern in date_patterns):
                            date_like_count += 1
                
                # Only consider it a date column if:
                # 1. Most values look like dates, AND
                # 2. No values look like time-only
                if (total_non_empty > 0 and 
                    date_like_count >= total_non_empty * 0.8 and 
                    time_only_count == 0):
                    is_date_column = True
        
        if is_date_column:
            date_columns.append(col)
    
    return date_columns


def render(company: str, config) -> None:
    show_update_master_data_interface(company, config)


def main():
    st.set_page_config(
        page_title="Update Master Data - DataHound Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()
    dh_page_header(
        title="📊 Update Master Data",
        subtitle="Update master parquet files with prepared data and track all changes"
    )
    dh_breadcrumbs(["Dashboard", "Data Processing", "Update Master Data"])
    company, config = select_company_config()
    if not company or not config:
        dh_alert("Please select a valid company configuration to proceed.", "warning")
        return
    dh_alert(f"Active Company: {company}", "success")
    render(company, config)


def show_update_master_data_interface(company: str, config: Dict[str, Any]):
    """Show the update master data interface"""
    
    # Get paths
    data_dir = Path("data") / company
    downloads_dir = data_dir / "downloads"
    parquet_dir = Path("companies") / company / "parquet"
    
    # Get prepared files
    prepared_files = find_prepared_files(downloads_dir) if downloads_dir.exists() else {}
    
    # Create tabs for manual and automated execution
    tab1, tab2, tab3 = st.tabs(["📊 Manual Update", "🤖 Automation", "📊 Scheduler Status"])
    
    with tab1:
        st.markdown("### Manual Update Master Data")
        
        # Debug: Show file detection results
        if st.checkbox("🔧 Show file detection debug info"):
            st.markdown("**File Detection Debug:**")
            st.write(f"Downloads directory: `{downloads_dir}`")
            st.write(f"Directory exists: {downloads_dir.exists()}")
            
            if downloads_dir.exists():
                all_parquet = list(downloads_dir.glob("prepared_*.parquet"))
                st.write(f"Found {len(all_parquet)} prepared parquet files:")
                for f in all_parquet:
                    st.write(f"  - {f.name}")
                
                st.write(f"Detected file types: {list(prepared_files.keys())}")
                for ftype, fpath in prepared_files.items():
                    st.write(f"  - {ftype}: {fpath.name}")
            else:
                st.error("Downloads directory not found!")
        
        # Status section
        st.markdown("### 📊 System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Check for prepared files (already loaded above)
            st.metric("📥 Prepared Files", f"{len(prepared_files)}")
        if prepared_files:
            st.success(f"Ready: {', '.join(prepared_files.keys())}")
        else:
            st.warning("No prepared files found")
    
    with col2:
        # Check master files
        master_files = list(parquet_dir.glob("*.parquet")) if parquet_dir.exists() else []
        core_files = [f for f in master_files if not f.name.endswith("_master.parquet")]
        st.metric("📋 Master Files", f"{len(core_files)}")
        if core_files:
            st.success(f"Available: {len(core_files)} files")
        else:
            st.warning("No master files found")
    
    with col3:
        # Check recent changes log
        changes_log = data_dir / "logs" / "integrated_upsert_operations.jsonl"
        if changes_log.exists():
            try:
                import os
                size_mb = os.path.getsize(changes_log) / (1024 * 1024)
                st.metric("📝 Changes Tracked", f"{size_mb:.1f} MB")
                st.success("Change tracking active")
            except:
                st.metric("📝 Changes Tracked", "Unknown")
                st.info("Change tracking available")
        else:
            st.metric("📝 Changes Tracked", "Not Started")
            st.info("No changes tracked yet")
    
    # Prepared files section
    st.markdown("### 📥 Prepared Files Analysis")
    
    if prepared_files:
        file_info = []
        for ftype, file_path in prepared_files.items():
            try:
                # Load parquet file
                df = pd.read_parquet(file_path)
                    
                stat = file_path.stat()
                file_info.append({
                    "Type": ftype.title(),
                    "File": file_path.name,
                    "Records": f"{len(df):,}",
                    "Columns": len(df.columns),
                    "Size (MB)": f"{stat.st_size / (1024*1024):.1f}",
                    "Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "Path": str(file_path)
                })
            except Exception as e:
                file_info.append({
                    "Type": ftype.title(),
                    "File": file_path.name,
                    "Records": "Error",
                    "Columns": "Error",
                    "Size (MB)": "Error",
                    "Modified": "Error",
                    "Path": str(file_path)
                })
        
        st.dataframe(file_info, width='stretch')
        
        # Show sample data
        if st.checkbox("Show sample data from prepared files"):
            selected_type = st.selectbox("Select file type to preview:", list(prepared_files.keys()))
            if selected_type:
                try:
                    file_path = prepared_files[selected_type]
                    df = pd.read_parquet(file_path)
                    st.markdown(f"**Sample data from {selected_type}:**")
                    st.dataframe(df.head(5), width='stretch')
                except Exception as e:
                    st.error(f"Error reading {selected_type}: {e}")
        
        # Debug changes for any file type
        if st.checkbox("🔍 Debug changes (show what's different)"):
            debug_file_type = st.selectbox(
                "Select file type to debug:", 
                list(prepared_files.keys()),
                help="Shows detailed analysis of what's changing in the selected file type"
            )
            if debug_file_type:
                debug_file_changes(prepared_files[debug_file_type], parquet_dir, debug_file_type)
    
    else:
        st.info("No prepared files found. Use the Download and Prepare Data pages to create prepared files.")
        return
    
    # Configuration section
    st.markdown("### ⚙️ Processing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Processing Options**")
        
        backup_files = st.checkbox("Backup master files before update", value=True)
        dry_run = st.checkbox("Dry run (preview changes only)", value=False)
        
        write_mode = st.selectbox(
            "Write mode",
            ["inplace", "rewrite"],
            index=0,
            help="Inplace: Update existing files. Rewrite: Create new files."
        )
    
    with col2:
        st.markdown("**Change Tracking**")
        
        st.info("✅ All changes (upserts and inserts) are automatically tracked in detailed logs")
        st.info("📊 Master data files will be updated with prepared data")
        st.info("🔄 Use Historical Events page for event processing after updates")
    
    # Action buttons
    st.markdown("### 🚀 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Update Master Data", type="primary", width='stretch'):
            run_master_data_update(
                company=company,
                config=config,
                prepared_files=prepared_files,
                data_dir=data_dir,
                parquet_dir=parquet_dir,
                backup_files=backup_files,
                dry_run=dry_run,
                write_mode=write_mode
            )
    
    with col2:
        if st.button("📊 Preview Changes Only", width='stretch'):
            preview_changes(prepared_files, parquet_dir, config)
    
    with col3:
        if st.button("📋 View Processing Logs", width='stretch'):
            show_processing_logs(data_dir)
    
    # Additional utility buttons
    st.markdown("### 🛠️ Data Standardization Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📅 Master Data Date Format", width='stretch', help="Standardize all date columns in master files to ISO format (excludes time-only columns)"):
            standardize_master_date_formats(parquet_dir, data_dir)
    
    with col2:
        if st.button("🕐 Restore Time Columns", width='stretch', help="Restore time-only columns from backups (fixes corrupted Call Time, etc.)"):
            restore_time_columns_from_backups(parquet_dir, data_dir)
    
    with col3:
        st.button("📊 Data Quality Check", disabled=True, width='stretch', help="Data quality analysis coming soon")
    
    # Show recent activity
    show_recent_upsert_activity(data_dir)
    
    with tab2:
        st.markdown("### 🤖 Automated Master Data Update")
        st.info("Configure automatic master data updates to run on a schedule. Updates will use the same settings configured in the Manual tab.")
        
        # Store current settings in session state
        if 'upsert_settings' not in st.session_state:
            st.session_state.upsert_settings = {}
        
        st.session_state.upsert_settings[company] = {
            'backup_files': backup_files if 'backup_files' in locals() else True,
            'dry_run': dry_run if 'dry_run' in locals() else False,
            'write_mode': write_mode if 'write_mode' in locals() else 'inplace'
        }
        
        # Show current settings
        st.markdown("**Current Update Settings:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• Prepared files found: {len(prepared_files)}")
            st.write(f"• Backup files: {'Yes' if st.session_state.upsert_settings[company]['backup_files'] else 'No'}")
        with col2:
            st.write(f"• Dry run: {'Yes' if st.session_state.upsert_settings[company]['dry_run'] else 'No'}")
            st.write(f"• Write mode: {st.session_state.upsert_settings[company]['write_mode']}")
        
        st.markdown("---")
        st.markdown("### Schedule Configuration")
        
        # Render schedule configuration
        schedule_config = render_schedule_config(
            key_prefix="upsert",
            default_interval_minutes=120  # 2 hours default
        )
        
        # Create scheduled task button
        if st.button("📅 Create Scheduled Update Task", type="primary", key="create_upsert_schedule"):
            task_config = {
                'backup_files': st.session_state.upsert_settings[company]['backup_files'],
                'dry_run': st.session_state.upsert_settings[company]['dry_run'],
                'write_mode': st.session_state.upsert_settings[company]['write_mode']
            }
            
            success = create_scheduled_task(
                task_type=TaskType.INTEGRATED_UPSERT,
                company=company,
                task_name=f"Update Master Data - {company}",
                task_description="Automated master data update with prepared files",
                schedule_config=schedule_config,
                task_config_overrides=task_config
            )
            
            if success:
                st.success("✅ Scheduled update task created successfully!")
                st.rerun()
            else:
                st.error("Failed to create scheduled task")
        
        st.markdown("---")
        
        # Show existing scheduled tasks
        render_task_manager(
            task_type=TaskType.INTEGRATED_UPSERT,
            company=company,
            task_name="Update Master Data",
            task_description="Automated master data update",
            key_context="upsert_tab",
        )
    
    with tab3:
        st.markdown("### 📊 Scheduler Status")
        render_scheduler_status(key_context="upsert")


def run_master_data_update(company: str, config: Dict[str, Any], prepared_files: Dict[str, Path],
                          data_dir: Path, parquet_dir: Path, backup_files: bool, dry_run: bool, write_mode: str):
    """Run master data update with change tracking"""
    
    st.markdown("---")
    st.markdown("### 📊 Updating Master Data")
    
    # Create progress containers
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        overall_progress = st.progress(0)
        status_text = st.empty()
        phase_metrics = st.container()
    
    try:
        # Initialize the integrated engine
        status_text.text("Initializing integrated upsert engine...")
        engine = IntegratedUpsertEngine(company, data_dir, parquet_dir)
        
        # Create upsert configuration
        upsert_config = create_upsert_config(config)
        upsert_config.update({
            'backup': backup_files,
            'dry_run': dry_run,
            'write_mode': write_mode
        })
        
        # Progress callback
        def progress_callback(message: str, progress: float):
            overall_progress.progress(progress)
            status_text.text(message)
        
        # Run the integrated workflow
        start_time = time.time()
        
        result = engine.process_prepared_files(
            prepared_files=prepared_files,
            config=upsert_config,
            progress_callback=progress_callback
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Update final progress
        overall_progress.progress(1.0)
        status_text.text("✅ Master data update completed successfully!")
        
        # Show results
        with results_container:
            st.markdown("### 📊 Update Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📝 Total Changes", f"{result.total_changes:,}")
            with col2:
                st.metric("🆕 New Records", f"{result.total_new_records:,}")
            with col3:
                st.metric("✏️ Updated Records", f"{result.total_updated_records:,}")
            
            # Phase durations
            if result.phase_durations:
                st.markdown("**Phase Performance:**")
                phase_data = []
                for phase, duration_sec in result.phase_durations.items():
                    phase_data.append({
                        "Phase": phase.replace('_', ' ').title(),
                        "Duration (s)": f"{duration_sec:.2f}",
                        "Percentage": f"{(duration_sec / result.total_duration_seconds) * 100:.1f}%"
                    })
                
                st.dataframe(phase_data, width='stretch')
            
            # File processing results
            if result.upsert_results:
                st.markdown("**File Processing Results:**")
                upsert_data = []
                for ftype, upsert_result in result.upsert_results.items():
                    upsert_data.append({
                        "File Type": ftype.title(),
                        "Examined": f"{upsert_result.examined_rows:,}",
                        "Updated": f"{upsert_result.updated_rows:,}",
                        "New": f"{upsert_result.new_rows:,}",
                        "Changes": f"{len(upsert_result.audit_changes):,}"
                    })
                
                st.dataframe(upsert_data, width='stretch')
            
            # Errors and warnings
            if result.errors:
                st.markdown("**Errors:**")
                for error in result.errors:
                    st.error(f"❌ {error}")
            
            if result.warnings:
                st.markdown("**Warnings:**")
                for warning in result.warnings:
                    st.warning(f"⚠️ {warning}")
            
            # Overall success message
            if not result.errors:
                st.balloons()
                st.success(f"🎉 Master data update completed successfully in {duration:.1f} seconds!")
                
                # Show next steps
                st.markdown("### 🎯 What's Updated")
                st.info("""
                ✅ **Master Files**: All prepared data has been upserted to master parquet files
                📝 **Change Tracking**: All changes (upserts and inserts) have been logged in detail
                🔄 **Next Steps**: Use the Historical Events page to process events and update core data
                📊 **Ready for Analysis**: Master data is updated and ready for event processing
                """)
        
        # Log the successful operation
        log_upsert_operation(data_dir, "master_data_update_success", result.to_dict(), duration)
        
    except Exception as e:
        overall_progress.progress(0)
        status_text.text("❌ Error during master data update")
        st.error(f"Master data update failed: {str(e)}")
        
        # Log the error
        log_upsert_operation(data_dir, "master_data_update_error", {"error": str(e)}, 0)


def preview_changes(prepared_files: Dict[str, Path], parquet_dir: Path, config: Any):
    """Preview what changes would be made without actually applying them - uses same logic as actual upsert"""
    
    st.markdown("---")
    st.markdown("### 📊 Change Preview")
    
    # Convert config to dictionary format (same as actual upsert)
    from datahound.upsert.integrated_engine import create_upsert_config
    try:
        upsert_config = create_upsert_config(config)
    except Exception as e:
        st.error(f"Configuration error: {e}")
        return
    
    preview_data = []
    
    for ftype, prepared_path in prepared_files.items():
        try:
            # Load prepared data (parquet only)
            prepared_df = pd.read_parquet(prepared_path)
            
            # Find corresponding master file
            master_files = list(parquet_dir.glob(f"{ftype.capitalize()}.parquet"))
            if not master_files:
                preview_data.append({
                    "Type": ftype.title(),
                    "Status": "New Master File",
                    "Records": len(prepared_df),
                    "New": len(prepared_df),
                    "Updates": 0,
                    "Changes": 0,
                    "Skip": False
                })
                continue
            
            master_df = pd.read_parquet(master_files[0])
            
            # Determine ID column (same logic as upsert engine)
            id_col = upsert_config.get('id_columns', {}).get(ftype)
            if not id_col:
                id_col = f"{ftype.rstrip('s').capitalize()} ID"
            
            if id_col not in prepared_df.columns or id_col not in master_df.columns:
                preview_data.append({
                    "Type": ftype.title(),
                    "Status": "ID Column Not Found",
                    "Records": len(prepared_df),
                    "New": "Unknown",
                    "Updates": "Unknown",
                    "Changes": "Unknown",
                    "Skip": True
                })
                continue
            
            # Use SAME comparison logic as actual upsert engine
            from datahound.upsert.engine import _canonicalize_series_for_compare
            
            # Normalize data same way as upsert engine
            master_df[id_col] = master_df[id_col].astype("string")
            prepared_df[id_col] = prepared_df[id_col].astype("string")
            master_df[id_col] = _canonicalize_series_for_compare(master_df[id_col], id_col, id_col)
            prepared_df[id_col] = _canonicalize_series_for_compare(prepared_df[id_col], id_col, id_col)
            
            # Merge to find changes (same as upsert engine)
            merged = master_df.merge(prepared_df, how="outer", on=id_col, indicator=True, suffixes=("_x", "_y"))
            
            new_records = len(merged[merged["_merge"] == "right_only"])
            existing_records = len(merged[merged["_merge"] == "both"])
            
            # Count changes using SAME logic as upsert engine
            changes = 0
            both_df = merged[merged["_merge"] == "both"]
            for col in master_df.columns:
                if col == id_col:
                    continue
                master_col = f"{col}_x"
                prepared_col = f"{col}_y"
                if master_col in both_df.columns and prepared_col in both_df.columns:
                    # Use SAME canonicalization as upsert engine
                    lhs = _canonicalize_series_for_compare(both_df[master_col], col, id_col)
                    rhs = _canonicalize_series_for_compare(both_df[prepared_col], col, id_col)
                    # FIX: Reset indices to ensure proper alignment for comparison
                    lhs = lhs.reset_index(drop=True)
                    rhs = rhs.reset_index(drop=True)
                    different = (lhs != rhs)
                    changes += different.sum()
            
            # Determine if file should be skipped
            should_skip = new_records == 0 and changes == 0
            
            preview_data.append({
                "Type": ftype.title(),
                "Status": "⏭️ Skip (No Changes)" if should_skip else "Ready",
                "Records": len(prepared_df),
                "New": new_records,
                "Updates": existing_records,
                "Changes": changes,
                "Skip": should_skip
            })
            
        except Exception as e:
            preview_data.append({
                "Type": ftype.title(),
                "Status": f"Error: {str(e)}",
                "Records": "Unknown",
                "New": "Unknown",
                "Updates": "Unknown", 
                "Changes": "Unknown",
                "Skip": True
            })
    
    st.dataframe(preview_data, width='stretch')
    
    # Summary
    total_new = sum(row["New"] for row in preview_data if isinstance(row["New"], int))
    total_changes = sum(row["Changes"] for row in preview_data if isinstance(row["Changes"], int))
    files_to_skip = sum(1 for row in preview_data if row.get("Skip", False))
    files_to_process = len([r for r in preview_data if r["Status"] == "Ready"])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🆕 Total New Records", f"{total_new:,}")
    with col2:
        st.metric("📝 Total Changes", f"{total_changes:,}")
    with col3:
        st.metric("📁 Files to Process", files_to_process)
    with col4:
        st.metric("⏭️ Files to Skip", files_to_skip)
        
    if files_to_skip > 0:
        st.info(f"💡 **Optimization**: {files_to_skip} file(s) will be skipped during processing (no changes detected)")


def show_processing_logs(data_dir: Path):
    """Show recent processing logs"""
    
    st.markdown("---")
    st.markdown("### 📋 Recent Processing Logs")
    
    log_file = data_dir / "logs" / "integrated_upsert_log.jsonl"
    
    if not log_file.exists():
        st.info("No processing logs found.")
        return
    
    try:
        # Read recent log entries
        recent_entries = []
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Get last 20 entries
            for line in lines[-20:]:
                try:
                    entry = json.loads(line.strip())
                    recent_entries.append(entry)
                except:
                    continue
        
        if not recent_entries:
            st.info("No recent processing logs found.")
            return
        
        # Display recent entries
        for entry in reversed(recent_entries):  # Most recent first
            timestamp = entry.get('timestamp', '')[:19].replace('T', ' ')
            level = entry.get('level', 'info')
            message = entry.get('message', '')
            
            level_colors = {
                "info": "🔵",
                "warning": "🟡", 
                "error": "🔴",
                "success": "🟢"
            }
            
            level_icon = level_colors.get(level, "⚫")
            
            with st.expander(f"{level_icon} {message}", expanded=False):
                st.markdown(f"**Time:** {timestamp}")
                st.markdown(f"**Level:** {level.title()}")
                
                # Show additional details if available
                details = {k: v for k, v in entry.items() 
                          if k not in ['timestamp', 'level', 'message', 'company', 'component']}
                if details:
                    st.json(details)
    
    except Exception as e:
        st.error(f"Error reading processing logs: {e}")


def show_recent_upsert_activity(data_dir: Path):
    """Show recent upsert activity"""
    
    st.markdown("---")
    st.markdown("### 📈 Recent Activity")
    
    # Check for recent upsert operations
    upsert_log = data_dir / "logs" / "integrated_upsert_log.jsonl"
    apply_log = data_dir / "logs" / "apply_log.jsonl"
    
    recent_operations = []
    
    # Read from both log files
    for log_file in [upsert_log, apply_log]:
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:  # Last 10 entries
                        try:
                            entry = json.loads(line.strip())
                            recent_operations.append(entry)
                        except:
                            continue
            except:
                continue
    
    if recent_operations:
        # Sort by timestamp
        recent_operations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recent_count = len([op for op in recent_operations[-24:] if 'upsert' in op.get('message', '').lower()])
            st.metric("🔄 Recent Operations", recent_count)
        
        with col2:
            success_count = len([op for op in recent_operations[-24:] if op.get('level') in ['info', 'success']])
            st.metric("✅ Successful", success_count)
        
        with col3:
            error_count = len([op for op in recent_operations[-24:] if op.get('level') == 'error'])
            st.metric("❌ Errors", error_count)
        
        # Show recent operations
        st.markdown("**Recent Operations:**")
        for op in recent_operations[:5]:  # Show last 5
            timestamp = op.get('timestamp', '')[:16].replace('T', ' ')
            level = op.get('level', 'info')
            message = op.get('message', 'Unknown operation')
            
            if level == 'error':
                st.error(f"{timestamp}: {message}")
            elif level == 'warning':
                st.warning(f"{timestamp}: {message}")
            else:
                st.success(f"{timestamp}: {message}")
    else:
        st.info("No recent upsert activity found.")


def debug_file_changes(prepared_file: Path, parquet_dir: Path, file_type: str):
    """Debug function to show what's causing changes in any file type"""
    
    st.markdown(f"#### 🔍 {file_type.title()} Changes Analysis")
    
    try:
        # Load prepared parquet file
        prepared_df = pd.read_parquet(prepared_file)
            
        master_files = list(parquet_dir.glob(f"{file_type.capitalize()}.parquet"))
        
        if not master_files:
            st.error(f"No master {file_type.capitalize()}.parquet file found")
            return
        
        master_df = pd.read_parquet(master_files[0])
        
        # Debug: Show file info
        st.write(f"**Debug Info for {file_type}:**")
        st.write(f"Prepared file: {prepared_file.name}")
        st.write(f"Prepared columns: {list(prepared_df.columns)[:10]}...")  # Show first 10
        st.write(f"Master file: {master_files[0].name}")
        st.write(f"Master columns: {list(master_df.columns)[:10]}...")  # Show first 10
        
        # Find ID column (dynamic based on file type)
        id_col = None
        possible_id_cols = [f"{file_type.rstrip('s').capitalize()} ID", "Customer ID", "ID", "id"]
        
        st.write(f"Looking for ID columns: {possible_id_cols}")
        
        for possible_col in possible_id_cols:
            in_prepared = possible_col in prepared_df.columns
            in_master = possible_col in master_df.columns
            st.write(f"  {possible_col}: prepared={in_prepared}, master={in_master}")
            
            if in_prepared and in_master:
                id_col = possible_col
                break
        
        if not id_col:
            st.error(f"Could not find matching ID column for {file_type}")
            st.write("**Available columns in prepared:**")
            st.write(list(prepared_df.columns))
            st.write("**Available columns in master:**")
            st.write(list(master_df.columns))
            return
        
        st.success(f"Using ID column: {id_col}")
        
        # Merge data
        st.write(f"**Merging data on {id_col}...**")
        master_df[id_col] = master_df[id_col].astype(str)
        prepared_df[id_col] = prepared_df[id_col].astype(str)
        
        st.write(f"Master records: {len(master_df)}")
        st.write(f"Prepared records: {len(prepared_df)}")
        
        merged = master_df.merge(prepared_df, on=id_col, how="inner", suffixes=("_master", "_prepared"))
        
        st.write(f"Merged records: {len(merged)}")
        
        if len(merged) == 0:
            st.warning(f"No matching records found between master and prepared {file_type} files!")
            st.write("**Sample Master IDs:**")
            master_sample = list(master_df[id_col].head(5))
            st.write(master_sample)
            st.write("**Sample Prepared IDs:**")
            prepared_sample = list(prepared_df[id_col].head(5))
            st.write(prepared_sample)
            
            # Check if this is expected (all new records)
            total_prepared = len(prepared_df)
            st.info(f"💡 This might be expected if all {total_prepared} prepared records are NEW {file_type} records (not updates to existing records)")
            return
        
        # Analyze first few records
        st.markdown(f"**Sample Change Analysis (first 3 {file_type} records):**")
        
        sample_records = merged.head(3)
        
        st.write(f"**Processing {len(sample_records)} sample records...**")
        
        for idx, row in sample_records.iterrows():
            try:
                record_id = row[id_col]
                st.markdown(f"**{id_col}: {record_id}**")
                
                changes_found = []
                
                for col in master_df.columns:
                    if col == id_col:
                        continue
                    
                    master_col = f"{col}_master"
                    prepared_col = f"{col}_prepared"
                    
                    if master_col in row.index and prepared_col in row.index:
                        try:
                            # Use SAME canonicalization logic as actual upsert engine
                            from datahound.upsert.engine import _canonicalize_series_for_compare
                            
                            master_series = pd.Series([row[master_col]])
                            prepared_series = pd.Series([row[prepared_col]])
                            
                            # Use consistent comparison logic
                            lhs = _canonicalize_series_for_compare(master_series, col, id_col)
                            rhs = _canonicalize_series_for_compare(prepared_series, col, id_col)
                            is_diff_value = lhs.iloc[0] != rhs.iloc[0]
                            
                            if is_diff_value:  # If truly different using engine logic
                                master_val = str(row[master_col]) if pd.notna(row[master_col]) else ""
                                prepared_val = str(row[prepared_col]) if pd.notna(row[prepared_col]) else ""
                                
                                changes_found.append({
                                    "Column": col,
                                    "Master": master_val[:50] + "..." if len(master_val) > 50 else master_val,
                                    "Prepared": prepared_val[:50] + "..." if len(prepared_val) > 50 else prepared_val,
                                    "Type": "Real Change"
                                })
                        except Exception as col_error:
                            st.error(f"Error comparing column {col}: {col_error}")
                            continue
                
                if changes_found:
                    st.dataframe(pd.DataFrame(changes_found), width='stretch')
                else:
                    st.success(f"No changes found for this {file_type} record")
                
                st.markdown("---")
                
            except Exception as row_error:
                st.error(f"Error processing row {idx}: {row_error}")
                continue
        
        # Summary statistics - using SAME logic as preview and actual processing
        st.markdown("**Change Summary by Column:**")
        st.write(f"**Analyzing {len(master_df.columns)} columns for changes...**")
        
        # Import the same comparison function used by preview and processing
        from datahound.upsert.engine import _canonicalize_series_for_compare
        
        column_changes = {}
        
        for col in master_df.columns:
            if col == id_col:
                continue
            
            master_col = f"{col}_master"
            prepared_col = f"{col}_prepared"
            
            if master_col in merged.columns and prepared_col in merged.columns:
                try:
                    # Use SAME canonicalization logic as actual upsert engine
                    lhs = _canonicalize_series_for_compare(merged[master_col], col, id_col)
                    rhs = _canonicalize_series_for_compare(merged[prepared_col], col, id_col)
                    # FIX: Reset indices to ensure proper alignment for comparison
                    lhs = lhs.reset_index(drop=True)
                    rhs = rhs.reset_index(drop=True)
                    different = (lhs != rhs)
                    changes_count = different.sum()
                    
                    if changes_count > 0:
                        column_changes[col] = changes_count
                        
                except Exception as col_summary_error:
                    st.warning(f"Error analyzing column {col}: {col_summary_error}")
                    continue
        
        if column_changes:
            # Sort by most changes
            sorted_changes = sorted(column_changes.items(), key=lambda x: x[1], reverse=True)
            
            change_summary = []
            for col, count in sorted_changes[:10]:  # Top 10 columns with most changes
                change_summary.append({
                    "Column": col,
                    "Records Changed": f"{count:,}",
                    "Percentage": f"{(count/len(merged)*100):.1f}%"
                })
            
            st.dataframe(pd.DataFrame(change_summary), width='stretch')
            
            # Special debugging for date columns showing changes
            for col, count in sorted_changes[:3]:  # Top 3 problematic columns
                if 'date' in col.lower() and count > 0:
                    st.markdown(f"#### 🔍 Date Column Debug: {col}")
                    debug_date_column_differences(merged, col, id_col)
        
    except Exception as e:
        st.error(f"Error analyzing customer changes: {e}")


def debug_date_column_differences(merged: pd.DataFrame, col: str, id_col: str):
    """Debug function to analyze date column differences in detail"""
    
    try:
        master_col = f"{col}_master"
        prepared_col = f"{col}_prepared"
        
        # Debug info about the merged DataFrame
        st.write(f"**Debug DataFrame Info:**")
        st.write(f"Columns available: {list(merged.columns)[:10]}...")  # Show first 10 columns
        st.write(f"Records in merged data: {len(merged)}")
        
        # Find records where this column differs
        from datahound.upsert.engine import _canonicalize_series_for_compare
        
        # Check if required columns exist
        if master_col not in merged.columns:
            st.error(f"Master column '{master_col}' not found in merge data")
            return
        if prepared_col not in merged.columns:
            st.error(f"Prepared column '{prepared_col}' not found in merge data")
            return
        
        # Since debug uses inner join, all records are "both" records
        both_df = merged.copy()
        
        # Apply same canonicalization as the engine
        lhs = _canonicalize_series_for_compare(both_df[master_col], col, id_col)
        rhs = _canonicalize_series_for_compare(both_df[prepared_col], col, id_col)
        
        # FIX: Reset indices to ensure proper alignment for comparison
        lhs = lhs.reset_index(drop=True)
        rhs = rhs.reset_index(drop=True)
        different_mask = (lhs != rhs)
        
        # DEBUG: Show bulk comparison details
        st.write("**🔍 Bulk Comparison Debug:**")
        st.write(f"LHS dtype: {lhs.dtype}, RHS dtype: {rhs.dtype}")
        st.write(f"LHS first 5 values: {lhs.head().tolist()}")
        st.write(f"RHS first 5 values: {rhs.head().tolist()}")
        st.write(f"Are they equal (bulk): {(lhs == rhs).sum()}/{len(lhs)} equal")
        st.write(f"Are they different (bulk): {different_mask.sum()}/{len(lhs)} different")
        
        # Check for data type issues
        if lhs.dtype != rhs.dtype:
            st.warning(f"⚠️ Data type mismatch: {lhs.dtype} vs {rhs.dtype}")
        
        # Check for NaN issues
        lhs_nulls = lhs.isna().sum()
        rhs_nulls = rhs.isna().sum()
        if lhs_nulls > 0 or rhs_nulls > 0:
            st.info(f"Null values: LHS={lhs_nulls}, RHS={rhs_nulls}")
        
        different_records = both_df[different_mask].head(10)  # Show first 10 different records
        
        if len(different_records) == 0:
            st.success("No differences found in this date column after canonicalization")
            return
            
        st.write(f"**Found {different_mask.sum()} different records. Showing first 10:**")
        
        # Show sample differences
        sample_data = []
        for idx, row in different_records.iterrows():
            try:
                master_val = row[master_col]
                prepared_val = row[prepared_col]
                master_canonical = _canonicalize_series_for_compare(pd.Series([master_val]), col, id_col).iloc[0]
                prepared_canonical = _canonicalize_series_for_compare(pd.Series([prepared_val]), col, id_col).iloc[0]
                
                sample_data.append({
                    "ID": row[id_col],
                    "Master Raw": str(master_val)[:30],
                    "Prepared Raw": str(prepared_val)[:30],
                    "Master Canonical": str(master_canonical)[:30],
                    "Prepared Canonical": str(prepared_canonical)[:30],
                    "Same After Canonicalization": master_canonical == prepared_canonical
                })
            except Exception as row_error:
                st.error(f"Error processing row {idx}: {row_error}")
                continue
        
        if sample_data:
            st.dataframe(pd.DataFrame(sample_data), width='stretch')
        
        # Analyze patterns
        st.markdown("**Pattern Analysis:**")
        try:
            master_patterns = both_df[master_col].astype(str).str[:20].value_counts().head(5)
            prepared_patterns = both_df[prepared_col].astype(str).str[:20].value_counts().head(5)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Master File Patterns:**")
                st.write(master_patterns)
            with col2:
                st.markdown("**Prepared File Patterns:**")
                st.write(prepared_patterns)
        except Exception as pattern_error:
            st.error(f"Error analyzing patterns: {pattern_error}")
            
    except Exception as e:
        st.error(f"Error in date column debugging: {str(e)}")
        st.error(f"Column: {col}, ID Column: {id_col}")
        st.error(f"DataFrame shape: {merged.shape}")
        st.error(f"DataFrame columns: {list(merged.columns) if hasattr(merged, 'columns') else 'No columns'}")


def standardize_master_date_formats(parquet_dir: Path, data_dir: Path):
    """Standardize all date columns in master files to ISO datetime format"""
    
    st.markdown("---")
    st.markdown("### 📅 Standardizing Master Data Date Formats")
    
    progress_container = st.container()
    
    with progress_container:
        overall_progress = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
    
    try:
        status_text.text("Scanning master files for date columns...")
        
        # Find all master parquet files
        master_files = [f for f in parquet_dir.glob("*.parquet") if not f.name.startswith("customer_")]
        
        overall_progress.progress(0.1)
        
        results = []
        total_files = len(master_files)
        
        for i, master_file in enumerate(master_files):
            try:
                status_text.text(f"Processing {master_file.name}...")
                
                # Load master file
                df = pd.read_parquet(master_file)
                original_shape = df.shape
                
                # Find date columns (improved detection)
                date_columns = find_date_columns(df)
                
                if not date_columns:
                    results.append({
                        "File": master_file.name,
                        "Status": "No date columns found",
                        "Date Columns": 0,
                        "Records Updated": 0
                    })
                    continue
                
                # Track changes
                columns_updated = 0
                total_values_updated = 0
                
                # Convert each date column to ISO format
                for col in date_columns:
                    if col in df.columns:
                        original_values = df[col].copy()
                        
                        # Convert to ISO datetime format with warnings suppressed
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="Could not infer format")
                            
                            # Parse dates
                            parsed_dates = pd.to_datetime(df[col], errors='coerce')
                            
                            # Format to ISO datetime string
                            iso_formatted = parsed_dates.dt.strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Replace NaT with empty string
                            iso_formatted = iso_formatted.fillna("")
                            
                            # Count actual changes
                            changes_in_col = (original_values.astype(str) != iso_formatted.astype(str)).sum()
                            
                            if changes_in_col > 0:
                                df[col] = iso_formatted
                                columns_updated += 1
                                total_values_updated += changes_in_col
                
                # Save updated file if changes were made
                if columns_updated > 0:
                    # Create backup first
                    backup_dir = parquet_dir / "backups"
                    backup_dir.mkdir(exist_ok=True)
                    backup_file = backup_dir / f"{master_file.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                    
                    # Copy original to backup
                    import shutil
                    shutil.copy2(master_file, backup_file)
                    
                    # Save updated file
                    df.to_parquet(master_file, index=False)
                    
                    results.append({
                        "File": master_file.name,
                        "Status": "✅ Updated",
                        "Date Columns": len(date_columns),
                        "Columns Updated": columns_updated,
                        "Values Updated": total_values_updated,
                        "Backup": backup_file.name
                    })
                else:
                    results.append({
                        "File": master_file.name,
                        "Status": "✅ Already ISO format",
                        "Date Columns": len(date_columns),
                        "Columns Updated": 0,
                        "Values Updated": 0
                    })
                
                # Update progress
                file_progress = 0.1 + (0.8 * (i + 1) / total_files)
                overall_progress.progress(file_progress)
                
            except Exception as e:
                results.append({
                    "File": master_file.name,
                    "Status": f"❌ Error: {str(e)}",
                    "Date Columns": 0,
                    "Columns Updated": 0,
                    "Values Updated": 0
                })
        
        # Complete
        overall_progress.progress(1.0)
        status_text.text("✅ Master data date format standardization completed!")
        
        # Show results
        with results_container:
            st.markdown("### 📊 Standardization Results")
            
            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, width='stretch')
                
                # Summary metrics
                total_updated = sum(r.get("Values Updated", 0) for r in results)
                files_updated = sum(1 for r in results if r.get("Columns Updated", 0) > 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📁 Files Updated", files_updated)
                with col2:
                    st.metric("📅 Values Standardized", f"{total_updated:,}")
                with col3:
                    st.metric("📂 Backups Created", files_updated)
                
                if total_updated > 0:
                    st.success(f"✅ Successfully standardized {total_updated:,} date values to ISO format!")
                    st.info("💡 **Next Steps**: Re-run preparation to ensure prepared files match the new ISO format")
                else:
                    st.info("ℹ️ All master files already use consistent date formatting")
            
        # Log the operation
        log_upsert_operation(data_dir, "standardize_master_dates", {
            "files_processed": len(master_files),
            "files_updated": len([r for r in results if r.get("Columns Updated", 0) > 0]),
            "total_values_updated": sum(r.get("Values Updated", 0) for r in results)
        }, 0)
        
    except Exception as e:
        overall_progress.progress(0)
        status_text.text("❌ Error during date format standardization")
        st.error(f"Date format standardization failed: {str(e)}")


def restore_time_columns_from_backups(parquet_dir: Path, data_dir: Path):
    """Restore time-only columns from backup files"""
    
    st.markdown("---")
    st.markdown("### 🕐 Restoring Time Columns from Backups")
    
    progress_container = st.container()
    
    with progress_container:
        overall_progress = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
    
    try:
        status_text.text("Scanning for backup files...")
        
        # Find backup directory
        backup_dir = parquet_dir / "backups"
        if not backup_dir.exists():
            st.error("No backup directory found. Cannot restore time columns.")
            return
        
        # Find backup files
        backup_files = list(backup_dir.glob("*_backup_*.parquet"))
        
        if not backup_files:
            st.error("No backup files found. Cannot restore time columns.")
            return
        
        overall_progress.progress(0.1)
        
        results = []
        
        # Process each backup file
        for i, backup_file in enumerate(backup_files):
            try:
                # Determine corresponding master file
                master_name = backup_file.name.split('_backup_')[0] + '.parquet'
                master_file = parquet_dir / master_name
                
                if not master_file.exists():
                    results.append({
                        "File": master_name,
                        "Status": "❌ Master file not found",
                        "Time Columns Restored": 0
                    })
                    continue
                
                status_text.text(f"Processing {master_name}...")
                
                # Load both files
                backup_df = pd.read_parquet(backup_file)
                current_df = pd.read_parquet(master_file)
                
                # Find time-only columns in backup
                time_columns = []
                for col in backup_df.columns:
                    if 'time' in col.lower() or 'duration' in col.lower() or 'length' in col.lower():
                        # Check if values look like time-only
                        sample_values = backup_df[col].dropna().head(5).astype(str).tolist()
                        
                        time_patterns = [
                            r'^\d{1,2}:\d{2} (AM|PM)$',
                            r'^\d{1,2}:\d{2}:\d{2}$',
                            r'^\d{1,2}:\d{2}$',
                        ]
                        
                        import re
                        time_count = sum(1 for val in sample_values 
                                       if any(re.match(pattern, str(val).strip()) for pattern in time_patterns))
                        
                        if time_count >= len(sample_values) * 0.6:
                            time_columns.append(col)
                
                # Restore time columns from backup
                columns_restored = 0
                if time_columns:
                    for col in time_columns:
                        if col in backup_df.columns and col in current_df.columns:
                            current_df[col] = backup_df[col]
                            columns_restored += 1
                    
                    # Save updated master file
                    if columns_restored > 0:
                        current_df.to_parquet(master_file, index=False)
                
                results.append({
                    "File": master_name,
                    "Status": "✅ Processed",
                    "Time Columns Found": len(time_columns),
                    "Time Columns Restored": columns_restored,
                    "Restored Columns": time_columns if time_columns else []
                })
                
                # Update progress
                file_progress = 0.1 + (0.8 * (i + 1) / len(backup_files))
                overall_progress.progress(file_progress)
                
            except Exception as e:
                results.append({
                    "File": backup_file.name,
                    "Status": f"❌ Error: {str(e)}",
                    "Time Columns Restored": 0
                })
        
        # Complete
        overall_progress.progress(1.0)
        status_text.text("✅ Time column restoration completed!")
        
        # Show results
        with results_container:
            st.markdown("### 📊 Restoration Results")
            
            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, width='stretch')
                
                # Summary
                total_restored = sum(r.get("Time Columns Restored", 0) for r in results)
                files_updated = sum(1 for r in results if r.get("Time Columns Restored", 0) > 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📁 Files Updated", files_updated)
                with col2:
                    st.metric("🕐 Time Columns Restored", total_restored)
                with col3:
                    st.metric("📂 Backups Used", len(backup_files))
                
                if total_restored > 0:
                    st.success(f"✅ Successfully restored {total_restored} time columns!")
                    st.info("💡 Time columns like 'Call Time' should now show original format (7:48 PM)")
                else:
                    st.info("ℹ️ No time columns needed restoration")
        
        # Log the operation
        log_upsert_operation(data_dir, "restore_time_columns", {
            "backup_files_found": len(backup_files),
            "files_processed": len(results),
            "time_columns_restored": sum(r.get("Time Columns Restored", 0) for r in results)
        }, 0)
        
    except Exception as e:
        overall_progress.progress(0)
        status_text.text("❌ Error during time column restoration")
        st.error(f"Time column restoration failed: {str(e)}")


def log_upsert_operation(data_dir: Path, operation: str, details: Dict[str, Any], duration: float):
    """Log upsert operation to JSONL file"""
    
    log_file = data_dir / "logs" / "integrated_upsert_operations.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    log_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": operation,
        "duration_seconds": duration,
        "details": details
    }
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception:
        pass  # Don't let logging errors break the main process


if __name__ == "__main__":
    main()
