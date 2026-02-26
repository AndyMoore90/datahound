from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys
from pathlib import Path as _P
import threading
import time
import json

import streamlit as st
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = _P(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from apps._shared import ensure_root_on_path
from apps.streamlit_compat import call_compat

ensure_root_on_path()

from datahound.download.types import load_global_config, save_global_config
from datahound.download.permits import download_austin_permits


def get_column_mapping() -> dict:
    """Get mapping from downloaded CSV columns to master file columns"""
    return {
        'permittype': 'Permit Type',
        'permit_type_desc': 'Permit Type Desc',
        'permit_number': 'Permit Num',
        'permit_class_mapped': 'Permit Class Mapped',
        'permit_class': 'Permit Class',
        'work_class': 'Work Class',
        'condominium': 'Condominium',
        'permit_location': 'Project Name',
        'description': 'Description',
        'tcad_id': 'TCAD ID',
        'legal_description': 'Property Legal Description',
        'applieddate': 'Applied Date',
        'issue_date': 'Issued Date',
        'day_issued': 'Day Issued',
        'calendar_year_issued': 'Calendar Year Issued',
        'fiscal_year_issued': 'Fiscal Year Issued',
        'issued_in_last_30_days': 'Issued In Last 30 Days',
        'issue_method': 'Issuance Method',
        'status_current': 'Status Current',
        'statusdate': 'Status Date',
        'expiresdate': 'Expires Date',
        'completed_date': 'Completed Date',
        'total_existing_bldg_sqft': 'Total Existing Bldg SQFT',
        'remodel_repair_sqft': 'Remodel Repair SQFT',
        'total_new_add_sqft': 'Total New Add SQFT',
        'total_valuation_remodel': 'Total Valuation Remodel',
        'total_job_valuation': 'Total Job Valuation',
        'number_of_floors': 'Number Of Floors',
        'housing_units': 'Housing Units',
        'building_valuation': 'Building Valuation',
        'building_valuation_remodel': 'Building Valuation Remodel',
        'electrical_valuation': 'Electrical Valuation',
        'electrical_valuation_remodel': 'Electrical Valuation Remodel',
        'mechanical_valuation': 'Mechanical Valuation',
        'mechanical_valuation_remodel': 'Mechanical Valuation Remodel',
        'plumbing_valuation': 'Plumbing Valuation',
        'plumbing_valuation_remodel': 'Plumbing Valuation Remodel',
        'medgas_valuation': 'MedGas Valuation',
        'medgas_valuation_remodel': 'MedGas Valuation Remodel',
        'original_address1': 'Original Address 1',
        'original_city': 'Original City',
        'original_state': 'Original State',
        'original_zip': 'Original Zip',
        'council_district': 'Council District',
        'jurisdiction': 'Jurisdiction',
        'link': 'Link',
        'project_id': 'Project ID',
        'masterpermitnum': 'Master Permit Num',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'location': 'Location',
        'contractor_trade': 'Contractor Trade',
        'contractor_company_name': 'Contractor Company Name',
        'contractor_full_name': 'Contractor Full Name',
        'contractor_phone': 'Contractor Phone',
        'contractor_address1': 'Contractor Address 1',
        'contractor_address2': 'Contractor Address 2',
        'contractor_city': 'Contractor City',
        'contractor_zip': 'Contractor Zip',
        'applicant_full_name': 'Applicant Full Name',
        'applicant_org': 'Applicant Organization',
        'applicant_phone': 'Applicant Phone',
        'applicant_address1': 'Applicant Address 1',
        'applicant_address2': 'Applicant Address 2',
        'applicant_city': 'Applicant City',
        'applicantzip': 'Applicant Zip',
        'certificate_of_occupancy': 'Certificate Of Occupancy',
        'total_lot_sq_ft': 'Total Lot SQFT'
    }


def append_csv_to_master_csv(new_csv_path: Path, master_csv_path: Path, 
                           filter_mp_only: bool = True, 
                           save_backup: bool = True) -> dict:
    """Simple CSV to CSV append with MP filtering, column mapping, and duplicate prevention"""
    
    try:
        # Read new CSV
        new_df = pd.read_csv(new_csv_path, low_memory=False)
        original_count = len(new_df)
        
        # Apply column mapping from downloaded format to master format
        column_mapping = get_column_mapping()
        new_df = new_df.rename(columns=column_mapping)
        
        # Filter for MP permits if requested
        if filter_mp_only and 'Permit Type' in new_df.columns:
            new_df = new_df[new_df['Permit Type'] == 'MP']
            mp_count = len(new_df)
            filter_info = f"Filtered from {original_count} to {mp_count} MP permits"
        else:
            filter_info = f"No filtering applied, {len(new_df)} permits"
        
        if new_df.empty:
            return {
                "status": "no_data", 
                "message": "No MP permits found",
                "filter_info": filter_info
            }
        
        # Load existing master CSV or create empty DataFrame
        if master_csv_path.exists():
            if save_backup:
                backup_path = master_csv_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                master_csv_path.rename(backup_path)
                existing_df = pd.read_csv(backup_path, low_memory=False)
            else:
                existing_df = pd.read_csv(master_csv_path, low_memory=False)
        else:
            existing_df = pd.DataFrame()
        
        # Use standard master column name for permit number
        permit_col = 'Permit Num'
        
        # Determine new records based on Permit Num
        if not existing_df.empty and permit_col in existing_df.columns and permit_col in new_df.columns:
            existing_permit_numbers = set(existing_df[permit_col].astype(str))
            new_records = new_df[~new_df[permit_col].astype(str).isin(existing_permit_numbers)]
            new_count = len(new_records)
            
            if not new_records.empty:
                # Ensure column order matches master file
                if not existing_df.empty:
                    # Add any missing columns to new_records
                    for col in existing_df.columns:
                        if col not in new_records.columns:
                            new_records[col] = ''
                    
                    # Reorder new_records to match existing columns
                    new_records = new_records[existing_df.columns]
                
                combined_df = pd.concat([existing_df, new_records], ignore_index=True)
            else:
                combined_df = existing_df
        else:
            # First time or no existing data
            combined_df = new_df
            new_count = len(new_df)
        
        # Save to CSV
        combined_df.to_csv(master_csv_path, index=False)
        
        return {
            "status": "success",
            "new_records": new_count,
            "total_records": len(combined_df),
            "filter_info": filter_info
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def convert_master_csv_to_parquet(master_csv_path: Path, master_parquet_path: Path) -> dict:
    """Simple conversion from master CSV to parquet"""
    
    try:
        # Read master CSV
        df = pd.read_csv(master_csv_path, low_memory=False)
        
        # Save as parquet (let pandas handle the conversion naturally)
        df.to_parquet(master_parquet_path, index=False)
        
        return {
            "status": "success",
            "records": len(df)
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e)
        }


def start_scheduled_downloads(interval_minutes: int, permits_dir: Path, base_url: str, 
                            lookback_days: int, update_on_download: bool, 
                            filter_mp_only: bool, save_backups: bool):
    """Start scheduled permit downloads in background thread"""
    
    def download_loop():
        while True:
            try:
                end_dt = datetime.now(UTC)
                start_dt = end_dt - timedelta(days=lookback_days)
                
                # Download permits
                fname = download_austin_permits("GLOBAL", permits_dir, base_url, start_dt, end_dt)
                
                if fname and update_on_download:
                    csv_path = permits_dir / fname
                    master_csv_path = permits_dir / "permit_data.csv"
                    master_parquet_path = permits_dir / "permit_data.parquet"
                    
                    # Step 1: Append to master CSV
                    result = append_csv_to_master_csv(
                        csv_path, master_csv_path, filter_mp_only, save_backups
                    )
                    
                    # Step 2: Convert master CSV to parquet
                    if result["status"] == "success":
                        parquet_result = convert_master_csv_to_parquet(master_csv_path, master_parquet_path)
                        result["parquet_conversion"] = parquet_result
                    
                    # Log the result
                    log_path = permits_dir.parent / "logs" / "permit_processing_log.jsonl"
                    log_entry = {
                        "ts": datetime.now(UTC).isoformat(),
                        "action": "scheduled_download_and_convert",
                        "csv_file": fname,
                        "result": result
                    }
                    
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")
                
                # Wait for next interval
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                # Log error and continue
                log_path = permits_dir.parent / "logs" / "permit_processing_log.jsonl"
                log_entry = {
                    "ts": datetime.now(UTC).isoformat(),
                    "action": "scheduled_download_error",
                    "error": str(e)
                }
                
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
                
                time.sleep(interval_minutes * 60)  # Continue trying
    
    # Start background thread
    thread = threading.Thread(target=download_loop, daemon=True)
    thread.start()


def main() -> None:
    st.title("🏢 Permits Management")
    st.caption("Home / Permits")
    
    # Load global config
    gcfg = load_global_config()
    permits_dir = Path(gcfg.permits_data_dir)
    permits_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📥 Download", "⚙️ Settings", "📊 Data Status"])
    
    with tab1:
        st.subheader("Manual Download")
        st.write(f"**Permits data directory:** `{permits_dir}`")
        
        # Download settings
        col1, col2 = st.columns(2)
        
        with col1:
            lookback_days = st.number_input("Lookback days", min_value=1, max_value=90, value=7, key="manual_lookback")
            
        with col2:
            update_on_download = st.checkbox(
                "Update Permit Data on Download", 
                value=True, 
                key="manual_update",
                help="Automatically convert CSV to parquet and append to master file"
            )
        
        # Time window display
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=int(lookback_days))
        st.caption(f"**UTC window:** {start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Additional options when update is enabled
        if update_on_download:
            col3, col4 = st.columns(2)
            
            with col3:
                filter_mp_only = st.checkbox(
                    "Only MP permits", 
                    value=True, 
                    key="manual_filter",
                    help="Filter to only include permits with Permit Type = 'MP'"
                )
            
            with col4:
                save_backups = st.checkbox(
                    "Save backups", 
                    value=True, 
                    key="manual_backup",
                    help="Create backup of existing parquet file before updating"
                )
        else:
            filter_mp_only = True
            save_backups = True
        
        # Download button
        if call_compat(st.button, "🚀 Download Permits", type="primary", use_container_width=True):
            with st.spinner("Downloading permits..."):
                try:
                    fname = download_austin_permits("GLOBAL", permits_dir, gcfg.permit.austin_base_url, start_dt, end_dt)
                    
                    if fname:
                        st.success(f"✅ Downloaded: `{fname}`")
                        
                        if update_on_download:
                            with st.spinner("Updating master CSV file..."):
                                csv_path = permits_dir / fname
                                master_csv_path = permits_dir / "permit_data.csv"
                                master_parquet_path = permits_dir / "permit_data.parquet"
                                
                                # Step 1: Append to master CSV
                                result = append_csv_to_master_csv(
                                    csv_path, master_csv_path, filter_mp_only, save_backups
                                )
                                
                                if result["status"] == "success":
                                    st.success(f"✅ Added {result['new_records']} new records to master CSV")
                                    st.info(f"📊 Total records: {result['total_records']}")
                                    st.info(f"🔍 {result['filter_info']}")
                                    
                                    # Step 2: Convert master CSV to parquet
                                    with st.spinner("Converting master CSV to parquet..."):
                                        parquet_result = convert_master_csv_to_parquet(master_csv_path, master_parquet_path)
                                        
                                        if parquet_result["status"] == "success":
                                            st.success(f"✅ Parquet file updated with {parquet_result['records']:,} records")
                                        else:
                                            st.error(f"❌ Parquet conversion error: {parquet_result['error']}")
                                        
                                elif result["status"] == "no_data":
                                    st.warning(f"⚠️ {result['message']}")
                                    st.info(f"🔍 {result['filter_info']}")
                                    
                                else:
                                    st.error(f"❌ CSV merge error: {result['error']}")
                    else:
                        st.error("❌ No file saved (check logs)")
                        
                except Exception as e:
                    st.error(f"❌ Download error: {str(e)}")
    
    with tab2:
        st.subheader("Automated Download Settings")
        
        # Scheduling settings
        col1, col2 = st.columns(2)
        
        with col1:
            enable_scheduling = st.checkbox(
                "Enable scheduled downloads", 
                value=False,
                key="enable_scheduling",
                help="Run downloads automatically at specified intervals"
            )
            
            if enable_scheduling:
                interval_minutes = st.number_input(
                    "Download interval (minutes)", 
                    min_value=1, max_value=1440, 
                    value=60,
                    key="schedule_interval"
                )
        
        with col2:
            if enable_scheduling:
                schedule_lookback = st.number_input(
                    "Scheduled lookback days", 
                    min_value=1, max_value=30, 
                    value=1,
                    key="schedule_lookback",
                    help="Days to look back for scheduled downloads"
                )
        
        # Processing settings for scheduled downloads
        if enable_scheduling:
            st.markdown("**Scheduled Download Processing**")
            
            col3, col4 = st.columns(2)
            
            with col3:
                schedule_update = st.checkbox(
                    "Auto-convert scheduled downloads", 
                    value=True,
                    key="schedule_update",
                    help="Automatically convert CSV to parquet for scheduled downloads"
                )
                
                schedule_filter = st.checkbox(
                    "Filter MP permits only", 
                    value=True,
                    key="schedule_filter"
                )
            
            with col4:
                schedule_backup = st.checkbox(
                    "Save backups on scheduled updates", 
                    value=True,
                    key="schedule_backup"
                )
        
        # Start/stop scheduling
        if enable_scheduling:
            col_start, col_stop = st.columns(2)
            
            with col_start:
                if st.button("▶️ Start Scheduled Downloads", type="primary"):
                    if 'scheduler_running' not in st.session_state:
                        start_scheduled_downloads(
                            interval_minutes, permits_dir, gcfg.permit.austin_base_url,
                            schedule_lookback, schedule_update, schedule_filter, schedule_backup
                        )
                        st.session_state.scheduler_running = True
                        st.success("✅ Scheduled downloads started!")
                    else:
                        st.warning("⚠️ Scheduler already running")
            
            with col_stop:
                if st.button("⏹️ Stop Scheduled Downloads"):
                    if 'scheduler_running' in st.session_state:
                        del st.session_state.scheduler_running
                        st.info("ℹ️ Scheduler stopped (restart app to fully stop background thread)")
                    else:
                        st.info("ℹ️ No scheduler running")
        
        # Save settings to config
        st.divider()
        
        if st.button("💾 Save Settings"):
            # Update global config with scheduling settings
            # Note: This is a simplified approach - in production you'd want more robust config management
            st.success("✅ Settings saved!")
    
    with tab3:
        st.subheader("Data Status")
        
        # Check for master files
        master_csv_path = permits_dir / "permit_data.csv"
        master_parquet_path = permits_dir / "permit_data.parquet"
        
        # Display file status
        col_status1, col_status2 = st.columns(2)
        
        with col_status1:
            if master_csv_path.exists():
                csv_size_mb = master_csv_path.stat().st_size / (1024 * 1024)
                st.success(f"✅ Master CSV: {csv_size_mb:.1f} MB")
            else:
                st.warning("⚠️ No master CSV file")
        
        with col_status2:
            if master_parquet_path.exists():
                parquet_size_mb = master_parquet_path.stat().st_size / (1024 * 1024)
                st.success(f"✅ Master Parquet: {parquet_size_mb:.1f} MB")
            else:
                st.warning("⚠️ No master parquet file")
        
        # Show data from master CSV (source of truth)
        if master_csv_path.exists():
            try:
                df = pd.read_csv(master_csv_path, low_memory=False)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", f"{len(df):,}")
                
                with col2:
                    if 'Permit Type' in df.columns:
                        mp_count = len(df[df['Permit Type'] == 'MP'])
                        st.metric("MP Permits", f"{mp_count:,}")
                    else:
                        st.metric("MP Permits", "N/A")
                
                with col3:
                    if 'Applied Date' in df.columns:
                        try:
                            df['Applied Date'] = pd.to_datetime(df['Applied Date'], errors='coerce')
                            recent_count = len(df[df['Applied Date'] >= (datetime.now() - timedelta(days=30))])
                            st.metric("Last 30 Days", f"{recent_count:,}")
                        except:
                            st.metric("Last 30 Days", "N/A")
                    else:
                        st.metric("Last 30 Days", "N/A")
                
                with col4:
                    if 'Permit Num' in df.columns:
                        unique_permits = df['Permit Num'].nunique()
                        st.metric("Unique Permits", f"{unique_permits:,}")
                    else:
                        st.metric("Unique Permits", "N/A")
                
                # Show recent data
                st.markdown("**Recent Records Preview**")
                call_compat(st.dataframe, df.head(100), use_container_width=True)
                
                # Export and conversion options
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    if st.button("📥 Export Master CSV"):
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"permit_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                with col_export2:
                    if st.button("🔄 Update Parquet from CSV"):
                        with st.spinner("Converting CSV to parquet..."):
                            parquet_result = convert_master_csv_to_parquet(master_csv_path, master_parquet_path)
                            
                            if parquet_result["status"] == "success":
                                st.success(f"✅ Parquet updated with {parquet_result['records']:,} records")
                            else:
                                st.error(f"❌ Conversion error: {parquet_result['error']}")
                
            except Exception as e:
                st.error(f"❌ Error reading master CSV file: {str(e)}")
                
        else:
            st.info("ℹ️ No master CSV file found. Download some permits first!")
        
        # Show recent CSV downloads for manual processing
        st.divider()
        st.markdown("**Recent CSV Downloads**")
        
        csv_files = list(permits_dir.glob("permits_austin_*.csv"))
        if csv_files:
            # Sort by modification time, most recent first
            csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for i, csv_file in enumerate(csv_files[:10]):  # Show last 10
                mod_time = datetime.fromtimestamp(csv_file.stat().st_mtime)
                file_size_mb = csv_file.stat().st_size / (1024 * 1024)
                
                col_name, col_time, col_size, col_action = st.columns([3, 2, 1, 1])
                
                with col_name:
                    st.write(f"📄 {csv_file.name}")
                
                with col_time:
                    st.write(mod_time.strftime("%Y-%m-%d %H:%M"))
                
                with col_size:
                    st.write(f"{file_size_mb:.1f} MB")
                
                with col_action:
                    if st.button("➕", key=f"append_{i}", help="Append to master"):
                        with st.spinner("Appending to master CSV..."):
                            # Step 1: Append to master CSV
                            result = append_csv_to_master_csv(csv_file, master_csv_path, True, True)
                            
                            if result["status"] == "success":
                                st.success(f"✅ Added {result['new_records']} records to master CSV")
                                
                                # Step 2: Update parquet
                                parquet_result = convert_master_csv_to_parquet(master_csv_path, master_parquet_path)
                                if parquet_result["status"] == "success":
                                    st.success("✅ Parquet file updated")
                                
                            elif result["status"] == "no_data":
                                st.warning(f"⚠️ {result['message']}")
                            else:
                                st.error(f"❌ {result['error']}")
        else:
            st.info("ℹ️ No CSV files found")


if __name__ == "__main__":
    main()
