"""
Custom Data Extraction - Configure and extract recent event data with customer enrichment
"""

import json
import sys
import time
from pathlib import Path as _P
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = _P(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._shared import select_company_config
from datahound.extract import (
    CustomExtractionEngine, ExtractionConfig, ExtractionBatch, 
    TimeFilter, TimeFilterType, NumericFilter, NumericFilterType, EnrichmentConfig
)
from apps.components.scheduler_ui import (
    render_schedule_config, render_task_manager, 
    create_scheduled_task, render_scheduler_status
)
from datahound.scheduler import TaskType, ScheduleType

# Import UI components
try:
    from apps.components.ui_components import (
        inject_custom_css, dh_page_header, dh_metric_card, dh_professional_metric_grid,
        dh_chart_card, dh_professional_chart_theme, dh_create_metric_chart,
        dh_alert, dh_breadcrumbs, dh_data_table, dh_status_badge, dh_progress_bar
    )
    UI_COMPONENTS_AVAILABLE = True
except ImportError:
    UI_COMPONENTS_AVAILABLE = False


def render(company: str, config) -> None:
    data_dir = _P("data") / company
    parquet_dir = _P("companies") / company / "parquet"

    if 'extraction_engine' not in st.session_state:
        st.session_state.extraction_engine = CustomExtractionEngine(company, data_dir, parquet_dir)

    engine = st.session_state.extraction_engine

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Configure", "Execute", "Results", "History", "Scheduler Status"])

    with tab1:
        show_configuration_interface(engine)
    with tab2:
        show_execution_interface(engine, company)
    with tab3:
        show_results_interface(engine, company)
    with tab4:
        show_history_interface(engine, company)
    with tab5:
        render_scheduler_status(key_context="extraction")


def main():
    st.set_page_config(
        page_title="DataHound Pro - Custom Extraction",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    if UI_COMPONENTS_AVAILABLE:
        inject_custom_css()
        dh_page_header(
            title="Custom Data Extraction",
            subtitle="Extract and enrich recent event data with configurable time filters"
        )
        dh_breadcrumbs(["Dashboard", "Data Operations", "Custom Extraction"])
    else:
        st.title("Custom Data Extraction")
    company, config = select_company_config()
    if not company or not config:
        st.warning("Please select a valid company configuration.")
        return
    render(company, config)


def show_configuration_interface(engine: CustomExtractionEngine):
    """Show the extraction configuration interface"""
    
    if UI_COMPONENTS_AVAILABLE:
        dh_alert("Configure your custom data extraction rules below. Each extraction rule can filter recent events and enrich them with customer data.", "info")
    else:
        st.info("Configure your custom data extraction rules below. Each extraction rule can filter recent events and enrich them with customer data.")
    
    # Get available extraction templates
    templates = engine.get_available_extractions()
    
    # Initialize session state for configurations
    if 'extraction_configs' not in st.session_state:
        # Try to load saved configuration first, fall back to templates
        saved_configs = load_extraction_configuration(engine.company)
        if saved_configs:
            st.session_state.extraction_configs = saved_configs
            st.success(f"📥 Loaded saved configuration ({len(saved_configs)} rules)")
        else:
            st.session_state.extraction_configs = templates
    
    # Backward compatibility: Add enabled attribute to existing TimeFilter objects
    configs = st.session_state.extraction_configs
    for config in configs:
        if not hasattr(config.time_filter, 'enabled'):
            config.time_filter.enabled = True  # Default to enabled for existing configs
    
    # Configuration management buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Reset to Defaults", width='stretch'):
            st.session_state.extraction_configs = templates
            # Clear session state to force reload
            if 'extraction_engine' in st.session_state:
                del st.session_state.extraction_engine
            st.rerun()
    
    with col2:
        if st.button("➕ Add Custom Rule", width='stretch'):
            # Add a blank configuration
            new_config = ExtractionConfig(
                name="Custom Extraction",
                description="Custom extraction rule",
                enabled=False,
                source_event_type="custom",
                source_file_name="custom_master.parquet",
                time_filter=TimeFilter(
                    enabled=True,
                    filter_type=TimeFilterType.DAYS_BACK,
                    field_name="detected_at",
                    days_back=14
                ),
                numeric_filter=NumericFilter(
                    enabled=False,
                    field_name="",
                    filter_type=NumericFilterType.LESS_THAN
                ),
                enrichment=EnrichmentConfig(),
                output_file_name="custom_extraction.parquet"
            )
            st.session_state.extraction_configs.append(new_config)
            st.rerun()
    
    with col3:
        if st.button("💾 Save Configuration", width='stretch'):
            try:
                save_extraction_configuration(engine.company, configs)
                st.success("✅ Configuration saved successfully!")
                # Successfully saved - no additional action needed
            except Exception as e:
                st.error(f"❌ Error saving configuration: {e}")
    
    with col4:
        if st.button("🔄 Reload Configuration", width='stretch'):
            try:
                loaded_configs = load_extraction_configuration(engine.company)
                if loaded_configs:
                    st.session_state.extraction_configs = loaded_configs
                    st.success("✅ Configuration reloaded from disk!")
                    st.rerun()
                else:
                    st.warning("⚠️ No saved configuration found - using defaults")
            except Exception as e:
                st.error(f"❌ Error loading configuration: {e}")
    
    st.markdown("---")
    
    # Configuration editor
    st.subheader("📝 Extraction Rules Configuration")
    
    for i, config in enumerate(configs):
        with st.expander(f"{'✅' if config.enabled else '❌'} {config.name}", expanded=config.enabled):
            
            # Basic settings
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input(
                    "Rule Name", 
                    value=config.name, 
                    key=f"name_{i}"
                )
                config.name = new_name
                
                new_description = st.text_area(
                    "Description",
                    value=config.description,
                    key=f"desc_{i}",
                    height=80
                )
                config.description = new_description
                
                config.enabled = st.checkbox(
                    "Enable this extraction",
                    value=config.enabled,
                    key=f"enabled_{i}"
                )
            
            with col2:
                # Event source configuration
                event_types = [
                    "canceled_jobs", "lost_customers", "aging_systems",
                    "unsold_estimates", "overdue_maintenance", "custom"
                ]
                
                current_idx = event_types.index(config.source_event_type) if config.source_event_type in event_types else 0
                config.source_event_type = st.selectbox(
                    "Source Event Type",
                    event_types,
                    index=current_idx,
                    key=f"event_type_{i}"
                )
                
                config.source_file_name = st.text_input(
                    "Source File Name",
                    value=config.source_file_name,
                    key=f"source_file_{i}",
                    help="Parquet file name in the master_files directory"
                )
                
                config.output_file_name = st.text_input(
                    "Output File Name", 
                    value=config.output_file_name,
                    key=f"output_file_{i}",
                    help="Name for the extracted data file"
                )
            
            st.markdown("**Time Filter Configuration**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                config.time_filter.enabled = st.checkbox(
                    "Enable Time Filter",
                    value=config.time_filter.enabled,
                    key=f"time_enabled_{i}",
                    help="Filter records based on date/time values"
                )
            
            with col2:
                filter_types = ["Days Back", "Date Range", "Static Date"]
                current_filter = {
                    TimeFilterType.DAYS_BACK: 0,
                    TimeFilterType.DATE_RANGE: 1,
                    TimeFilterType.STATIC_DATE: 2
                }.get(config.time_filter.filter_type, 0)
                
                filter_selection = st.selectbox(
                    "Filter Type",
                    filter_types,
                    index=current_filter,
                    key=f"filter_type_{i}",
                    disabled=not config.time_filter.enabled
                )
                
                # Update filter type
                if filter_selection == "Days Back":
                    config.time_filter.filter_type = TimeFilterType.DAYS_BACK
                elif filter_selection == "Date Range":
                    config.time_filter.filter_type = TimeFilterType.DATE_RANGE
                else:
                    config.time_filter.filter_type = TimeFilterType.STATIC_DATE
            
            with col3:
                config.time_filter.field_name = st.text_input(
                    "Date Field Name",
                    value=config.time_filter.field_name,
                    key=f"date_field_{i}",
                    help="Field name to filter on (e.g., 'completion_date', 'detected_at')",
                    disabled=not config.time_filter.enabled
                )
            
            with col4:
                # Dynamic filter configuration based on type
                if config.time_filter.filter_type == TimeFilterType.DAYS_BACK:
                    config.time_filter.days_back = st.number_input(
                        "Days Back",
                        min_value=1,
                        max_value=365,
                        value=config.time_filter.days_back or 14,
                        key=f"days_back_{i}",
                        disabled=not config.time_filter.enabled
                    )
                elif config.time_filter.filter_type == TimeFilterType.DATE_RANGE:
                    config.time_filter.start_date = st.date_input(
                        "Start Date",
                        value=config.time_filter.start_date or (date.today() - timedelta(days=30)),
                        key=f"start_date_{i}",
                        disabled=not config.time_filter.enabled
                    )
                    config.time_filter.end_date = st.date_input(
                        "End Date", 
                        value=config.time_filter.end_date or date.today(),
                        key=f"end_date_{i}",
                        disabled=not config.time_filter.enabled
                    )
                else:
                    config.time_filter.static_date = st.date_input(
                        "Target Date",
                        value=config.time_filter.static_date or date.today(),
                        key=f"static_date_{i}",
                        disabled=not config.time_filter.enabled
                    )
            
            # Numeric Filter Configuration
            st.markdown("**Numeric Filter Configuration**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                config.numeric_filter.enabled = st.checkbox(
                    "Enable Numeric Filter",
                    value=config.numeric_filter.enabled,
                    key=f"numeric_enabled_{i}",
                    help="Filter records based on numeric column values"
                )
                
                config.numeric_filter.field_name = st.text_input(
                    "Numeric Field Name",
                    value=config.numeric_filter.field_name,
                    key=f"numeric_field_{i}",
                    help="Column name to filter on (e.g., 'months_since_last_contact')",
                    disabled=not config.numeric_filter.enabled
                )
            
            with col2:
                filter_types = ["Less Than", "Greater Than", "Equals", "Between", "Less Than or Equal", "Greater Than or Equal"]
                current_numeric_filter = {
                    NumericFilterType.LESS_THAN: 0,
                    NumericFilterType.GREATER_THAN: 1,
                    NumericFilterType.EQUALS: 2,
                    NumericFilterType.BETWEEN: 3,
                    NumericFilterType.LESS_THAN_OR_EQUAL: 4,
                    NumericFilterType.GREATER_THAN_OR_EQUAL: 5
                }.get(config.numeric_filter.filter_type, 0)
                
                filter_selection = st.selectbox(
                    "Filter Type",
                    filter_types,
                    index=current_numeric_filter,
                    key=f"numeric_filter_type_{i}",
                    disabled=not config.numeric_filter.enabled
                )
                
                # Update filter type
                if filter_selection == "Less Than":
                    config.numeric_filter.filter_type = NumericFilterType.LESS_THAN
                elif filter_selection == "Greater Than":
                    config.numeric_filter.filter_type = NumericFilterType.GREATER_THAN
                elif filter_selection == "Equals":
                    config.numeric_filter.filter_type = NumericFilterType.EQUALS
                elif filter_selection == "Between":
                    config.numeric_filter.filter_type = NumericFilterType.BETWEEN
                elif filter_selection == "Less Than or Equal":
                    config.numeric_filter.filter_type = NumericFilterType.LESS_THAN_OR_EQUAL
                elif filter_selection == "Greater Than or Equal":
                    config.numeric_filter.filter_type = NumericFilterType.GREATER_THAN_OR_EQUAL
            
            with col3:
                # Dynamic numeric filter configuration based on type
                if config.numeric_filter.filter_type == NumericFilterType.BETWEEN:
                    config.numeric_filter.min_value = st.number_input(
                        "Min Value",
                        value=config.numeric_filter.min_value or 0.0,
                        key=f"numeric_min_{i}",
                        disabled=not config.numeric_filter.enabled
                    )
                    config.numeric_filter.max_value = st.number_input(
                        "Max Value", 
                        value=config.numeric_filter.max_value or 100.0,
                        key=f"numeric_max_{i}",
                        disabled=not config.numeric_filter.enabled
                    )
                else:
                    config.numeric_filter.value = st.number_input(
                        "Filter Value",
                        value=config.numeric_filter.value or 0.0,
                        key=f"numeric_value_{i}",
                        disabled=not config.numeric_filter.enabled
                    )
            
            # Enrichment configuration
            st.markdown("**Data Enrichment Options**")
            col1, col2 = st.columns(2)
            
            with col1:
                config.enrichment.include_customer_core_data = st.checkbox(
                    "Include Customer Core Data",
                    value=config.enrichment.include_customer_core_data,
                    key=f"enrich_core_{i}"
                )
                
                config.enrichment.include_rfm_analysis = st.checkbox(
                    "Include RFM Analysis",
                    value=config.enrichment.include_rfm_analysis,
                    key=f"enrich_rfm_{i}",
                    disabled=not config.enrichment.include_customer_core_data
                )
            
            with col2:
                config.enrichment.include_demographics = st.checkbox(
                    "Include Demographics",
                    value=config.enrichment.include_demographics,
                    key=f"enrich_demo_{i}",
                    disabled=not config.enrichment.include_customer_core_data
                )
                
                config.enrichment.include_segmentation = st.checkbox(
                    "Include Customer Segmentation",
                    value=config.enrichment.include_segmentation,
                    key=f"enrich_segment_{i}",
                    disabled=not config.enrichment.include_customer_core_data
                )
            
            # Additional options
            col1, col2 = st.columns(2)
            with col1:
                config.enrichment.customer_id_field = st.text_input(
                    "Customer ID Field",
                    value=config.enrichment.customer_id_field,
                    key=f"customer_id_field_{i}",
                    help="Field name containing the customer ID for enrichment"
                )
            
            with col2:
                config.max_records = st.number_input(
                    "Max Records (0 = unlimited)",
                    min_value=0,
                    value=config.max_records or 0,
                    key=f"max_records_{i}"
                )
                if config.max_records == 0:
                    config.max_records = None
            
            if config.name in ["Recent Cancellations", "Recent Unsold Estimates"]:
                col_calls, col_sms = st.columns(2)
                with col_calls:
                    st.markdown("**Inbound Call Invalidation**")
                    config.enable_inbound_call_check = st.checkbox(
                        "Enable inbound call invalidation",
                        value=getattr(config, "enable_inbound_call_check", True),
                        key=f"enable_inbound_call_{i}"
                    )
                    if config.enable_inbound_call_check:
                        config.inbound_call_window_days = st.number_input(
                            "Follow-up window (days, 0 for unlimited)",
                            min_value=0,
                            value=getattr(config, "inbound_call_window_days", 0),
                            key=f"inbound_window_{i}"
                        )
                        config.inbound_call_directions = st.multiselect(
                            "Valid call directions",
                            options=["Inbound", "Outbound"],
                            default=getattr(config, "inbound_call_directions", ["Inbound"]),
                            key=f"inbound_directions_{i}"
                        )
                with col_sms:
                    st.markdown("**SMS Invalidation**")
                    config.enable_sms_activity_check = st.checkbox(
                        "Enable SMS invalidation",
                        value=getattr(config, "enable_sms_activity_check", True),
                        key=f"enable_sms_activity_{i}"
                    )
                    if config.enable_sms_activity_check:
                        config.sms_window_minutes = st.number_input(
                            "Follow-up window (minutes, 0 for unlimited)",
                            min_value=0,
                            value=getattr(config, "sms_window_minutes", 0),
                            key=f"sms_window_{i}"
                        )
            
            # Remove rule button
            if st.button(f"🗑️ Remove Rule", key=f"remove_{i}", type="secondary"):
                st.session_state.extraction_configs.pop(i)
                st.rerun()


def show_execution_interface(engine: CustomExtractionEngine, company: str):
    """Show the extraction execution interface"""
    
    if 'extraction_configs' not in st.session_state:
        st.warning("Please configure extraction rules first in the Configure tab.")
        return
    
    configs = st.session_state.extraction_configs
    enabled_configs = [c for c in configs if c.enabled]
    
    if not enabled_configs:
        if UI_COMPONENTS_AVAILABLE:
            dh_alert("No extraction rules are enabled. Please enable at least one rule in the Configure tab.", "warning")
        else:
            st.warning("No extraction rules are enabled. Please enable at least one rule in the Configure tab.")
        return
    
    st.subheader("🚀 Execute Extractions")
    
    # Show enabled extractions summary
    if UI_COMPONENTS_AVAILABLE:
        dh_alert(f"✅ {len(enabled_configs)} extraction rules ready to execute", "success")
    else:
        st.success(f"✅ {len(enabled_configs)} extraction rules ready to execute")
    
    # Execution options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        execute_single = st.selectbox(
            "Execute Single Rule",
            ["Select a rule..."] + [c.name for c in enabled_configs],
            key="single_execution"
        )
    
    with col2:
        if st.button("▶️ Execute Selected", width='stretch', disabled=execute_single == "Select a rule..."):
            selected_config = next((c for c in enabled_configs if c.name == execute_single), None)
            if selected_config:
                execute_single_extraction(engine, selected_config)
    
    with col3:
        if st.button("🚀 Execute All Enabled", width='stretch', type="primary"):
            execute_batch_extractions(engine, enabled_configs, company)
    
    # Automation section
    st.markdown("---")
    st.markdown("### 🤖 Automation Settings")
    
    with st.expander("⏰ Schedule Automated Execution", expanded=False):
        st.info("Configure automatic execution of all enabled extractions at regular intervals.")
        
        # Show current enabled extractions
        st.markdown("**Extractions that will run:**")
        for config in enabled_configs:
            st.write(f"• {config.name}")
        
        st.markdown("---")
        st.markdown("### Execution Interval")
        
        interval_minutes = st.number_input(
            "Run every (minutes)",
            min_value=15,
            max_value=1440,
            value=60,
            step=15,
            key="extraction_interval",
            help="Execute all enabled extractions at this interval"
        )
        
        st.info(f"🕐 Will execute all enabled extractions every {interval_minutes} minutes")
        
        # Create scheduled task button
        if st.button("📅 Create Extraction Schedule", type="primary", key="create_extraction_schedule"):
            schedule_config = {
                'schedule_type': ScheduleType.INTERVAL,
                'interval_minutes': interval_minutes
            }
            
            task_config = {
                'execute_all_enabled': True,
                'extraction_configs': []  # Will be loaded from saved configs at execution time
            }
            
            success = create_scheduled_task(
                task_type=TaskType.CUSTOM_EXTRACTION,
                company=company,
                task_name=f"Execute All Extractions - {company}",
                task_description=f"Run all enabled extractions every {interval_minutes} minutes",
                schedule_config=schedule_config,
                task_config_overrides=task_config
            )
            
            if success:
                st.success("✅ Extraction schedule created successfully!")
                st.rerun()
            else:
                st.error("Failed to create scheduled task")
        
        st.markdown("---")
        
        # Show existing scheduled tasks
        st.markdown("### Existing Schedules")
        render_task_manager(
            task_type=TaskType.CUSTOM_EXTRACTION,
            company=company,
            task_name="Custom Extractions",
            task_description="Automated custom data extraction",
            key_context="extraction_tab",
        )
    
    # Preview enabled configurations
    if enabled_configs:
        st.markdown("---")
        st.subheader("📋 Enabled Extraction Rules")
        
        for config in enabled_configs:
            with st.expander(f"📊 {config.name}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Description:** {config.description}")
                    st.markdown(f"**Source Event:** {config.source_event_type}")
                    st.markdown(f"**Output File:** {config.output_file_name}")
                
                with col2:
                    # Time filter summary
                    if config.time_filter.enabled:
                        if config.time_filter.filter_type == TimeFilterType.DAYS_BACK:
                            filter_desc = f"Last {config.time_filter.days_back} days"
                        elif config.time_filter.filter_type == TimeFilterType.DATE_RANGE:
                            filter_desc = f"{config.time_filter.start_date} to {config.time_filter.end_date}"
                        else:
                            filter_desc = f"Date: {config.time_filter.static_date}"
                        
                        st.markdown(f"**Time Filter:** {filter_desc}")
                        st.markdown(f"**Date Field:** {config.time_filter.field_name}")
                    else:
                        st.markdown(f"**Time Filter:** Disabled")
                    
                    # Numeric filter summary
                    if config.numeric_filter.enabled:
                        if config.numeric_filter.filter_type == NumericFilterType.BETWEEN:
                            numeric_desc = f"{config.numeric_filter.field_name} between {config.numeric_filter.min_value} and {config.numeric_filter.max_value}"
                        else:
                            operator_map = {
                                NumericFilterType.LESS_THAN: "<",
                                NumericFilterType.GREATER_THAN: ">", 
                                NumericFilterType.EQUALS: "=",
                                NumericFilterType.LESS_THAN_OR_EQUAL: "≤",
                                NumericFilterType.GREATER_THAN_OR_EQUAL: "≥"
                            }
                            operator = operator_map.get(config.numeric_filter.filter_type, "?")
                            numeric_desc = f"{config.numeric_filter.field_name} {operator} {config.numeric_filter.value}"
                        
                        st.markdown(f"**Numeric Filter:** {numeric_desc}")
                    else:
                        st.markdown(f"**Numeric Filter:** Disabled")
                    
                    st.markdown(f"**Enrichment:** {'✅' if config.enrichment.include_customer_core_data else '❌'}")


def show_results_interface(engine: CustomExtractionEngine, company: str):
    """Show extraction results and data preview"""
    
    st.subheader("📊 Extraction Results")
    
    # Check for recent extraction results
    recent_events_dir = _P("data") / company / "recent_events"
    
    if not recent_events_dir.exists():
        if UI_COMPONENTS_AVAILABLE:
            dh_alert("No extraction results found. Run some extractions first.", "info")
        else:
            st.info("No extraction results found. Run some extractions first.")
        return
    
    # Find parquet files in recent events directory
    result_files = list(recent_events_dir.glob("*.parquet"))
    
    if not result_files:
        if UI_COMPONENTS_AVAILABLE:
            dh_alert("No extraction result files found.", "info")
        else:
            st.info("No extraction result files found.")
        return
    
    # File selection
    selected_file = st.selectbox(
        "Select Result File to Preview",
        [f.name for f in result_files],
        key="result_file_selection"
    )
    
    if selected_file:
        file_path = recent_events_dir / selected_file
        
        try:
            df = pd.read_parquet(file_path)
            
            # File statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                st.metric("File Size", f"{file_size:.2f} MB")
            with col4:
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                st.metric("Last Modified", mod_time.strftime("%Y-%m-%d %H:%M"))
            
            # Data preview
            st.markdown("---")
            st.subheader(f"📋 Data Preview: {selected_file}")
            
            # Show column information
            with st.expander("Column Information", expanded=False):
                col_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_count = int(df[col].isnull().sum())
                    try:
                        unique_count = int(df[col].nunique())
                    except TypeError:
                        unique_count = -1
                    col_info.append({
                        "Column": col,
                        "Data Type": dtype,
                        "Null Count": null_count,
                        "Unique Values": unique_count,
                    })
                col_df = pd.DataFrame(col_info)
                st.dataframe(col_df, width='stretch')
            
            st.markdown("**Sample Data (First 10 Rows)**")
            display_df = df.head(10).copy()
            for col in display_df.columns:
                if display_df[col].apply(lambda v: isinstance(v, (list, dict))).any():
                    display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df, width='stretch')

            safe_df = df.copy()
            for col in safe_df.columns:
                if safe_df[col].apply(lambda v: isinstance(v, (list, dict))).any():
                    safe_df[col] = safe_df[col].astype(str)
            csv_data = safe_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=selected_file.replace('.parquet', '.csv'),
                mime='text/csv',
                width='stretch'
            )
            
        except Exception as e:
            st.error(f"Error loading file: {e}")


def show_history_interface(engine: CustomExtractionEngine, company: str):
    """Show extraction execution history"""
    
    st.subheader("📋 Extraction History")
    
    # Try to read extraction logs
    logs_dir = _P("data") / company / "logs"
    log_file = logs_dir / "custom_extraction_log.jsonl"
    
    if not log_file.exists():
        if UI_COMPONENTS_AVAILABLE:
            dh_alert("No extraction history found.", "info")
        else:
            st.info("No extraction history found.")
        return
    
    try:
        # Read recent log entries
        log_entries = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    log_entries.append(entry)
                except:
                    continue
        
        # Show recent entries (last 50)
        recent_entries = log_entries[-50:]
        recent_entries.reverse()  # Most recent first
        
        if not recent_entries:
            st.info("No log entries found.")
            return
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            level_filter = st.selectbox(
                "Filter by Level",
                ["All", "info", "warning", "error"],
                key="log_level_filter"
            )
        
        with col2:
            message_filter = st.text_input(
                "Filter by Message",
                key="log_message_filter"
            )
        
        # Apply filters
        filtered_entries = recent_entries
        
        if level_filter != "All":
            filtered_entries = [e for e in filtered_entries if e.get('level') == level_filter]
        
        if message_filter:
            filtered_entries = [e for e in filtered_entries if message_filter.lower() in e.get('message', '').lower()]
        
        # Display log entries
        st.markdown(f"**Showing {len(filtered_entries)} log entries**")
        
        for entry in filtered_entries:
            timestamp = entry.get('timestamp', 'Unknown')
            level = entry.get('level', 'info')
            message = entry.get('message', 'No message')
            details = entry.get('details', {})
            
            # Level styling
            level_colors = {
                'info': '🔵',
                'warning': '🟡',
                'error': '🔴'
            }
            level_icon = level_colors.get(level, '⚪')
            
            with st.expander(f"{level_icon} {timestamp[:19]} - {message}", expanded=False):
                st.markdown(f"**Level:** {level.title()}")
                st.markdown(f"**Timestamp:** {timestamp}")
                st.markdown(f"**Message:** {message}")
                
                if details:
                    st.markdown("**Details:**")
                    st.json(details)
        
    except Exception as e:
        st.error(f"Error reading extraction history: {e}")


def execute_single_extraction(engine: CustomExtractionEngine, config: ExtractionConfig):
    """Execute a single extraction with progress display"""
    
    with st.spinner(f"Executing {config.name}..."):
        result = engine.extract_single(config)
    
    # Display result
    if result.success:
        st.success(f"✅ {config.name} completed successfully!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records Found", result.records_found)
        with col2:
            st.metric("Records Enriched", result.records_enriched)
        with col3:
            st.metric("Enrichment Rate", f"{result.enrichment_rate:.1f}%")
        with col4:
            st.metric("Duration", f"{result.duration_ms}ms")
        
        if result.records_saved > 0:
            st.info(f"💾 Results saved to: {result.output_file.name}")
        else:
            st.warning(f"⚠️ No records found matching criteria - no file created")
    else:
        st.error(f"❌ {config.name} failed: {result.error_message}")


def execute_batch_extractions(engine: CustomExtractionEngine, configs: List[ExtractionConfig], company: str):
    """Execute multiple extractions in batch"""
    
    batch = ExtractionBatch(
        extractions=configs,
        company=company,
        parallel_execution=False
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, config in enumerate(configs):
        status_text.text(f"Executing {config.name}...")
        progress_bar.progress((i) / len(configs))
        
        result = engine.extract_single(config)
        results.append(result)
    
    progress_bar.progress(1.0)
    status_text.text("Batch execution completed!")
    
    # Show summary
    st.markdown("---")
    st.subheader("🎯 Batch Execution Summary")
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Extractions", len(results))
    with col2:
        st.metric("Successful", len(successful))
    with col3:
        st.metric("Failed", len(failed))
    
    # Detailed results
    for result in results:
        status_icon = "✅" if result.success else "❌"
        
        with st.expander(f"{status_icon} {result.config_name}", expanded=not result.success):
            if result.success:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", result.records_found)
                with col2:
                    st.metric("Enriched", result.records_enriched)
                with col3:
                    st.metric("Duration", f"{result.duration_ms}ms")
                
                st.info(f"💾 Saved to: {result.output_file.name}")
            else:
                st.error(f"Error: {result.error_message}")


def save_extraction_configuration(company: str, configs: List[ExtractionConfig]):
    """Save extraction configuration to JSON file"""
    
    config_dir = _P("config") / "extraction"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize company name for file name (handle spaces)
    safe_company_name = company.replace(" ", "_").replace("/", "_").replace("\\", "_")
    config_file = config_dir / f"{safe_company_name}_extraction_config.json"
    
    # Convert configs to serializable format
    config_data = []
    for config in configs:
        config_dict = {
            "name": config.name,
            "description": config.description,
            "enabled": config.enabled,
            "source_event_type": config.source_event_type,
            "source_file_name": config.source_file_name,
            "output_file_name": config.output_file_name,
            "max_records": config.max_records,
            "enable_inbound_call_check": bool(getattr(config, "enable_inbound_call_check", False)),
            "inbound_call_window_days": getattr(config, "inbound_call_window_days", None),
            "inbound_call_directions": list(getattr(config, "inbound_call_directions", ["Inbound"])),
            "enable_sms_activity_check": bool(getattr(config, "enable_sms_activity_check", False)),
            "sms_window_minutes": getattr(config, "sms_window_minutes", None),
            "time_filter": {
                "enabled": config.time_filter.enabled,
                "filter_type": config.time_filter.filter_type.value,
                "field_name": config.time_filter.field_name,
                "days_back": config.time_filter.days_back,
                "start_date": config.time_filter.start_date.isoformat() if config.time_filter.start_date else None,
                "end_date": config.time_filter.end_date.isoformat() if config.time_filter.end_date else None,
                "static_date": config.time_filter.static_date.isoformat() if config.time_filter.static_date else None
            },
            "numeric_filter": {
                "enabled": config.numeric_filter.enabled,
                "field_name": config.numeric_filter.field_name,
                "filter_type": config.numeric_filter.filter_type.value,
                "value": config.numeric_filter.value,
                "min_value": config.numeric_filter.min_value,
                "max_value": config.numeric_filter.max_value
            },
            "enrichment": {
                "include_customer_core_data": config.enrichment.include_customer_core_data,
                "include_rfm_analysis": config.enrichment.include_rfm_analysis,
                "include_demographics": config.enrichment.include_demographics,
                "include_segmentation": config.enrichment.include_segmentation,
                "customer_id_field": config.enrichment.customer_id_field
            }
        }
        config_data.append(config_dict)
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)


def load_extraction_configuration(company: str) -> Optional[List[ExtractionConfig]]:
    """Load extraction configuration from JSON file"""
    
    # Normalize company name for file name (handle spaces)
    safe_company_name = company.replace(" ", "_").replace("/", "_").replace("\\", "_")
    config_file = _P("config") / "extraction" / f"{safe_company_name}_extraction_config.json"
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        configs = []
        for data in config_data:
            # Parse time filter
            tf_data = data["time_filter"]
            time_filter = TimeFilter(
                enabled=tf_data.get("enabled", True),  # Default to enabled for backward compatibility
                filter_type=TimeFilterType(tf_data["filter_type"]),
                field_name=tf_data["field_name"],
                days_back=tf_data.get("days_back"),
                start_date=date.fromisoformat(tf_data["start_date"]) if tf_data.get("start_date") else None,
                end_date=date.fromisoformat(tf_data["end_date"]) if tf_data.get("end_date") else None,
                static_date=date.fromisoformat(tf_data["static_date"]) if tf_data.get("static_date") else None
            )
            
            # Parse numeric filter config
            nf_data = data.get("numeric_filter", {
                "enabled": False,
                "field_name": "",
                "filter_type": "less_than",
                "value": 0.0,
                "min_value": None,
                "max_value": None
            })
            numeric_filter = NumericFilter(
                enabled=nf_data["enabled"],
                field_name=nf_data["field_name"],
                filter_type=NumericFilterType(nf_data["filter_type"]),
                value=nf_data.get("value"),
                min_value=nf_data.get("min_value"),
                max_value=nf_data.get("max_value")
            )
            
            # Parse enrichment config
            enrich_data = data["enrichment"]
            enrichment = EnrichmentConfig(
                include_customer_core_data=enrich_data["include_customer_core_data"],
                include_rfm_analysis=enrich_data["include_rfm_analysis"],
                include_demographics=enrich_data["include_demographics"],
                include_segmentation=enrich_data["include_segmentation"],
                customer_id_field=enrich_data["customer_id_field"]
            )
            
            # Create config
            config = ExtractionConfig(
                name=data["name"],
                description=data["description"],
                enabled=data["enabled"],
                source_event_type=data["source_event_type"],
                source_file_name=data["source_file_name"],
                output_file_name=data["output_file_name"],
                max_records=data.get("max_records"),
                enable_inbound_call_check=bool(data.get("enable_inbound_call_check", False)),
                inbound_call_window_days=data.get("inbound_call_window_days"),
                inbound_call_directions=list(data.get("inbound_call_directions", ["Inbound"])),
                enable_sms_activity_check=bool(data.get("enable_sms_activity_check", False)),
                sms_window_minutes=data.get("sms_window_minutes"),
                time_filter=time_filter,
                numeric_filter=numeric_filter,
                enrichment=enrichment
            )
            
            configs.append(config)
        
        return configs
        
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None


if __name__ == "__main__":
    main()
