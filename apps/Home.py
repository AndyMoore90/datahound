import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Add the project root to Python path for imports
import sys
import os
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from apps._shared import ensure_root_on_path, select_company_config, read_jsonl

# Import UI components
try:
    from apps.components.ui_components import (
        inject_custom_css, dh_page_header, dh_professional_metric_grid,
        dh_alert, dh_breadcrumbs
    )
    UI_COMPONENTS_AVAILABLE = True
except ImportError:
    UI_COMPONENTS_AVAILABLE = False

ensure_root_on_path()


def main():
    st.set_page_config(
        page_title="DataHound Pro - Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load professional styling
    if UI_COMPONENTS_AVAILABLE:
        inject_custom_css()
    
    # Page header
    if UI_COMPONENTS_AVAILABLE:
        dh_page_header(
            title="⚡ DataHound Pro",
            subtitle="Intelligence That Drives Growth - Business Dashboard"
        )
    else:
        st.title("⚡ DataHound Pro")
        st.markdown("**Intelligence That Drives Growth - Business Dashboard**")
    
    # Company selection
    company, config = select_company_config()
    if not company or not config:
        if UI_COMPONENTS_AVAILABLE:
            dh_alert("Please select a valid company configuration to view dashboard.", "warning")
        else:
            st.warning("Please select a valid company configuration to view dashboard.")
        return
    
    # Breadcrumbs and success indicator
    if UI_COMPONENTS_AVAILABLE:
        dh_breadcrumbs(["Dashboard", "Business Overview"])
        dh_alert(f"Active Company: {company}", "success")
    else:
        st.success(f"📊 Active Company: **{company}**")
    
    # Dashboard content
    show_professional_dashboard(company, config)


def show_professional_dashboard(company: str, config: Dict[str, Any]):
    
    # Get data file paths
    data_dir = Path("data") / company
    parquet_dir = Path("companies") / company / "parquet"
    
    # Load comprehensive business metrics
    metrics = get_comprehensive_business_metrics(parquet_dir, data_dir)
    
    # Key Performance Indicators
    st.markdown("### 📊 Key Performance Indicators")
    
    if UI_COMPONENTS_AVAILABLE:
        # Primary metrics row using professional components
        primary_metrics = [
            {
                'value': f"{metrics.get('total_customers', 0):,}",
                'label': 'Total Customers',
                'change': metrics.get('customers_change'),
                'change_type': 'positive',
                'icon': '👥'
            },
            {
                'value': f"{metrics.get('total_profiles', 0):,}",
                'label': 'Enhanced Profiles',
                'change': metrics.get('profiles_change'),
                'change_type': 'positive',
                'icon': '🎯'
            },
            {
                'value': metrics.get('active_events', 0),
                'label': 'Active Events',
                'change': metrics.get('events_change'),
                'change_type': 'info',
                'icon': '⚡'
            },
            {
                'value': f"${metrics.get('revenue_opportunities', 0):,.0f}",
                'label': 'Revenue Opportunities',
                'change': metrics.get('revenue_change'),
                'change_type': 'positive',
                'icon': '💰'
            }
        ]
        
        dh_professional_metric_grid(primary_metrics)
    else:
        # Fallback simple metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Customers", f"{metrics.get('total_customers', 0):,}", metrics.get('customers_change'))
        with col2:
            st.metric("🎯 Enhanced Profiles", f"{metrics.get('total_profiles', 0):,}", metrics.get('profiles_change'))
        with col3:
            st.metric("⚡ Active Events", metrics.get('active_events', 0), metrics.get('events_change'))
        with col4:
            st.metric("💰 Revenue Opportunities", f"${metrics.get('revenue_opportunities', 0):,.0f}", metrics.get('revenue_change'))
    
    # Secondary metrics row
    st.markdown("### 🎯 Business Intelligence Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Permits Identified", f"{metrics.get('permits_found', 0):,}")
    with col2:
        st.metric("🏢 Competitor Activity", f"{metrics.get('competitor_permits', 0):,}")
    with col3:
        st.metric("💎 Avg Customer Value", f"${metrics.get('avg_customer_value', 0):,.0f}")
    with col4:
        st.metric("📈 Conversion Rate", f"{metrics.get('conversion_rate', 0):.1f}%")
    
    # Dashboard sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        show_business_analytics(parquet_dir, metrics)
        show_recent_activity_feed(data_dir)
    
    with col2:
        show_system_health(data_dir, parquet_dir)
        show_quick_actions(company)


def get_comprehensive_business_metrics(parquet_dir: Path, data_dir: Path) -> Dict[str, Any]:
    
    metrics = {
        'total_customers': 0,
        'total_profiles': 0,
        'active_events': 0,
        'revenue_opportunities': 0,
        'permits_found': 0,
        'competitor_permits': 0,
        'avg_customer_value': 0,
        'conversion_rate': 0
    }
    
    try:
        # Customer metrics
        customers_file = parquet_dir / "Customers.parquet"
        if customers_file.exists():
            customers_df = pd.read_parquet(customers_file)
            metrics['total_customers'] = len(customers_df)
            
            # Calculate average customer value if revenue data available
            if 'Customers Lifetime Revenue' in customers_df.columns:
                lifetime_revenue = pd.to_numeric(customers_df['Customers Lifetime Revenue'], errors='coerce')
                avg_value = lifetime_revenue.mean()
                metrics['avg_customer_value'] = avg_value if pd.notna(avg_value) else 1250
            else:
                metrics['avg_customer_value'] = 1250  # Default estimate
        
        # Enhanced profile metrics - check both old and new core data files
        enhanced_profiles_file = parquet_dir / "customer_core_data.parquet"
        legacy_profiles_file = parquet_dir / "customer_profiles_core_data.parquet"
        
        if enhanced_profiles_file.exists():
            profiles_df = pd.read_parquet(enhanced_profiles_file)
            metrics['total_profiles'] = len(profiles_df)
        elif legacy_profiles_file.exists():
            profiles_df = pd.read_parquet(legacy_profiles_file)
            metrics['total_profiles'] = len(profiles_df)
        
        if 'profiles_df' in locals():
            
            # Permit analytics
            if 'permit_count' in profiles_df.columns:
                metrics['permits_found'] = int(profiles_df['permit_count'].sum())
            if 'competitor_permit_count' in profiles_df.columns:
                metrics['competitor_permits'] = int(profiles_df['competitor_permit_count'].sum())
        
        # Jobs and conversion metrics
        jobs_file = parquet_dir / "Jobs.parquet"
        estimates_file = parquet_dir / "Estimates.parquet"
        
        if jobs_file.exists() and estimates_file.exists():
            jobs_df = pd.read_parquet(jobs_file)
            estimates_df = pd.read_parquet(estimates_file)
            
            total_jobs = len(jobs_df)
            total_estimates = len(estimates_df)
            
            if total_estimates > 0:
                metrics['conversion_rate'] = (total_jobs / total_estimates) * 100
        
        # Event and opportunity metrics from master event files
        active_events = 0
        revenue_opportunities = 0
        
        # Count active events from master files
        event_files = [
            "lost_customers_master.parquet",
            "aging_systems_master.parquet", 
            "overdue_maintenance_master.parquet",
            "canceled_jobs_master.parquet",
            "unsold_estimates_master.parquet"
        ]
        
        for event_file in event_files:
            event_path = parquet_dir / event_file
            if event_path.exists():
                try:
                    event_df = pd.read_parquet(event_path)
                    active_events += len(event_df)
                    
                    # Calculate revenue opportunities based on event type
                    if "lost_customers" in event_file:
                        revenue_opportunities += len(event_df) * 2500  # Win-back opportunity
                    elif "aging_systems" in event_file:
                        revenue_opportunities += len(event_df) * 8000  # System replacement
                    elif "overdue_maintenance" in event_file:
                        revenue_opportunities += len(event_df) * 350   # Maintenance service
                    elif "unsold_estimates" in event_file:
                        revenue_opportunities += len(event_df) * 1200  # Re-engagement
                    elif "canceled_jobs" in event_file:
                        revenue_opportunities += len(event_df) * 800   # Recovery opportunity
                except:
                    pass
        
        # Add permit-based opportunities
        permit_opportunities = metrics.get('permits_found', 0) * 1500
        revenue_opportunities += permit_opportunities
        
        metrics['active_events'] = active_events
        metrics['revenue_opportunities'] = revenue_opportunities
        
        metrics['customers_change'] = None
        metrics['profiles_change'] = None
        metrics['events_change'] = None
        metrics['revenue_change'] = None
        
    except Exception as e:
        st.error(f"Error calculating business metrics: {e}")
    
    return metrics


def show_business_analytics(parquet_dir: Path, metrics: Dict[str, Any]):
    
    st.markdown("### 📈 Business Analytics")
    
    try:
        # Check for enhanced core data first, then legacy
        profiles_file = parquet_dir / "customer_core_data.parquet"
        if not profiles_file.exists():
            profiles_file = parquet_dir / "customer_profiles_core_data.parquet"
        
        if not profiles_file.exists():
            st.info("Build customer profiles to unlock advanced analytics")
            return
        
        profiles_df = pd.read_parquet(profiles_file)
        
        # Customer tier analysis
        if 'customer_tier' in profiles_df.columns:
            tier_data = profiles_df['customer_tier'].value_counts()
            
            fig = px.pie(
                names=tier_data.index,
                values=tier_data.values,
                title='Customer Tier Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        # Service activity analysis
        if all(col in profiles_df.columns for col in ['job_count', 'estimate_count', 'invoice_count']):
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Service type distribution
                service_totals = {
                    'Jobs': int(profiles_df['job_count'].sum()),
                    'Estimates': int(profiles_df['estimate_count'].sum()),
                    'Invoices': int(profiles_df['invoice_count'].sum()),
                    'Calls': int(profiles_df.get('call_count', pd.Series([0])).sum())
                }
                
                fig = px.bar(
                    x=list(service_totals.keys()),
                    y=list(service_totals.values()),
                    title='Service Activity Overview',
                    color_discrete_sequence=['#2E86AB']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Customer engagement levels
                engagement_data = []
                for _, row in profiles_df.iterrows():
                    total_services = (row.get('job_count', 0) + 
                                    row.get('estimate_count', 0) + 
                                    row.get('call_count', 0))
                    if total_services == 0:
                        engagement_data.append('No Activity')
                    elif total_services <= 2:
                        engagement_data.append('Low Engagement')
                    elif total_services <= 5:
                        engagement_data.append('Medium Engagement')
                    else:
                        engagement_data.append('High Engagement')
                
                engagement_counts = pd.Series(engagement_data).value_counts()
                
                fig = px.pie(
                    names=engagement_counts.index,
                    values=engagement_counts.values,
                    title='Customer Engagement Levels',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, width='stretch')
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")


def show_recent_activity_feed(data_dir: Path):
    
    st.markdown("### 📋 Recent System Activity")
    
    from central_logging.config import event_detection_dir
    company = data_dir.name
    event_log_path = event_detection_dir(company) / "event_detection.jsonl"
    fallback_event = data_dir / "logs" / "event_detection_log.jsonl"
    log_files = [
        (event_log_path if event_log_path.exists() else fallback_event, "Event Detection", "⚡")
    ]
    
    all_activities = []
    
    for log_file, source, icon in log_files:
        if log_file.exists():
            try:
                entries = read_jsonl(log_file, limit=20)
                for entry in entries[-10:]:
                    activity = {
                        'timestamp': entry.get('timestamp') or entry.get('ts', ''),
                        'message': entry.get('message') or entry.get('action') or str(entry.get('event_type', '')),
                        'level': entry.get('level', 'info'),
                        'source': source,
                        'icon': icon,
                        'details': entry.get('details', {}) or {k: v for k, v in entry.items() if k not in ('timestamp', 'ts', 'message', 'action', 'level')}
                    }
                    all_activities.append(activity)
            except:
                continue
    
    # Sort by timestamp
    all_activities = sorted(all_activities, key=lambda x: x['timestamp'], reverse=True)[:15]
    
    if all_activities:
        for activity in all_activities:
            timestamp = activity['timestamp'][:19].replace('T', ' ') if activity['timestamp'] else 'Unknown'
            level_colors = {
                "info": "🔵", 
                "warning": "🟡", 
                "error": "🔴", 
                "success": "🟢"
            }
            level_icon = level_colors.get(activity['level'], "⚫")
            
            with st.expander(f"{activity['icon']} {activity['message']}", expanded=False):
                st.markdown(f"""
                **Source**: {activity['source']}  
                **Time**: {timestamp}  
                **Level**: {level_icon} {activity['level'].title()}
                """)
                
                if activity['details']:
                    st.json(activity['details'])
    else:
        st.info("No recent activity found.")


def show_system_health(data_dir: Path, parquet_dir: Path):
    
    st.markdown("### 🔧 System Health")
    
    # Check file status
    critical_files = [
        (parquet_dir / "Customers.parquet", "Customer Data", "👥"),
        (parquet_dir / "customer_profiles_core_data.parquet", "Customer Profiles", "🎯"),
        (parquet_dir / "Jobs.parquet", "Jobs Data", "🔨"),
        (parquet_dir / "Estimates.parquet", "Estimates Data", "📝")
    ]
    
    st.markdown("#### 📁 Critical Files Status")
    for file_path, name, icon in critical_files:
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                st.success(f"{icon} {name}: ✅ {len(df):,} records")
            except:
                st.error(f"{icon} {name}: ⚠️ File Error")
        else:
            st.error(f"{icon} {name}: ❌ Missing")
    
    st.markdown("#### ⚡ Quick Navigation")
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/1_Data_Pipeline.py", label="Data Pipeline", icon="🔄")
        st.page_link("pages/2_Events.py", label="Events", icon="⚡")
    with col2:
        st.page_link("pages/3_Pipeline_Monitor.py", label="Pipeline Monitor", icon="📊")
        st.page_link("pages/5_Admin.py", label="Admin", icon="⚙️")


def show_quick_actions(company: str):
    
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("⚡ Run Event Detection", width='stretch'):
        st.success("Use Event Configs to run historical scans")
    
    # System maintenance actions
    st.markdown("#### 🔧 System Maintenance")
    
    if st.button("🔄 Refresh Data Cache", width='stretch'):
        st.info("Data cache refresh initiated")
    
    if st.button("📋 View System Logs", width='stretch'):
        st.info("Navigate to Logs page for detailed system logs")


if __name__ == "__main__":
  main()
