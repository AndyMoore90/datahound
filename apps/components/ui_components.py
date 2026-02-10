"""
DataHound Pro - Professional UI Components Library
Reusable, consistent UI components for the entire application
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime
from pathlib import Path
from html import escape


def load_custom_css():
    """Load the custom CSS design system"""
    with open("apps/assets/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def dh_page_header(title: str, subtitle: str = "", actions: List[Dict] = None):
    """
    Professional page header with title, subtitle, and action buttons
    
    Args:
        title: Main page title
        subtitle: Optional subtitle text
        actions: List of action buttons [{"label": "text", "type": "primary", "key": "unique_key"}]
    """
    st.markdown(f"""
    <div class="dh-page-header">
        <div class="dh-flex dh-items-center dh-justify-between">
            <div>
                <h1 class="dh-heading-1">{title}</h1>
                {f'<p class="dh-body-small">{subtitle}</p>' if subtitle else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if actions:
        cols = st.columns(len(actions))
        for i, action in enumerate(actions):
            with cols[i]:
                btn_class = f"dh-btn dh-btn-{action.get('type', 'secondary')}"
                if st.button(action['label'], key=action.get('key', f'action_{i}')):
                    return action.get('key', f'action_{i}')
    
    return None


def dh_metric_card(value: Union[str, int, float], label: str, change: Optional[str] = None, 
                   change_type: str = "neutral", icon: str = "📊", tooltip: Optional[str] = None):
    """
    Professional metric display card
    
    Args:
        value: The metric value to display
        label: Label for the metric
        change: Optional change indicator (e.g., "+12%")
        change_type: Type of change ("positive", "negative", "neutral")
        icon: Icon to display with the metric
    """
    change_class = f"dh-metric-change {change_type}" if change else ""
    change_html = f'<div class="{change_class}">{change}</div>' if change else ""
    
    tooltip_attr = f' title="{escape(tooltip)}"' if tooltip else ""
    st.markdown(f"""
    <div class="dh-card dh-metric">
        <div class="dh-metric-value">{icon} {value}</div>
        <div class="dh-metric-label"{tooltip_attr}>{label}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)


def dh_status_badge(text: str, status: str = "neutral"):
    """
    Professional status badge
    
    Args:
        text: Badge text
        status: Badge type ("success", "warning", "error", "info", "neutral")
    """
    return f'<span class="dh-badge dh-badge-{status}">{text}</span>'


def dh_progress_bar(value: float, max_value: float = 100, status: str = "primary", 
                    show_percentage: bool = True):
    """
    Professional progress bar
    
    Args:
        value: Current progress value
        max_value: Maximum value (default 100)
        status: Progress bar type ("primary", "success", "warning", "error")
        show_percentage: Whether to show percentage text
    """
    percentage = (value / max_value) * 100
    percentage_text = f"{percentage:.1f}%" if show_percentage else ""
    
    st.markdown(f"""
    <div class="dh-progress-container">
        <div class="dh-flex dh-justify-between dh-mb-sm">
            <span class="dh-body-small">Progress</span>
            <span class="dh-body-small">{percentage_text}</span>
        </div>
        <div class="dh-progress">
            <div class="dh-progress-bar {status}" style="width: {percentage}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def dh_info_card(title: str, items: List[Dict[str, str]], icon: str = "ℹ️"):
    """
    Professional information display card
    
    Args:
        title: Card title
        items: List of {"label": "Label", "value": "Value"} items
        icon: Card icon
    """
    items_html = ""
    for item in items:
        items_html += f"""
        <div class="dh-flex dh-justify-between dh-mb-sm">
            <span class="dh-body-small">{item['label']}</span>
            <span class="dh-body">{item['value']}</span>
        </div>
        """
    
    st.markdown(f"""
    <div class="dh-card">
        <div class="dh-card-header">
            <h3 class="dh-card-title">{icon} {title}</h3>
        </div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)


def dh_alert(message: str, alert_type: str = "info", dismissible: bool = False):
    """
    Professional alert component
    
    Args:
        message: Alert message
        alert_type: Alert type ("success", "warning", "error", "info")
        dismissible: Whether the alert can be dismissed
    """
    icons = {
        "success": "✅",
        "warning": "⚠️", 
        "error": "❌",
        "info": "ℹ️"
    }
    
    icon = icons.get(alert_type, "ℹ️")
    
    if alert_type == "success":
        st.success(f"{icon} {message}")
    elif alert_type == "warning":
        st.warning(f"{icon} {message}")
    elif alert_type == "error":
        st.error(f"{icon} {message}")
    else:
        st.info(f"{icon} {message}")


def dh_data_table(df: pd.DataFrame, title: str = "", searchable: bool = True, 
                  selectable: bool = False, page_size: int = 10):
    """
    Professional data table with enhanced features
    
    Args:
        df: DataFrame to display
        title: Optional table title
        searchable: Whether to include search functionality
        selectable: Whether rows are selectable
        page_size: Number of rows per page
    """
    if title:
        st.markdown(f'<h3 class="dh-heading-3">{title}</h3>', unsafe_allow_html=True)
    
    if searchable and not df.empty:
        search_term = st.text_input("🔍 Search table", key=f"search_{title.replace(' ', '_')}")
        if search_term:
            # Simple search across all string columns
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            df = df[mask]
    
    if not df.empty:
        if selectable:
            selected = st.dataframe(
                df,
                width='stretch',
                height=min(400, (len(df) + 1) * 35),
                on_select="rerun",
                selection_mode="single-row"
            )
            return selected
        else:
            st.dataframe(
                df,
                width='stretch',
                height=min(400, (len(df) + 1) * 35)
            )
    else:
        st.info("No data available")
    
    return None


def dh_chart_card(chart_func, title: str, subtitle: str = ""):
    """
    Wrapper for charts in professional card layout
    
    Args:
        chart_func: Function that creates and returns the chart
        title: Chart title
        subtitle: Optional chart subtitle
    """
    with st.container():
        st.markdown(f"""
        <div class="dh-card">
            <div class="dh-card-header">
                <h3 class="dh-card-title">{title}</h3>
                {f'<p class="dh-card-subtitle">{subtitle}</p>' if subtitle else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        chart = chart_func()
        if chart:
            st.plotly_chart(chart, width='stretch')


def dh_professional_metric_grid(metrics: List[Dict]):
    """
    Display metrics in a professional grid layout
    
    Args:
        metrics: List of metric dictionaries with keys: value, label, change, change_type, icon
    """
    cols = st.columns(len(metrics))
    
    for i, metric in enumerate(metrics):
        with cols[i]:
            dh_metric_card(
                value=metric.get('value', 0),
                label=metric.get('label', ''),
                change=metric.get('change'),
                change_type=metric.get('change_type', 'neutral'),
                icon=metric.get('icon', '📊'),
                tooltip=metric.get('tooltip')
            )


def dh_sidebar_navigation(nav_items: List[Dict], current_page: str = ""):
    """
    Professional sidebar navigation
    
    Args:
        nav_items: List of navigation items [{"label": "Home", "icon": "🏠", "key": "home", "children": [...]}]
        current_page: Currently active page key
    """
    st.sidebar.markdown("""
    <div class="dh-sidebar-header">
        <h2 class="dh-heading-2">DataHound Pro</h2>
        <p class="dh-caption">Intelligence That Drives Growth</p>
    </div>
    """, unsafe_allow_html=True)
    
    for item in nav_items:
        is_active = item.get('key') == current_page
        icon = item.get('icon', '📄')
        label = item.get('label', '')
        
        if is_active:
            st.sidebar.markdown(f"""
            <div class="dh-nav-item active">
                {icon} {label}
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.sidebar.button(f"{icon} {label}", key=f"nav_{item.get('key')}"):
                return item.get('key')
        
        # Handle sub-navigation
        if item.get('children') and is_active:
            for child in item['children']:
                child_icon = child.get('icon', '•')
                child_label = child.get('label', '')
                if st.sidebar.button(f"  {child_icon} {child_label}", key=f"nav_{child.get('key')}"):
                    return child.get('key')
    
    return None


def dh_loading_spinner(text: str = "Loading..."):
    """Professional loading spinner"""
    return st.spinner(text)


def dh_empty_state(icon: str, title: str, description: str, action_label: str = "", action_key: str = ""):
    """
    Professional empty state display
    
    Args:
        icon: Large icon for empty state
        title: Empty state title
        description: Empty state description
        action_label: Optional action button label
        action_key: Optional action button key
    """
    st.markdown(f"""
    <div class="dh-empty-state" style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
        <h3 class="dh-heading-3">{title}</h3>
        <p class="dh-body-large" style="color: var(--text-secondary); margin-bottom: 2rem;">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if action_label:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_label, key=action_key, type="primary"):
                return True
    
    return False


def dh_breadcrumbs(path: List[str]):
    """
    Professional breadcrumb navigation
    
    Args:
        path: List of breadcrumb items ["Home", "Customers", "Profile"]
    """
    breadcrumb_html = " > ".join([f'<span class="dh-body-small">{item}</span>' for item in path])
    st.markdown(f'<div class="dh-breadcrumbs" style="margin-bottom: 1rem;">{breadcrumb_html}</div>', unsafe_allow_html=True)


def dh_professional_chart_theme():
    """Return a professional chart theme for Plotly"""
    return {
        'layout': {
            'font': {'family': 'Inter, sans-serif', 'size': 12},
            'colorway': ['#2E86AB', '#F18F01', '#10B981', '#EF4444', '#8B5CF6', '#F59E0B'],
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
            'xaxis': {
                'gridcolor': '#E5E7EB',
                'linecolor': '#D1D5DB',
                'tickcolor': '#9CA3AF',
                'tickfont': {'size': 11}
            },
            'yaxis': {
                'gridcolor': '#E5E7EB',
                'linecolor': '#D1D5DB',
                'tickcolor': '#9CA3AF',
                'tickfont': {'size': 11}
            }
        }
    }


def dh_create_metric_chart(data: Dict, chart_type: str = "bar", title: str = ""):
    """
    Create a professional metric chart
    
    Args:
        data: Chart data {"labels": [...], "values": [...]}
        chart_type: Type of chart ("bar", "pie", "line", "area")
        title: Chart title
    """
    theme = dh_professional_chart_theme()
    
    if chart_type == "bar":
        fig = px.bar(x=data['labels'], y=data['values'], title=title)
    elif chart_type == "pie":
        fig = px.pie(names=data['labels'], values=data['values'], title=title)
    elif chart_type == "line":
        fig = px.line(x=data['labels'], y=data['values'], title=title)
    elif chart_type == "area":
        fig = px.area(x=data['labels'], y=data['values'], title=title)
    else:
        fig = px.bar(x=data['labels'], y=data['values'], title=title)
    
    fig.update_layout(**theme['layout'])
    return fig


# CSS Injection Helper
def inject_custom_css():
    """Inject the custom CSS into the Streamlit app"""
    try:
        load_custom_css()
    except FileNotFoundError:
        # Fallback if CSS file doesn't exist
        st.markdown("""
        <style>
        .dh-heading-1 { font-size: 2.25rem; font-weight: 700; margin-bottom: 1.5rem; }
        .dh-heading-2 { font-size: 1.875rem; font-weight: 600; margin-bottom: 1rem; }
        .dh-heading-3 { font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; }
        .dh-card { background: white; border-radius: 0.75rem; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .dh-metric { text-align: center; }
        .dh-metric-value { font-size: 2.25rem; font-weight: 700; }
        .dh-metric-label { font-size: 0.875rem; color: #6B7280; text-transform: uppercase; }
        </style>
        """, unsafe_allow_html=True)


def dh_path_with_copy(path: Path, label: str = "Path", key: str = "path_copy") -> None:
    p = Path(path)
    try:
        uri = p.resolve().as_uri()
    except Exception:
        uri = f"file:///{p.as_posix()}"
    st.markdown(f'<div><strong>{label}:</strong> <a href="{uri}" target="_blank">{p.as_posix()}</a></div>', unsafe_allow_html=True)
    st.code(p.as_posix())
