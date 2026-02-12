"""Pipeline Monitor - real-time monitoring of all DataHound pipeline stages."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, date
from pathlib import Path as _P
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = _P(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._shared import select_company_config  # type: ignore
from datahound.dashboard import (  # type: ignore
    clear_logs,
    clear_cache,
    load_dashboard_data,
    compute_transcript_metrics,
    backfill_processed_customers,
)
BACKFILL_AVAILABLE = True

try:
    from apps.components.ui_components import (  # type: ignore
        inject_custom_css,
        dh_page_header,
        dh_alert,
        dh_professional_metric_grid,
        dh_data_table,
    )

    UI_COMPONENTS_AVAILABLE = True
except ImportError:
    UI_COMPONENTS_AVAILABLE = False


# ------------------------- Data Utilities -------------------------


def _load_data(company: str) -> Dict[str, pd.DataFrame]:
    raw = load_dashboard_data(company)
    aggregated: Dict[str, List[pd.DataFrame]] = {}
    for df in raw.values():
        if df.empty:
            continue
        stage = df["stage"].iloc[0] if "stage" in df.columns else "unknown"
        aggregated.setdefault(stage, []).append(df)

    stage_order = [
        "download",
        "prepare",
        "integrated_upsert",
        "integrated_upsert_operations",
        "core_data",
        "historical_scan",
        "recent_events",
        "custom_extraction",
        "scheduler",
        "change_log",
    ]

    result: Dict[str, pd.DataFrame] = {}
    for stage in stage_order:
        frames = aggregated.get(stage, [])
        if frames:
            df = pd.concat(frames, ignore_index=True, sort=False)
            if "ts" in df.columns:
                df.sort_values("ts", inplace=True)
            result[stage] = df
        else:
            result[stage] = pd.DataFrame()

    # Include any additional stages not in the predefined list
    for stage, frames in aggregated.items():
        if stage not in result:
            df = pd.concat(frames, ignore_index=True, sort=False)
            if "ts" in df.columns:
                df.sort_values("ts", inplace=True)
            result[stage] = df

    return result


def _stage_summary(df: pd.DataFrame, ts_col: str = "ts") -> Dict[str, pd.Series]:
    summary: Dict[str, pd.Series] = {}
    if df.empty:
        return summary
    grouped = df.groupby("stage")
    summary["counts"] = grouped.size()
    summary["last"] = grouped[ts_col].max()
    if "duration_seconds" in df.columns:
        summary["duration"] = grouped["duration_seconds"].mean()
    return summary


# ------------------------- Visualization Helpers -------------------------


def _value_or_zero(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _format_duration(seconds: float) -> str:
    if pd.isna(seconds) or seconds is None:
        return "0s"
    td = timedelta(seconds=float(seconds))
    total_seconds = int(td.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, sec = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _stage_health_card(stage: str, df: pd.DataFrame) -> Dict[str, str]:
    if df.empty:
        return {
            "label": stage.replace("_", " ").title(),
            "value": "No data",
            "icon": "⚪",
            "change_type": "neutral",
        }
    last_ts = df["ts"].max()
    run_count = len(df)
    icon = "🟢"
    change_type = "positive"
    if "level" in df.columns:
        if (df["level"].str.lower() == "error").any():
            icon = "🔴"
            change_type = "negative"
        elif (df["level"].str.lower() == "warning").any():
            icon = "🟡"
            change_type = "neutral"
    return {
        "label": stage.replace("_", " ").title(),
        "value": f"{run_count:,} runs" if run_count else "No runs",
        "change": last_ts.strftime("%Y-%m-%d %H:%M") if isinstance(last_ts, pd.Timestamp) else "",
        "icon": icon,
        "change_type": change_type,
    }


def _timeline_chart(df: pd.DataFrame, title: str):
    if df.empty:
        return None
    if "duration_seconds" not in df.columns:
        df = df.copy()
        df["duration_seconds"] = 0
    display_df = df[["ts", "stage", "duration_seconds"]].dropna()
    display_df = display_df.tail(200)
    display_df["end_ts"] = display_df["ts"] + pd.to_timedelta(display_df["duration_seconds"], unit="s")
    display_df["duration_display"] = display_df["duration_seconds"].map(_format_duration)
    fig = px.timeline(
        display_df,
        x_start="ts",
        x_end="end_ts",
        y="stage",
        color="stage",
        title=title,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=500, showlegend=False)
    return fig


def _pipeline_volume_chart(df: pd.DataFrame):
    if df.empty:
        return None
    if "ts" not in df.columns:
        return None
    daily_counts = df.copy()
    daily_counts["date"] = daily_counts["ts"].dt.date
    grouped = daily_counts.groupby(["date", "stage"]).size().reset_index(name="count")
    fig = px.bar(grouped, x="date", y="count", color="stage", title="Daily Pipeline Volume")
    fig.update_layout(height=400)
    return fig


def _error_table(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(columns=["ts", "stage", "message"])
    error_df = df[df.get("level", "info").str.lower().isin(["error", "warning"])]
    if error_df.empty:
        return pd.DataFrame(columns=["ts", "stage", "message"])
    columns = ["ts", "stage"] + [col for col in ["level", "message", "details"] if col in error_df.columns]
    return error_df[columns].tail(200)


# ------------------------- Dashboard Rendering -------------------------


def _render_overview(cards: List[Dict[str, str]]):
    if not cards:
        if UI_COMPONENTS_AVAILABLE:
            dh_alert("No pipeline runs detected.", "warning")
        else:
            st.warning("No pipeline runs detected.")
        return
    if UI_COMPONENTS_AVAILABLE:
        dh_professional_metric_grid(cards)
    else:
        cols = st.columns(len(cards))
        for col, card in zip(cols, cards):
            with col:
                st.metric(card["label"], card["value"], card.get("change"))


def _render_section_header(title: str, subtitle: str = ""):
    st.markdown("---")
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def _render_data_table(df: pd.DataFrame, title: str, columns: List[str]):
    if df.empty:
        st.info(f"No data for {title} yet.")
        return
    display_df = df[columns].tail(1000)
    if UI_COMPONENTS_AVAILABLE:
        dh_data_table(display_df, title=title, searchable=True)
    else:
        st.dataframe(display_df, width='stretch')


def _prepare_transcript_cards(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    totals = metrics.get("totals", {})
    cards = [
        {
            "label": "Transcripts",
            "value": f"{totals.get('transcripts_processed', 0):,}",
            "icon": "📝",
            "change_type": "neutral",
            "tooltip": "Total number of individual transcripts the system has processed within the selected date range.",
        },
        {
            "label": "Unique Callers",
            "value": f"{totals.get('unique_callers_processed', 0):,}",
            "icon": "☎️",
            "change_type": "neutral",
            "tooltip": "Distinct phone numbers whose transcripts were analyzed during the selected period.",
        },
        {
            "label": "Leads Analyzed",
            "value": f"{totals.get('callers_with_lead_data', 0):,}",
            "icon": "🔍",
            "change_type": "neutral",
            "tooltip": "Callers that have AI lead analysis records linking back to the processed transcripts.",
        },
    ]
    return cards


def _render_transcript_metrics(metrics: Optional[Dict[str, Any]]):
    _render_section_header("Transcript Funnel", "Second chance lead call classification summary")
    if not metrics:
        st.info("Transcript metrics unavailable.")
        return

    cards = _prepare_transcript_cards(metrics)
    if UI_COMPONENTS_AVAILABLE:
        dh_professional_metric_grid(cards)
    else:
        cols = st.columns(len(cards))
        for col, card in zip(cols, cards):
            with col:
                st.metric(card["label"], card["value"], help=card.get("tooltip"))

    columns = st.columns(3)
    sections = [
        (
            "Customers",
            metrics.get("customers", {}),
            "Callers the AI confirmed were actual customers seeking HVAC help.",
        ),
        (
            "Service Requests",
            metrics.get("service_requests", {}),
            "Customer calls that included a request for service, repair, or maintenance.",
        ),
        (
            "Bookings",
            metrics.get("bookings", {}),
            "Service-requesting customers whose jobs were successfully booked.",
        ),
    ]
    for column, (title, values, description) in zip(columns, sections):
        with column:
            st.metric(
                title,
                f"{values.get('true', 0):,} true",
                help=f"{description} False: {values.get('false', 0):,} | Unknown: {values.get('unknown', 0):,}"
            )

    funnel_counts = metrics.get("funnel", {})
    if funnel_counts:
        funnel_fig = go.Figure(
            go.Funnel(
                y=[
                    "Transcripts",
                    "Callers",
                    "Customers",
                    "Service Requests",
                    "Not Booked",
                ],
                x=[
                    metrics.get("totals", {}).get("transcripts_processed", 0),
                    metrics.get("totals", {}).get("unique_callers_processed", 0),
                    funnel_counts.get("customers", 0),
                    funnel_counts.get("service_requests", 0),
                    funnel_counts.get("service_requests", 0) - funnel_counts.get("booked", 0),
                ],
            )
        )
        funnel_fig.update_layout(height=350, title="Conversion Funnel")
        st.plotly_chart(funnel_fig, width='stretch')

    timeline_records = metrics.get("timeline", {}).get("daily", [])
    if timeline_records:
        timeline_df = pd.DataFrame([
            {
                "date": record.get("date"),
                "customers": record.get("customers", {}).get("true", 0),
                "service_requests": record.get("service_requests", {}).get("true", 0),
                "bookings": record.get("bookings", {}).get("true", 0),
                "not_booked": record.get("not_booked", {}).get("true", 0),
            }
            for record in timeline_records
        ])
        timeline_df["date"] = pd.to_datetime(timeline_df["date"], errors="coerce")
        timeline_df.sort_values("date", inplace=True)
        melted = timeline_df.melt("date", var_name="stage", value_name="count")
        timeline_fig = px.line(melted, x="date", y="count", color="stage", title="Daily Transcript Outcomes")
        st.plotly_chart(timeline_fig, width='stretch')

    customer_details = metrics.get("customer_details", [])
    if customer_details:
        details_df = pd.DataFrame(customer_details)
        for column in ["was_customer_call", "was_service_request", "was_booked"]:
            if column in details_df.columns:
                details_df[column] = details_df[column].map(
                    lambda value: "TRUE" if value is True else "FALSE" if value is False else "UNKNOWN"
                )
        _render_data_table(details_df, "Customer Classification Detail", details_df.columns.tolist())


# ------------------------- Main -------------------------


def main():
    st.set_page_config(page_title="DataHound Pro - Pipeline Monitor", page_icon="📊", layout="wide")
    if UI_COMPONENTS_AVAILABLE:
        inject_custom_css()

    if UI_COMPONENTS_AVAILABLE:
        dh_page_header("Pipeline Monitor", "Real-time pipeline stage monitoring")
    else:
        st.title("Pipeline Monitor")
        st.caption("Real-time pipeline stage monitoring")

    company, _ = select_company_config()
    if UI_COMPONENTS_AVAILABLE:
        dh_alert(f"Active Company: {company}", "success")
    else:
        st.success(f"Active Company: {company}")

    auto_refresh = st.sidebar.checkbox("Auto-refresh every 60s", value=False)
    if auto_refresh:
        st_autorefresh = st.empty()
        with st_autorefresh:
            st.caption("Auto-refresh enabled")
        import time

        time.sleep(60)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Log Retention")
    st.sidebar.caption("Use the controls below to trim pipeline logs and start monitoring from a clean slate.")
    cutoff_mode = st.sidebar.radio(
        "Keep entries",
        ("newer", "older"),
        index=0,
        help="Retain entries newer than or older than the timestamp",
    )
    cutoff_date = st.sidebar.date_input("Cut-off date")
    cutoff_time = st.sidebar.time_input("Cut-off time")
    flush_clicked = st.sidebar.button("Flush logs", width='stretch')
    refresh_clicked = st.sidebar.button("Refresh dashboard data", width='stretch')
    if flush_clicked:
        cutoff_dt = datetime.combine(cutoff_date, cutoff_time)
        direction = "after" if cutoff_mode == "older" else "before"
        clear_logs(company, cutoff=cutoff_dt, direction=direction)
        clear_cache()
        st.session_state["dashboard_should_refresh"] = True
        st.sidebar.success("Log files flushed with new retention window.")
    if refresh_clicked:
        clear_cache()
        st.session_state["dashboard_should_refresh"] = True

    if st.session_state.get("dashboard_should_refresh"):
        st.session_state.pop("dashboard_should_refresh")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Transcript Metrics")
    use_start_filter = st.sidebar.checkbox("Filter by start date", value=False, key="transcript_start_toggle")
    transcript_start = (
        st.sidebar.date_input("Start date", value=date.today(), key="transcript_start_input")
        if use_start_filter
        else None
    )
    use_end_filter = st.sidebar.checkbox("Filter by end date", value=False, key="transcript_end_toggle")
    transcript_end = (
        st.sidebar.date_input("End date", value=date.today(), key="transcript_end_input")
        if use_end_filter
        else None
    )
    backfill_enabled = st.sidebar.checkbox("Backfill processed customers", value=False, disabled=not BACKFILL_AVAILABLE)
    transcript_refresh = st.sidebar.button("Refresh transcript stats", width='stretch')

    data = _load_data(company)
    transcript_metrics: Optional[Dict] = None

    start_date: Optional[date] = transcript_start if isinstance(transcript_start, date) else None
    end_date: Optional[date] = transcript_end if isinstance(transcript_end, date) else None
    current_filter = (
        start_date.isoformat() if start_date else None,
        end_date.isoformat() if end_date else None,
    )
    cached_filter = st.session_state.get("transcript_metrics_filters")

    if (
        transcript_refresh
        or "transcript_metrics_cache" not in st.session_state
        or cached_filter != current_filter
    ):
        if backfill_enabled and BACKFILL_AVAILABLE:
            with st.spinner("Backfilling processed customers..."):
                result = backfill_processed_customers(company)
                if result.get("backfilled", 0) > 0:
                    st.sidebar.success(f"Backfilled {result['backfilled']} customers")
                else:
                    st.sidebar.info(f"Backfill: {result.get('reason', 'no new customers')}")
        try:
            transcript_metrics = compute_transcript_metrics(start_date=start_date, end_date=end_date, company=company)
            st.session_state["transcript_metrics_cache"] = transcript_metrics
            st.session_state["transcript_metrics_filters"] = current_filter
        except FileNotFoundError:
            st.sidebar.warning("Transcript data files not found.")
            st.session_state["transcript_metrics_cache"] = None
            st.session_state["transcript_metrics_filters"] = current_filter
    else:
        transcript_metrics = st.session_state.get("transcript_metrics_cache")

    primary_stages = [
        "download",
        "prepare",
        "integrated_upsert",
        "core_data",
        "historical_scan",
        "recent_events",
        "custom_extraction",
        "scheduler",
    ]

    overview_cards = [_stage_health_card(stage, data.get(stage, pd.DataFrame())) for stage in primary_stages]
    _render_overview(overview_cards)

    combined_df = pd.concat(data.values(), ignore_index=True, sort=False)
    _render_section_header("Pipeline Activity", "Timeline of recent stage executions")
    timeline_fig = _timeline_chart(combined_df, "Pipeline Stage Timeline")
    if timeline_fig:
        st.plotly_chart(timeline_fig, width='stretch')
    else:
        st.info("No timeline data available yet.")

    _render_transcript_metrics(transcript_metrics)

    _render_section_header("Volume Trends", "Daily volume by stage")
    volume_fig = _pipeline_volume_chart(combined_df)
    if volume_fig:
        st.plotly_chart(volume_fig, width='stretch')
    else:
        st.info("No volume data available yet.")

    _render_section_header("Recent Errors & Warnings")
    error_df = _error_table(combined_df)
    if not error_df.empty:
        _render_data_table(error_df, "Pipeline Errors", error_df.columns.tolist())
    else:
        st.success("No recent errors detected.")

    st.markdown("### Stage Detail Views")
    stage_tabs = st.tabs([stage.replace("_", " ").title() for stage in data.keys()])
    for tab, (stage, df) in zip(stage_tabs, data.items()):
        with tab:
            if df.empty:
                st.info(f"No entries for {stage}.")
                continue
            columns = [col for col in ["ts", "level", "message", "details", "file_type", "status"] if col in df.columns]
            if not columns:
                columns = df.columns.tolist()
            _render_data_table(df, f"{stage.replace('_', ' ').title()} Activity", columns)

    _render_section_header("Service Logs", "Live output from background services")
    _render_service_logs()

    st.markdown("---")
    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _render_service_logs():
    log_dir = _P(__file__).resolve().parents[2] / "data" / "service_logs"
    if not log_dir.exists():
        st.info("No service logs yet. Start services from Admin > Services.")
        return

    log_files = {
        "transcript_pipeline": "Second Chance Lead Detection",
        "event_upload": "Event Upload to Google Sheets",
        "sms_sheet_sync": "SMS Activity Sync",
    }

    available = []
    for key, label in log_files.items():
        path = log_dir / f"{key}.log"
        if path.exists():
            available.append((key, label, path))

    if not available:
        st.info("No service logs found yet.")
        return

    tabs = st.tabs([label for _, label, _ in available])
    for tab, (key, label, path) in zip(tabs, available):
        with tab:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                size_kb = path.stat().st_size / 1024
                modified = datetime.fromtimestamp(path.stat().st_mtime)
                st.caption(f"Log file: {path.name} | {size_kb:.0f} KB | Last updated: {modified.strftime('%Y-%m-%d %H:%M:%S')}")

                tail_lines = st.slider(
                    "Lines to show", min_value=50, max_value=2000, value=200,
                    step=50, key=f"svc_log_lines_{key}",
                )
                lines = content.splitlines()
                shown = lines[-tail_lines:]
                st.code("\n".join(shown), language="text")
            except Exception as e:
                st.error(f"Error reading log: {e}")

    if st.button("Refresh Logs", key="svc_log_refresh"):
        st.rerun()


if __name__ == "__main__":
    main()

