from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

try:
    import polars as pl
    import gspread
    from dotenv import load_dotenv
    from oauth2client.service_account import ServiceAccountCredentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EVENT_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "cancellations": {
        "title": "Cancellations",
        "icon": "🚫",
        "summary": (
            "Jobs that were scheduled but then canceled. These represent opportunities "
            "to win back business by following up with the customer to understand why "
            "the job was canceled and whether they still need the service."
        ),
        "formula": (
            "**Conversion Rate** = Customers who booked a new job after cancellation "
            "/ Total cancellations detected"
        ),
        "master_file": "canceled_jobs_master.parquet",
    },
    "unsold_estimates": {
        "title": "Unsold Estimates",
        "icon": "📝",
        "summary": (
            "Estimates that were created but never converted into sales - they remain "
            "'Open' or were 'Dismissed' by the customer. Following up on these can help "
            "convert potential revenue that didn't initially materialize."
        ),
        "formula": (
            "**Conversion Rate** = Estimates that eventually led to a booked job "
            "/ Total unsold estimates detected"
        ),
        "master_file": "unsold_estimates_master.parquet",
    },
    "overdue_maintenance": {
        "title": "Overdue Maintenance",
        "icon": "🔧",
        "summary": (
            "Customers or locations that haven't had maintenance service in 12+ months. "
            "Severity ranges from Medium (15-18 months) to Critical (24+ months). "
            "These customers are at risk for bigger problems and may be considering "
            "other service providers."
        ),
        "formula": (
            "**Conversion Rate** = Customers who completed maintenance after being flagged "
            "/ Total overdue maintenance cases detected"
        ),
        "master_file": "overdue_maintenance_master.parquet",
    },
    "lost_customers": {
        "title": "Lost Customers",
        "icon": "👋",
        "summary": (
            "Customers who used to work with us but are now using competitors for "
            "HVAC work, detected through public building permit records. A customer "
            "is classified as 'lost' when a competitor completes work at their address "
            "after our last service date."
        ),
        "formula": (
            "**Conversion Rate** = Lost customers who returned and booked a job "
            "/ Total lost customers detected"
        ),
        "master_file": "lost_customers_master.parquet",
    },
    "second_chance_leads": {
        "title": "Second Chance Leads",
        "icon": "📞",
        "summary": (
            "Customers who called requesting service but didn't end up booking an "
            "appointment. AI analysis of call transcripts identifies these leads by "
            "confirming: (1) it was a customer call, (2) they requested service, "
            "(3) the service was NOT booked. Invalid reasons (reschedule, parts "
            "confirmation, etc.) are automatically filtered out."
        ),
        "formula": (
            "**Conversion Rate** = Second chance leads who eventually booked a job "
            "/ Total second chance leads detected (within 12-month window)"
        ),
        "master_file": None,
    },
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_event_master(company: str, master_file: str) -> pd.DataFrame:
    paths = [
        Path("data") / company / "events" / "master_files" / master_file,
        Path("companies") / company / "parquet" / master_file,
    ]
    for p in paths:
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                continue
    return pd.DataFrame()


def _load_jobs_pd(company: str) -> pd.DataFrame:
    path = Path("companies") / company / "parquet" / "Jobs.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _normalize_col(name: str) -> str:
    return re.sub(r'(?<=[a-z])(?=[A-Z])', '_', name).lower().replace(' ', '_')


def _get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {_normalize_col(c): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        norm = _normalize_col(c)
        if norm in norm_map:
            return norm_map[norm]
    return None


# ---------------------------------------------------------------------------
# Generic conversion calculation
# ---------------------------------------------------------------------------

def _calculate_conversion(
    events_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    event_date_col: str,
    event_customer_col: str,
    jobs_date_col: str,
    jobs_customer_col: str,
    lookback_days: int = 365,
) -> Dict[str, Any]:
    if events_df.empty or jobs_df.empty:
        return {"total": len(events_df), "converted": 0, "rate": 0.0}

    events_df = events_df.copy()
    jobs_df = jobs_df.copy()
    events_df[event_date_col] = pd.to_datetime(events_df[event_date_col], errors="coerce")
    jobs_df[jobs_date_col] = pd.to_datetime(jobs_df[jobs_date_col], errors="coerce")
    events_df[event_customer_col] = events_df[event_customer_col].astype(str).str.strip()
    jobs_df[jobs_customer_col] = jobs_df[jobs_customer_col].astype(str).str.strip()

    converted = 0
    for _, event in events_df.iterrows():
        cust_id = event[event_customer_col]
        event_date = event[event_date_col]
        if pd.isna(event_date) or not cust_id or cust_id in ("", "nan", "None"):
            continue
        end_date = event_date + timedelta(days=lookback_days)
        matching = jobs_df[
            (jobs_df[jobs_customer_col] == cust_id)
            & (jobs_df[jobs_date_col] > event_date)
            & (jobs_df[jobs_date_col] <= end_date)
        ]
        status_col = _get_col(matching, ["status", "Status", "Jobs Status"])
        if status_col and not matching.empty:
            ok = matching[
                matching[status_col].astype(str).str.lower()
                .isin(["completed", "scheduled", "in progress", "booked"])
            ]
            if not ok.empty:
                converted += 1
        elif not matching.empty:
            converted += 1

    total = len(events_df)
    rate = (converted / total * 100) if total > 0 else 0.0
    return {"total": total, "converted": converted, "rate": rate}


def _render_event_info(event_key: str) -> None:
    info = EVENT_DESCRIPTIONS[event_key]
    st.markdown(f"### {info['icon']} {info['title']}")
    st.markdown(info["summary"])
    st.info(info["formula"])


# ---------------------------------------------------------------------------
# Generic event dashboard
# ---------------------------------------------------------------------------

def _render_generic_dashboard(company: str, event_key: str) -> None:
    info = EVENT_DESCRIPTIONS[event_key]
    _render_event_info(event_key)

    master_file = info.get("master_file")
    if not master_file:
        st.warning("No master data file configured for this event type.")
        return

    events_df = _load_event_master(company, master_file)
    if events_df.empty:
        st.info(f"No {info['title'].lower()} data found. Run event detection first.")
        return

    st.metric(f"Total {info['title']} Detected", f"{len(events_df):,}")
    jobs_df = _load_jobs_pd(company)

    event_date_col = _get_col(events_df, [
        "event_date", "created_date", "date", "cancellation_date",
        "detection_date", "last_maintenance_date",
    ])
    event_cust_col = _get_col(events_df, ["customer_id", "entity_id", "cust_id"])
    jobs_date_col = _get_col(jobs_df, ["created_date", "Jobs Created Date", "completion_date"]) if not jobs_df.empty else None
    jobs_cust_col = _get_col(jobs_df, ["customer_id", "Jobs Customer Id", "Customers Id"]) if not jobs_df.empty else None

    if event_date_col and event_cust_col and jobs_date_col and jobs_cust_col:
        with st.spinner("Calculating conversion rates..."):
            conversion = _calculate_conversion(
                events_df, jobs_df,
                event_date_col, event_cust_col,
                jobs_date_col, jobs_cust_col,
            )
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Events", f"{conversion['total']:,}")
        c2.metric("Converted", f"{conversion['converted']:,}")
        c3.metric("Conversion Rate", f"{conversion['rate']:.1f}%")
    else:
        st.info("Unable to calculate conversion rates - required columns not found.")

    st.markdown("#### Data Overview")
    display_cols = []
    for c in ["customer_id", "entity_id", "event_date", "created_date", "status", "severity", "customer_name", "location_id"]:
        found = _get_col(events_df, [c])
        if found and found not in display_cols:
            display_cols.append(found)
    if not display_cols:
        display_cols = list(events_df.columns[:8])

    if event_date_col and event_date_col in events_df.columns:
        events_df[event_date_col] = pd.to_datetime(events_df[event_date_col], errors="coerce")
        events_sorted = events_df.sort_values(event_date_col, ascending=False, na_position="last")
    else:
        events_sorted = events_df

    st.dataframe(events_sorted[display_cols].head(200), width="stretch")


# ---------------------------------------------------------------------------
# Cancellations dashboard (with historical JSONL fallback)
# ---------------------------------------------------------------------------

def _render_cancellations_dashboard(company: str) -> None:
    _render_event_info("cancellations")
    _render_generic_dashboard(company, "cancellations")


# ---------------------------------------------------------------------------
# Second Chance Leads - Google Sheets integration
# ---------------------------------------------------------------------------

def _get_gspread_client() -> Any:
    if not GSHEETS_AVAILABLE:
        return None
    load_dotenv()
    credentials_path = os.environ.get("GOOGLE_CREDS")
    if not credentials_path:
        return None
    path_obj = Path(credentials_path)
    if not path_obj.is_absolute():
        path_obj = ROOT / credentials_path
    if not path_obj.exists():
        return None
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(str(path_obj), scope)
    return gspread.authorize(credentials)


@st.cache_data(ttl=300)
def _load_second_chance_leads_from_sheets() -> Optional[Any]:
    if not GSHEETS_AVAILABLE:
        return None
    try:
        load_dotenv()
        sheet_id = os.environ.get("SECOND_CHANCE_SHEET_ID") or os.environ.get("RECENT_SECOND_CHANCE_LEADS_SHEET_ID")
        if not sheet_id:
            return None
        client = _get_gspread_client()
        if not client:
            return None
        spreadsheet = client.open_by_key(sheet_id)

        all_frames = []
        for sheet_name in ["Second Chance Leads", "Invalid Events"]:
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
                values = worksheet.get_all_values()
                if not values or len(values) < 2:
                    continue
                headers = values[0]
                rows = values[1:]
                df = pd.DataFrame(rows, columns=headers)
                df["source_tab"] = sheet_name
                all_frames.append(df)
            except Exception:
                continue

        if not all_frames:
            return None

        combined = pd.concat(all_frames, ignore_index=True)
        frame = pl.from_pandas(combined)
        if frame.is_empty():
            return frame

        date_columns = ["Analysis Timestamp", "Call Date", "Detected At", "Updated At"]
        entry_date_set = False
        for col in date_columns:
            if col in frame.columns and not entry_date_set:
                if col == "Analysis Timestamp":
                    frame = frame.with_columns(
                        pl.col(col).cast(pl.Utf8)
                        .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False)
                        .alias("entry_date")
                    )
                elif col == "Call Date":
                    frame = frame.with_columns(
                        pl.col(col).cast(pl.Utf8)
                        .str.strptime(pl.Date, "%m/%d/%Y", strict=False)
                        .dt.cast_time_unit("us")
                        .dt.replace_time_zone(None)
                        .alias("entry_date")
                    )
                else:
                    frame = frame.with_columns(
                        pl.col(col).cast(pl.Utf8)
                        .str.strptime(pl.Datetime, strict=False)
                        .alias("entry_date")
                    )
                entry_date_set = True

        if not entry_date_set:
            frame = frame.with_columns(pl.lit(None).cast(pl.Datetime).alias("entry_date"))

        frame = frame.filter(pl.col("entry_date").is_not_null())

        if "Customer ID" in frame.columns:
            frame = frame.with_columns(pl.col("Customer ID").cast(pl.Utf8, strict=False))
        if "Customer Phone" in frame.columns:
            frame = frame.with_columns(
                pl.col("Customer Phone").cast(pl.Utf8)
                .str.replace_all(r"[^0-9]", "")
                .alias("normalized_phone")
            )
        return frame
    except Exception as e:
        st.error(f"Error loading from Google Sheets: {e}")
        return None


def _load_jobs_polars(company: str) -> Any:
    if not GSHEETS_AVAILABLE:
        return None
    path = Path("companies") / company / "parquet" / "Jobs.parquet"
    if not path.exists():
        return None
    try:
        frame = pl.read_parquet(path)
        mapping = {c: re.sub(r"[^0-9A-Za-z]+", "_", c.strip()).strip("_").lower() for c in frame.columns}
        frame = frame.rename(mapping)

        phone_col = None
        for c in ["customer_phone", "phone"]:
            if c in frame.columns:
                phone_col = c
                break
        if phone_col:
            frame = frame.with_columns(
                pl.col(phone_col).cast(pl.Utf8).str.replace_all(r"[^0-9]", "").alias("normalized_phone")
            )
        else:
            frame = frame.with_columns(pl.lit("").cast(pl.Utf8).alias("normalized_phone"))

        if "customer_id" in frame.columns:
            frame = frame.with_columns(pl.col("customer_id").cast(pl.Utf8, strict=False))

        for dt_col in ["created_date", "completion_date", "sold_on", "scheduled_date"]:
            if dt_col in frame.columns:
                frame = frame.with_columns(
                    pl.col(dt_col).cast(pl.Utf8).str.strptime(pl.Datetime, strict=False).alias(dt_col)
                )
        return frame
    except Exception:
        return None


def _find_first_conversion(lead: dict, jobs_df: Any, entry_date: datetime) -> Optional[datetime]:
    if jobs_df is None or jobs_df.is_empty():
        return None
    end_date = entry_date + timedelta(days=365)
    customer_id = lead.get("Customer ID")
    normalized_phone = lead.get("normalized_phone", "")

    jobs_filter = (
        (pl.col("created_date") >= entry_date)
        & (pl.col("created_date") <= end_date)
    )

    if customer_id and str(customer_id).strip() and "customer_id" in jobs_df.columns:
        jobs_filter = jobs_filter & (pl.col("customer_id") == str(customer_id))
    elif normalized_phone and str(normalized_phone).strip() and "normalized_phone" in jobs_df.columns:
        jobs_filter = jobs_filter & (pl.col("normalized_phone") == str(normalized_phone))
    else:
        return None

    booked = jobs_df.filter(
        jobs_filter
        & pl.col("status").str.to_lowercase().is_in(["completed", "scheduled", "in progress", "booked"])
    ).sort("created_date")

    if len(booked) > 0:
        return booked.select(pl.col("created_date").min()).item()
    return None


def _format_period(date: datetime, period_type: str) -> str:
    if period_type == "week":
        week_start = date - timedelta(days=date.weekday())
        return week_start.strftime("%Y-%m-%d")
    elif period_type == "month":
        return date.strftime("%Y-%m")
    elif period_type == "quarter":
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}-Q{quarter}"
    return str(date.year)


def _calculate_scl_performance(leads_df: Any, jobs_df: Any, period_type: str) -> Any:
    lead_data = []
    for row in leads_df.iter_rows(named=True):
        entry_date = row.get("entry_date")
        if entry_date is None:
            continue
        conversion_date = _find_first_conversion(row, jobs_df, entry_date)
        entry_period = _format_period(entry_date, period_type)
        conversion_period = _format_period(conversion_date, period_type) if conversion_date else None
        lead_data.append({
            "entry_date": entry_date,
            "entry_period": entry_period,
            "conversion_date": conversion_date,
            "conversion_period": conversion_period,
        })

    if not lead_data:
        return pl.DataFrame()

    leads_conv = pl.DataFrame(lead_data)
    all_periods = set()
    for row in leads_conv.iter_rows(named=True):
        all_periods.add(row["entry_period"])
        if row["conversion_period"]:
            all_periods.add(row["conversion_period"])
    all_periods = sorted(all_periods)

    period_summary = []
    for period_str in all_periods:
        if period_type == "week":
            period_start = datetime.strptime(period_str, "%Y-%m-%d")
            period_end = period_start + timedelta(days=7)
        elif period_type == "month":
            period_start = datetime.strptime(period_str + "-01", "%Y-%m-%d")
            period_end = (datetime(period_start.year, period_start.month + 1, 1)
                          if period_start.month < 12
                          else datetime(period_start.year + 1, 1, 1))
        elif period_type == "quarter":
            year, q = period_str.split("-Q")
            month = (int(q) - 1) * 3 + 1
            period_start = datetime(int(year), month, 1)
            period_end = (datetime(period_start.year + 1, 1, 1)
                          if period_start.month == 10
                          else datetime(period_start.year, period_start.month + 3, 1))
        else:
            period_start = datetime(int(period_str), 1, 1)
            period_end = datetime(period_start.year + 1, 1, 1)

        eligible_start = period_start - timedelta(days=365)

        detected = leads_conv.filter(pl.col("entry_period") == period_str)
        conversions = leads_conv.filter(pl.col("conversion_period") == period_str)
        eligible = leads_conv.filter(
            (pl.col("entry_date") >= pl.lit(eligible_start))
            & (pl.col("entry_date") < pl.lit(period_end))
        )

        total_leads = len(detected)
        conv_count = len(conversions)
        eligible_pool = len(eligible)
        rate = (conv_count / eligible_pool * 100) if eligible_pool > 0 else 0.0

        period_summary.append({
            "period": period_str,
            "total_leads": total_leads,
            "conversions": conv_count,
            "eligible_pool": eligible_pool,
            "conversion_rate": rate,
        })

    if not period_summary:
        return pl.DataFrame()
    return pl.DataFrame(period_summary).sort("period")


def _render_second_chance_leads_dashboard(company: str) -> None:
    _render_event_info("second_chance_leads")

    if not GSHEETS_AVAILABLE:
        st.warning(
            "Google Sheets integration requires: `polars`, `gspread`, `python-dotenv`, `oauth2client`. "
            "Install them and set `GOOGLE_CREDS` + `SECOND_CHANCE_SHEET_ID` environment variables."
        )
        return

    with st.spinner("Loading data from Google Sheets..."):
        leads_df = _load_second_chance_leads_from_sheets()
        jobs_df = _load_jobs_polars(company)

    if leads_df is None or leads_df.is_empty():
        st.warning(
            "No second chance leads found. Ensure `GOOGLE_CREDS` and `SECOND_CHANCE_SHEET_ID` "
            "environment variables are set correctly."
        )
        return

    period_type = st.selectbox(
        "View By", ["week", "month", "quarter", "year"], index=1,
        key="scl_period_type",
    )

    min_date_val = None
    max_date_val = None
    if "entry_date" in leads_df.columns:
        mn = leads_df.select(pl.col("entry_date").min()).item()
        mx = leads_df.select(pl.col("entry_date").max()).item()
        if mn:
            min_date_val = mn.date() if isinstance(mn, datetime) else mn
        if mx:
            max_date_val = mx.date() if isinstance(mx, datetime) else mx

    col1, col2 = st.columns(2)
    with col1:
        min_entry = st.date_input("Min Entry Date", value=min_date_val or datetime.now().date(), key="scl_min_date")
    with col2:
        max_entry = st.date_input("Max Entry Date", value=max_date_val or datetime.now().date(), key="scl_max_date")

    filtered = leads_df
    if "entry_date" in filtered.columns:
        min_dt = datetime.combine(min_entry, datetime.min.time())
        max_dt = datetime.combine(max_entry, datetime.max.time())
        filtered = filtered.filter(
            (pl.col("entry_date") >= min_dt) & (pl.col("entry_date") <= max_dt)
        )

    st.metric("Total Leads Tracked", len(filtered))

    with st.spinner("Calculating performance..."):
        perf = _calculate_scl_performance(filtered, jobs_df, period_type)

    if perf.is_empty():
        st.info("No performance data for the selected filters.")
        return

    period_label = period_type.capitalize()
    total_leads = perf.select(pl.col("total_leads").sum()).item() or 0
    total_conv = perf.select(pl.col("conversions").sum()).item() or 0
    overall_rate = (total_conv / total_leads * 100) if total_leads > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Leads Detected", f"{total_leads:,}")
    c2.metric("Total Converted", f"{total_conv:,}")
    c3.metric("Overall Conversion Rate", f"{overall_rate:.1f}%")

    st.subheader(f"{period_label}ly Breakdown")

    if ALTAIR_AVAILABLE:
        chart_data = perf.to_pandas()
        melted = pd.melt(
            chart_data,
            id_vars=["period", "conversion_rate", "eligible_pool"],
            value_vars=["total_leads", "conversions"],
            var_name="metric_type", value_name="count",
        )
        melted["metric_label"] = melted["metric_type"].map({
            "total_leads": "Leads Detected", "conversions": "Converted",
        })

        bars = alt.Chart(melted).mark_bar(opacity=0.85).encode(
            x=alt.X("period:O", title=f"{period_label}", axis=alt.Axis(labelAngle=-45)),
            xOffset=alt.XOffset("metric_label:N"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color(
                "metric_label:N",
                scale=alt.Scale(domain=["Leads Detected", "Converted"], range=["#1f77b4", "#ff7f0e"]),
            ),
            tooltip=["period:O", "metric_label:N", "count:Q", "conversion_rate:Q"],
        )
        line = alt.Chart(chart_data).mark_line(point=True, color="#2ca02c", strokeWidth=3).encode(
            x=alt.X("period:O"),
            y=alt.Y("conversion_rate:Q", title="Conversion Rate (%)"),
            tooltip=["period:O", "conversions:Q", "eligible_pool:Q", "conversion_rate:Q"],
        )
        combined = (
            alt.layer(bars, line)
            .resolve_scale(y="independent")
            .properties(title=f"Second Chance Leads by {period_label}", height=500)
        )
        st.altair_chart(combined, width="stretch")

    display_df = perf.with_columns(pl.col("conversion_rate").round(2))
    st.dataframe(display_df.to_pandas(), width="stretch")

    csv = display_df.to_pandas().to_csv(index=False)
    st.download_button(
        label=f"Download {period_label}ly Performance Data",
        data=csv,
        file_name=f"second_chance_leads_{period_type}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="scl_download",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_all_dashboards(company: str) -> None:
    event_keys = list(EVENT_DESCRIPTIONS.keys())
    labels = [f"{EVENT_DESCRIPTIONS[k]['icon']} {EVENT_DESCRIPTIONS[k]['title']}" for k in event_keys]
    tabs = st.tabs(labels)

    for tab, key in zip(tabs, event_keys):
        with tab:
            if key == "second_chance_leads":
                _render_second_chance_leads_dashboard(company)
            elif key == "cancellations":
                _render_cancellations_dashboard(company)
            else:
                _render_generic_dashboard(company, key)
