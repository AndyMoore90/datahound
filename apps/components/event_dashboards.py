from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, date
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

BOOKED_STATUSES = {"completed", "scheduled", "in progress", "booked"}

# Each event type uses a different column for the "real" event date and
# customer ID because the data structure differs per event.
#
# detected_at is a batch-scan timestamp (same for all rows) and is NOT
# suitable for period analysis.  The actual business date lives in a
# different column for each event:
#   cancellations       -> completion_date  (date the job was canceled)
#   unsold_estimates    -> creation_date    (date the estimate was created)
#   overdue_maintenance -> completion_date  (date of LAST maintenance)
#   lost_customers      -> last_contact_date (date we last heard from them)

EVENT_META: Dict[str, Dict[str, Any]] = {
    "cancellations": {
        "title": "Cancellations",
        "icon": "🚫",
        "summary": (
            "Jobs that were scheduled but then canceled. These represent "
            "opportunities to win back business by following up with the "
            "customer to understand why the job was canceled."
        ),
        "formula": (
            "**Conversion Rate** = Customers who booked a *different* job "
            "within 12 months of cancellation / Total cancellations"
        ),
        "master_file": "canceled_jobs_master.parquet",
        "event_date_col": "completion_date",
        "customer_id_col": "Customer ID",
        "entity_id_col": "entity_id",
        "detected_label": "Cancellations",
        "converted_label": "Rebooked",
        "detail": {
            "how_detected": (
                "The system scans **Jobs.parquet** for any job where the `Status` "
                "column equals **Canceled** (or Cancelled). Each canceled job becomes "
                "one event record. The `completion_date` field records when the "
                "cancellation happened, and `Customer ID` links back to the customer."
            ),
            "how_converted": (
                "A cancellation is considered **converted** (rebooked) when the same "
                "customer — matched by `Customer ID` — books a **different** job within "
                "12 months after the cancellation date. The canceled job itself is "
                "excluded so we don't count the original record. A 'booked' job is any "
                "job with a status of Completed, Scheduled, In Progress, or Booked."
            ),
            "data_source": (
                "- **Event data**: `data/{company}/events/master_files/canceled_jobs_master.parquet`\n"
                "- **Conversion check**: `companies/{company}/parquet/Jobs.parquet`\n"
                "- **Key columns in event file**: `completion_date` (cancellation date), "
                "`Customer ID`, `entity_id` (= Job ID of the canceled job), `Job Class`, `Summary`\n"
                "- **Key columns in Jobs**: `Created Date`, `Customer ID`, `Job ID`, `Status`"
            ),
            "example": (
                "Customer **103353551** had job **137259212** canceled on **2025-02-12**. "
                "Within the next 12 months, the system found 2 other booked jobs for the "
                "same customer in Jobs.parquet — so this cancellation counts as **converted**."
            ),
        },
    },
    "unsold_estimates": {
        "title": "Unsold Estimates",
        "icon": "📝",
        "summary": (
            "Estimates that were created but never converted into sales. "
            "Following up on these can convert potential revenue that "
            "didn't initially materialize."
        ),
        "formula": (
            "**Conversion Rate** = Customers who booked a job within "
            "12 months of the unsold estimate / Total unsold estimates"
        ),
        "master_file": "unsold_estimates_master.parquet",
        "event_date_col": "creation_date",
        "customer_id_col": "Customer ID",
        "entity_id_col": "entity_id",
        "detected_label": "Unsold Estimates",
        "converted_label": "Converted",
        "detail": {
            "how_detected": (
                "The system scans **Estimates.parquet** for estimates with a status of "
                "**Dismissed** or **Open** — meaning the customer never accepted them. "
                "Estimates containing 'This is an empty' in the summary are excluded "
                "(test records). The `creation_date` records when the estimate was created, "
                "and `Customer ID` links to the customer. Records with `Customer ID = 0` "
                "are filtered out as invalid."
            ),
            "how_converted": (
                "An unsold estimate is considered **converted** when the same customer — "
                "matched by `Customer ID` — books any job (status: Completed, Scheduled, "
                "In Progress, or Booked) within 12 months after the estimate creation date. "
                "This indicates the customer eventually moved forward with some service, "
                "even if it wasn't the exact estimate."
            ),
            "data_source": (
                "- **Event data**: `data/{company}/events/master_files/unsold_estimates_master.parquet`\n"
                "- **Conversion check**: `companies/{company}/parquet/Jobs.parquet`\n"
                "- **Key columns in event file**: `creation_date` (estimate date), "
                "`Customer ID`, `entity_id` (= Estimate ID), `estimate_status`, `Estimate Summary`\n"
                "- **Key columns in Jobs**: `Created Date`, `Customer ID`, `Status`"
            ),
            "example": (
                "Customer **103696502** received an estimate on **2025-12-01** with status "
                "Dismissed. Within 12 months, 1 booked job was found in Jobs.parquet for "
                "the same customer — so this unsold estimate counts as **converted**."
            ),
        },
    },
    "overdue_maintenance": {
        "title": "Overdue Maintenance",
        "icon": "🔧",
        "summary": (
            "Customers or locations that haven't had maintenance service in "
            "12+ months. The date shown is when their last maintenance was "
            "completed. Severity ranges from Medium to Critical (24+ months)."
        ),
        "formula": (
            "**Conversion Rate** = Customers who booked any job after their "
            "last maintenance date (within 12 months) / Total overdue cases"
        ),
        "master_file": "overdue_maintenance_master.parquet",
        "event_date_col": "completion_date",
        "customer_id_col": "Customer ID",
        "entity_id_col": "entity_id",
        "detected_label": "Overdue Flagged",
        "converted_label": "Serviced",
        "detail": {
            "how_detected": (
                "The system scans **Jobs.parquet** for maintenance-type jobs "
                "(identified by `Job Type` or `Job Class` containing 'Maintenance'). "
                "For each customer/location, it finds the most recent maintenance "
                "completion date. If that date is 12+ months ago, the customer is "
                "flagged as overdue. The `completion_date` in the event file represents "
                "when the customer last had maintenance — NOT when the event was detected. "
                "Severity is assigned based on months overdue: Medium (15-18), High (18-24), "
                "Critical (24+)."
            ),
            "how_converted": (
                "An overdue maintenance case is considered **converted** (serviced) when "
                "the same customer — matched by `Customer ID` — books any job (status: "
                "Completed, Scheduled, In Progress, or Booked) with a `Created Date` after "
                "their last maintenance `completion_date`, within a 12-month window. "
                "This means the customer came back for some type of service."
            ),
            "data_source": (
                "- **Event data**: `data/{company}/events/master_files/overdue_maintenance_master.parquet`\n"
                "- **Conversion check**: `companies/{company}/parquet/Jobs.parquet`\n"
                "- **Key columns in event file**: `completion_date` (last maintenance date), "
                "`Customer ID`, `months_overdue`, `Job Type`, `Location ID`\n"
                "- **Key columns in Jobs**: `Created Date`, `Customer ID`, `Status`"
            ),
            "example": (
                "Customer **10003020** last had maintenance on **2023-06-27** (32 months overdue). "
                "The system found 1 booked job created after that date — so this overdue case "
                "counts as **converted** (serviced)."
            ),
        },
    },
    "lost_customers": {
        "title": "Lost Customers",
        "icon": "👋",
        "summary": (
            "Customers who used to work with us but are now using competitors, "
            "detected through public building permit records. The date shown "
            "is when we last had contact with the customer."
        ),
        "formula": (
            "**Conversion Rate** = Lost customers who booked a job within "
            "12 months of last contact / Total lost customers"
        ),
        "master_file": "lost_customers_master.parquet",
        "event_date_col": "last_contact_date",
        "customer_id_col": "customer_id",
        "entity_id_col": "entity_id",
        "detected_label": "Lost Customers",
        "converted_label": "Returned",
        "detail": {
            "how_detected": (
                "The system cross-references customer addresses from **Customers.parquet** "
                "with public **Austin building permit records**. When an HVAC permit is "
                "filed at a customer's address by a different contractor, and that permit "
                "date is AFTER our last service date for that customer, the customer is "
                "classified as 'lost'. The `last_contact_date` is the date of our most "
                "recent interaction with the customer (job, call, or estimate). The "
                "`competitor_used` field shows which contractor did the work."
            ),
            "how_converted": (
                "A lost customer is considered **returned** when they book any job (status: "
                "Completed, Scheduled, In Progress, or Booked) with a `Created Date` after "
                "their `last_contact_date`, within a 12-month window. This means the "
                "customer came back to us despite having used a competitor."
            ),
            "data_source": (
                "- **Event data**: `data/{company}/events/master_files/lost_customers_master.parquet`\n"
                "- **Conversion check**: `companies/{company}/parquet/Jobs.parquet`\n"
                "- **Permit data**: `global_data/permits/permit_data.parquet`\n"
                "- **Key columns in event file**: `last_contact_date`, `customer_id`, "
                "`competitor_used`, `severity`, `address`, `months_since_last_contact`\n"
                "- **Key columns in Jobs**: `Created Date`, `Customer ID`, `Status`"
            ),
            "example": (
                "Customer **10054757** last contacted us on **2018-01-18**. A competitor "
                "filed an HVAC permit at their address after that date. No jobs were found "
                "in Jobs.parquet for this customer after their last contact — so this lost "
                "customer has **not converted** (not returned)."
            ),
        },
    },
    "second_chance_leads": {
        "title": "Second Chance Leads",
        "icon": "📞",
        "summary": (
            "Customers who called requesting service but didn't book. AI "
            "analysis of call transcripts identifies: (1) customer call, "
            "(2) service requested, (3) NOT booked. Invalid reasons are "
            "automatically filtered out."
        ),
        "formula": (
            "**Conversion Rate** = Leads who booked a job within 12 months "
            "of detection / Total leads detected"
        ),
        "master_file": None,
        "event_date_col": None,
        "customer_id_col": None,
        "entity_id_col": None,
        "detected_label": "Leads Detected",
        "converted_label": "Converted",
        "detail": {
            "how_detected": (
                "Phone call transcripts are downloaded and processed by an AI system "
                "(DeepSeek). For each customer's calls, the AI answers three questions:\n\n"
                "1. **Was this a customer call?** (not sales, spam, or verification)\n"
                "2. **Did they request service?** (HVAC repair, maintenance, or installation)\n"
                "3. **Was it booked?** (did the call result in a scheduled appointment)\n\n"
                "A lead is created ONLY when: customer call = YES, service requested = YES, "
                "booked = NO. The AI also checks for invalid reasons (reschedule, parts "
                "confirmation, missed tech call, invoice request, estimate approval, out of "
                "area, accidental call, membership signup, etc.) and filters those out."
            ),
            "how_converted": (
                "A second chance lead is considered **converted** when the same customer — "
                "matched by `Customer ID` or `Customer Phone` — books a job (status: "
                "Completed, Scheduled, In Progress, or Booked) within 12 months after the "
                "lead was detected. Phone matching is used as a fallback when Customer ID "
                "is unavailable, with phone numbers normalized (digits only) for comparison."
            ),
            "data_source": (
                "- **Lead data**: Google Sheets (loaded via `SECOND_CHANCE_SHEET_ID` env var)\n"
                "- **Conversion check**: `companies/{company}/parquet/Jobs.parquet`\n"
                "- **Key columns in Sheets**: `Analysis Timestamp` or `Call Date` (detection date), "
                "`Customer ID`, `Customer Phone`\n"
                "- **Key columns in Jobs**: `created_date` (normalized), `customer_id`, "
                "`customer_phone`, `status`"
            ),
            "example": (
                "A customer called on **2025-03-15** requesting AC repair. The AI confirmed "
                "it was a customer call with a service request, but the job was not booked. "
                "Within 12 months, the system checks if that customer (by ID or phone) booked "
                "any job in Jobs.parquet. If they did, it counts as **converted**."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_event_master(company: str, master_file: str) -> pd.DataFrame:
    for p in [
        Path("data") / company / "events" / "master_files" / master_file,
        Path("companies") / company / "parquet" / master_file,
    ]:
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                continue
    return pd.DataFrame()


@st.cache_data(ttl=120)
def _load_jobs(company: str) -> pd.DataFrame:
    path = Path("companies") / company / "parquet" / "Jobs.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df["_created"] = pd.to_datetime(df["Created Date"], errors="coerce")
        df["_cust_id"] = df["Customer ID"].astype(str).str.strip()
        df["_job_id"] = df["Job ID"].astype(str).str.strip()
        df["_status"] = df["Status"].astype(str).str.lower()
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

def _fmt_period(dt: datetime, pt: str) -> str:
    if pt == "week":
        return (dt - timedelta(days=dt.weekday())).strftime("%Y-%m-%d")
    if pt == "month":
        return dt.strftime("%Y-%m")
    if pt == "quarter":
        return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
    return str(dt.year)


def _period_bounds(ps: str, pt: str):
    if pt == "week":
        s = datetime.strptime(ps, "%Y-%m-%d")
        return s, s + timedelta(days=7)
    if pt == "month":
        s = datetime.strptime(ps + "-01", "%Y-%m-%d")
        nm = s.month % 12 + 1
        ny = s.year + (1 if s.month == 12 else 0)
        return s, datetime(ny, nm, 1)
    if pt == "quarter":
        y, q = ps.split("-Q")
        m = (int(q) - 1) * 3 + 1
        s = datetime(int(y), m, 1)
        em = m + 3
        ey = int(y) + (1 if em > 12 else 0)
        return s, datetime(ey, em if em <= 12 else em - 12, 1)
    s = datetime(int(ps), 1, 1)
    return s, datetime(s.year + 1, 1, 1)


# ---------------------------------------------------------------------------
# Per-event conversion logic
# ---------------------------------------------------------------------------

def _flag_conversions(
    events: pd.DataFrame,
    jobs: pd.DataFrame,
    event_key: str,
    period_type: str,
) -> pd.DataFrame:
    """For each event row, determine if the customer converted (booked a job)
    within 12 months of the event date.  Returns a DataFrame with one row per
    event containing: event_period, event_date, converted, conversion_period."""

    meta = EVENT_META[event_key]
    date_col = meta["event_date_col"]
    cid_col = meta["customer_id_col"]
    eid_col = meta["entity_id_col"]

    ev = events.copy()
    ev["_edate"] = pd.to_datetime(ev[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    ev["_cid"] = ev[cid_col].astype(str).str.strip()
    ev = ev.dropna(subset=["_edate"])
    ev = ev[ev["_cid"].notna() & ~ev["_cid"].isin(["", "nan", "None", "0"])]

    if ev.empty or jobs.empty:
        return pd.DataFrame(columns=["event_period", "event_date", "converted", "conversion_period"])

    booked = jobs[jobs["_status"].isin(BOOKED_STATUSES)].dropna(subset=["_created"])

    rows: list[dict] = []
    for _, r in ev.iterrows():
        cid = r["_cid"]
        edate = r["_edate"]
        window_end = edate + timedelta(days=365)

        cust_booked = booked[
            (booked["_cust_id"] == cid)
            & (booked["_created"] > edate)
            & (booked["_created"] <= window_end)
        ]

        # For cancellations, exclude the canceled job itself
        if event_key == "cancellations" and eid_col in ev.columns:
            canceled_job_id = str(r[eid_col]).strip()
            cust_booked = cust_booked[cust_booked["_job_id"] != canceled_job_id]

        converted = len(cust_booked) > 0
        conv_date = cust_booked["_created"].min() if converted else pd.NaT

        rows.append({
            "event_period": _fmt_period(edate, period_type),
            "event_date": edate,
            "converted": converted,
            "conversion_period": _fmt_period(conv_date, period_type) if converted else None,
        })

    return pd.DataFrame(rows)


def _build_period_table(flags: pd.DataFrame, period_type: str) -> pd.DataFrame:
    if flags.empty:
        return pd.DataFrame()

    all_periods = sorted(set(flags["event_period"].tolist()))
    rows: list[dict] = []
    for ps in all_periods:
        p_start, p_end = _period_bounds(ps, period_type)
        eligible_start = p_start - timedelta(days=365)

        detected = flags[flags["event_period"] == ps]
        conversions = flags[flags["conversion_period"] == ps]
        eligible = flags[
            (flags["event_date"] >= pd.Timestamp(eligible_start))
            & (flags["event_date"] < pd.Timestamp(p_end))
        ]

        n_det = len(detected)
        n_conv = len(conversions)
        pool = len(eligible)
        rate = (n_conv / pool * 100) if pool > 0 else 0.0

        rows.append({
            "period": ps,
            "total_detected": n_det,
            "conversions": n_conv,
            "eligible_pool": pool,
            "conversion_rate": round(rate, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared chart + table rendering
# ---------------------------------------------------------------------------

def _render_chart_and_table(
    perf: pd.DataFrame,
    period_type: str,
    event_key: str,
) -> None:
    meta = EVENT_META[event_key]
    det_label = meta["detected_label"]
    conv_label = meta["converted_label"]
    plabel = period_type.capitalize()

    total_det = int(perf["total_detected"].sum())
    total_conv = int(perf["conversions"].sum())
    rate = (total_conv / total_det * 100) if total_det > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total {det_label}", f"{total_det:,}")
    c2.metric(f"Total {conv_label}", f"{total_conv:,}")
    c3.metric("Overall Conversion Rate", f"{rate:.1f}%")

    st.subheader(f"{plabel}ly Breakdown")

    if ALTAIR_AVAILABLE and not perf.empty:
        melted = pd.melt(
            perf,
            id_vars=["period", "conversion_rate", "eligible_pool"],
            value_vars=["total_detected", "conversions"],
            var_name="metric_type", value_name="count",
        )
        melted["metric_label"] = melted["metric_type"].map({
            "total_detected": det_label, "conversions": conv_label,
        })
        bars = alt.Chart(melted).mark_bar(opacity=0.85).encode(
            x=alt.X("period:O", title=plabel, axis=alt.Axis(labelAngle=-45)),
            xOffset=alt.XOffset("metric_label:N"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color(
                "metric_label:N",
                scale=alt.Scale(domain=[det_label, conv_label], range=["#1f77b4", "#ff7f0e"]),
                legend=alt.Legend(title="Metric"),
            ),
            tooltip=[
                alt.Tooltip("period:O", title=plabel),
                alt.Tooltip("metric_label:N", title="Type"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("conversion_rate:Q", title="Conv. Rate (%)", format=".1f"),
            ],
        )
        line = alt.Chart(perf).mark_line(point=True, color="#2ca02c", strokeWidth=3).encode(
            x=alt.X("period:O"),
            y=alt.Y("conversion_rate:Q", title="Conversion Rate (%)"),
            tooltip=[
                alt.Tooltip("period:O", title=plabel),
                alt.Tooltip("conversions:Q", title=conv_label),
                alt.Tooltip("eligible_pool:Q", title="Eligible Pool"),
                alt.Tooltip("conversion_rate:Q", title="Conv. Rate (%)", format=".1f"),
            ],
        )
        chart = (
            alt.layer(bars, line)
            .resolve_scale(y="independent")
            .properties(title=f"{meta['title']} Detection & Conversion by {plabel}", height=500)
        )
        st.altair_chart(chart, width="stretch")

    st.markdown(
        f"<div style='text-align:center; color:#888; font-size:0.85em; margin-top:-10px; margin-bottom:16px'>"
        f"<span style='color:#1f77b4'>&#9632;</span> {det_label} &nbsp;&nbsp; "
        f"<span style='color:#ff7f0e'>&#9632;</span> {conv_label} &nbsp;&nbsp; "
        f"<span style='color:#2ca02c'>&#9644;&#9644;</span> Conversion Rate (right axis)"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        perf, width="stretch",
        column_config={
            "period": st.column_config.TextColumn(plabel),
            "total_detected": st.column_config.NumberColumn(det_label),
            "conversions": st.column_config.NumberColumn(conv_label),
            "eligible_pool": st.column_config.NumberColumn("Eligible Pool (12-mo)"),
            "conversion_rate": st.column_config.NumberColumn("Conversion Rate (%)", format="%.1f"),
        },
    )
    csv = perf.to_csv(index=False)
    st.download_button(
        f"Download {plabel}ly Data", csv,
        file_name=f"{event_key}_{period_type}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv", key=f"dl_{event_key}_{period_type}",
    )


# ---------------------------------------------------------------------------
# Collapsible detail section
# ---------------------------------------------------------------------------

def _render_detail_section(event_key: str) -> None:
    meta = EVENT_META[event_key]
    detail = meta.get("detail")
    if not detail:
        return

    with st.expander("How This Works — Detection, Conversion & Data Details", expanded=False):
        st.markdown("#### How the Event is Detected")
        st.markdown(detail["how_detected"])

        st.markdown("---")
        st.markdown("#### How Conversion is Determined")
        st.markdown(detail["how_converted"])

        st.markdown("---")
        st.markdown("#### Data Sources & Columns Referenced")
        st.markdown(detail["data_source"])

        st.markdown("---")
        st.markdown("#### Example")
        st.markdown(detail["example"])


# ---------------------------------------------------------------------------
# Standard event dashboard (cancellations, unsold, overdue, lost)
# ---------------------------------------------------------------------------

def _render_standard_dashboard(company: str, event_key: str) -> None:
    meta = EVENT_META[event_key]

    st.markdown(f"### {meta['icon']} {meta['title']}")
    st.markdown(meta["summary"])
    st.info(meta["formula"])
    _render_detail_section(event_key)

    events_df = _load_event_master(company, meta["master_file"])
    if events_df.empty:
        st.warning(f"No {meta['title'].lower()} data found. Run event detection first.")
        return

    jobs_df = _load_jobs(company)
    if jobs_df.empty:
        st.warning("Jobs data not found. Import Jobs.parquet first.")
        return

    date_col = meta["event_date_col"]
    events_df["_display_date"] = pd.to_datetime(events_df[date_col], errors="coerce", utc=True).dt.tz_localize(None)

    col1, col2, col3 = st.columns(3)
    with col1:
        period_type = st.selectbox(
            "View By", ["week", "month", "quarter", "year"],
            index=1, key=f"{event_key}_period",
        )
    min_dt = events_df["_display_date"].min()
    max_dt = events_df["_display_date"].max()
    with col2:
        min_date = st.date_input(
            "Min Date", value=min_dt.date() if pd.notna(min_dt) else date.today(),
            key=f"{event_key}_min",
        )
    with col3:
        max_date = st.date_input(
            "Max Date", value=max_dt.date() if pd.notna(max_dt) else date.today(),
            key=f"{event_key}_max",
        )

    filtered = events_df[
        (events_df["_display_date"] >= pd.Timestamp(min_date))
        & (events_df["_display_date"] <= pd.Timestamp(max_date) + pd.Timedelta(days=1))
    ]
    st.metric(f"Total {meta['detected_label']} in Range", f"{len(filtered):,}")

    with st.spinner("Calculating conversion rates..."):
        flags = _flag_conversions(filtered, jobs_df, event_key, period_type)
        perf = _build_period_table(flags, period_type)

    if perf.empty:
        st.info("No performance data for the selected filters.")
        return

    _render_chart_and_table(perf, period_type, event_key)


# ---------------------------------------------------------------------------
# Second Chance Leads (Google Sheets)
# ---------------------------------------------------------------------------

def _get_gspread_client() -> Any:
    if not GSHEETS_AVAILABLE:
        return None
    load_dotenv()
    creds_path = os.environ.get("GOOGLE_CREDS")
    if not creds_path:
        return None
    path_obj = Path(creds_path)
    if not path_obj.is_absolute():
        path_obj = ROOT / creds_path
    if not path_obj.exists():
        return None
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    return gspread.authorize(
        ServiceAccountCredentials.from_json_keyfile_name(str(path_obj), scope)
    )


@st.cache_data(ttl=300)
def _load_scl_from_sheets() -> Optional[Any]:
    if not GSHEETS_AVAILABLE:
        return None
    try:
        load_dotenv()
        sheet_id = (
            os.environ.get("SECOND_CHANCE_SHEET_ID")
            or os.environ.get("RECENT_SECOND_CHANCE_LEADS_SHEET_ID")
        )
        if not sheet_id:
            return None
        client = _get_gspread_client()
        if not client:
            return None
        spreadsheet = client.open_by_key(sheet_id)
        frames = []
        for name in ["Second Chance Leads", "Invalid Events"]:
            try:
                ws = spreadsheet.worksheet(name)
                vals = ws.get_all_values()
                if vals and len(vals) >= 2:
                    df = pd.DataFrame(vals[1:], columns=vals[0])
                    df["source_tab"] = name
                    frames.append(df)
            except Exception:
                continue
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        frame = pl.from_pandas(combined)
        if frame.is_empty():
            return frame
        entry_set = False
        for col in ["Analysis Timestamp", "Call Date", "Detected At", "Updated At"]:
            if col in frame.columns and not entry_set:
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
                entry_set = True
        if not entry_set:
            frame = frame.with_columns(pl.lit(None).cast(pl.Datetime).alias("entry_date"))
        frame = frame.filter(pl.col("entry_date").is_not_null())
        if "Customer ID" in frame.columns:
            frame = frame.with_columns(pl.col("Customer ID").cast(pl.Utf8, strict=False))
        if "Customer Phone" in frame.columns:
            frame = frame.with_columns(
                pl.col("Customer Phone").cast(pl.Utf8)
                .str.replace_all(r"[^0-9]", "").alias("normalized_phone")
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
        phone_col = next((c for c in ["customer_phone", "phone"] if c in frame.columns), None)
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


def _scl_find_conversion(lead: dict, jobs_df: Any, entry_date: datetime) -> Optional[datetime]:
    if jobs_df is None or jobs_df.is_empty():
        return None
    end = entry_date + timedelta(days=365)
    cid = lead.get("Customer ID")
    phone = lead.get("normalized_phone", "")
    base_filter = (pl.col("created_date") >= entry_date) & (pl.col("created_date") <= end)
    if cid and str(cid).strip() and "customer_id" in jobs_df.columns:
        base_filter = base_filter & (pl.col("customer_id") == str(cid))
    elif phone and str(phone).strip() and "normalized_phone" in jobs_df.columns:
        base_filter = base_filter & (pl.col("normalized_phone") == str(phone))
    else:
        return None
    booked = jobs_df.filter(
        base_filter & pl.col("status").str.to_lowercase().is_in(list(BOOKED_STATUSES))
    ).sort("created_date")
    if len(booked) > 0:
        return booked.select(pl.col("created_date").min()).item()
    return None


def _scl_build_perf(leads_df: Any, jobs_df: Any, period_type: str) -> pd.DataFrame:
    rows: list[dict] = []
    for row in leads_df.iter_rows(named=True):
        entry = row.get("entry_date")
        if entry is None:
            continue
        conv = _scl_find_conversion(row, jobs_df, entry)
        rows.append({
            "event_date": entry,
            "event_period": _fmt_period(entry, period_type),
            "converted": conv is not None,
            "conversion_period": _fmt_period(conv, period_type) if conv else None,
        })
    if not rows:
        return pd.DataFrame()
    flags = pd.DataFrame(rows)
    flags["event_date"] = pd.to_datetime(flags["event_date"], errors="coerce")
    return _build_period_table(flags, period_type)


def _render_second_chance_dashboard(company: str) -> None:
    meta = EVENT_META["second_chance_leads"]
    st.markdown(f"### {meta['icon']} {meta['title']}")
    st.markdown(meta["summary"])
    st.info(meta["formula"])
    _render_detail_section("second_chance_leads")

    if not GSHEETS_AVAILABLE:
        st.warning(
            "Google Sheets integration requires: polars, gspread, python-dotenv, "
            "oauth2client. Set GOOGLE_CREDS + SECOND_CHANCE_SHEET_ID env vars."
        )
        return

    with st.spinner("Loading data from Google Sheets..."):
        leads_df = _load_scl_from_sheets()
        jobs_df = _load_jobs_polars(company)

    if leads_df is None or leads_df.is_empty():
        st.warning("No second chance leads found. Check env vars GOOGLE_CREDS and SECOND_CHANCE_SHEET_ID.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        period_type = st.selectbox("View By", ["week", "month", "quarter", "year"], index=1, key="scl_period")
    mn = leads_df.select(pl.col("entry_date").min()).item()
    mx = leads_df.select(pl.col("entry_date").max()).item()
    with col2:
        min_entry = st.date_input(
            "Min Date",
            value=mn.date() if mn and isinstance(mn, datetime) else (mn if mn else date.today()),
            key="scl_min",
        )
    with col3:
        max_entry = st.date_input(
            "Max Date",
            value=mx.date() if mx and isinstance(mx, datetime) else (mx if mx else date.today()),
            key="scl_max",
        )

    filtered = leads_df
    if "entry_date" in filtered.columns:
        filtered = filtered.filter(
            (pl.col("entry_date") >= datetime.combine(min_entry, datetime.min.time()))
            & (pl.col("entry_date") <= datetime.combine(max_entry, datetime.max.time()))
        )
    st.metric("Total Leads in Range", len(filtered))

    with st.spinner("Calculating conversion rates..."):
        perf = _scl_build_perf(filtered, jobs_df, period_type)

    if perf.empty:
        st.info("No performance data for the selected filters.")
        return

    _render_chart_and_table(perf, period_type, "second_chance_leads")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_all_dashboards(company: str) -> None:
    keys = list(EVENT_META.keys())
    labels = [f"{EVENT_META[k]['icon']} {EVENT_META[k]['title']}" for k in keys]
    tabs = st.tabs(labels)
    for tab, key in zip(tabs, keys):
        with tab:
            if key == "second_chance_leads":
                _render_second_chance_dashboard(company)
            else:
                _render_standard_dashboard(company, key)
