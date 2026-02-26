from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datahound.observability.ai_usage_v2 import (
    load_usage_records,
    summarize_usage,
    load_provider_attributed_usage,
    build_daily_reconciliation,
)

st.set_page_config(page_title="AI Ops Usage v2", layout="wide")

st.title("AI Ops Usage v2")
st.caption("Aggregates usage across swarm tasks, OpenClaw sessions, and CI workflows.")

col_a, col_b = st.columns(2)
with col_a:
    days_back = st.slider("Days back", min_value=1, max_value=90, value=30)
with col_b:
    st.markdown("**Execution paths**: swarm, direct-agent, CI")

records = load_usage_records(days_back=days_back)

if not records:
    st.info("No usage records found for the selected window. Ensure log paths are configured.")
    st.stop()

summary = summarize_usage(records)

data = pd.DataFrame([r.to_dict() for r in records])

metric_cols = st.columns(4)
metric_cols[0].metric("Total Tokens", f"{summary['total_tokens']:,}")
metric_cols[1].metric("Total Cost (USD)", f"${summary['total_cost_usd']:.4f}")
metric_cols[2].metric("Exact Records", summary["counts"].get("exact", 0))
metric_cols[3].metric("Estimated Records", summary["counts"].get("estimated", 0))

st.subheader("Breakdowns")

breakdown_tabs = st.tabs(["Provider", "Model", "Agent", "Execution Path", "Source"])

for tab, key, label in [
    (breakdown_tabs[0], "by_provider", "Provider"),
    (breakdown_tabs[1], "by_model", "Model"),
    (breakdown_tabs[2], "by_agent", "Agent"),
    (breakdown_tabs[3], "by_execution_path", "Execution Path"),
    (breakdown_tabs[4], "by_source", "Source"),
]:
    with tab:
        rows = [
            {label: k, "tokens": v["tokens"], "cost_usd": round(v["cost_usd"], 4), "count": v["count"]}
            for k, v in summary[key].items()
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("tokens", ascending=False), use_container_width=True)
        else:
            st.caption("No data available.")

st.subheader("Exact vs Estimated")
status_counts = data.groupby(["usage_type"]).size().reset_index(name="records")
status_tokens = (
    data.groupby("usage_type")["tokens_total"].sum().reset_index(name="tokens_total")
)
status_table = status_counts.merge(status_tokens, on="usage_type", how="left")
st.dataframe(status_table, use_container_width=True)

st.subheader("Daily Reconciliation")
provider_usage = load_provider_attributed_usage(days_back=days_back)
if provider_usage:
    reconciliation = build_daily_reconciliation(records, provider_usage)
    st.dataframe(pd.DataFrame(reconciliation), use_container_width=True)
else:
    st.caption("No provider-attributed usage file found (logging/ai_ops/provider_usage.jsonl).")

st.subheader("Raw Records")
with st.expander("Show records"):
    st.dataframe(data.sort_values("timestamp", ascending=False), use_container_width=True)
