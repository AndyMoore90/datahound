import json
import sys
from dataclasses import dataclass
from pathlib import Path as _P
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import UTC, datetime
import time
import html
import re
import streamlit as st
import pandas as pd

ROOT = _P(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from datahound.events.llm_utils import LLMConfig, LLMAnalyzer, build_aging_systems_system_prompt
from apps._shared import select_company_config
from apps.components.scheduler_ui import (
    render_schedule_config, render_task_manager, 
    create_scheduled_task, render_scheduler_status
)
from datahound.scheduler import TaskType


def _extract_answer_value(answer_obj):
  """Extract the answer value from LLM response answer object"""
  if not answer_obj or not isinstance(answer_obj, dict):
    return None
  return answer_obj.get("answer")


def _parse_llm_response(llm_result):
  """Parse LLM response to extract structured data, handling various formats"""
  if not llm_result or not isinstance(llm_result, dict):
    return None, "Invalid input"
  
  # Case 1: Direct structured response with 'answers' key
  if "answers" in llm_result:
    return llm_result, "Direct structured response"
  
  # Case 2: Response wrapped in 'raw_response' 
  if "raw_response" in llm_result:
    raw_content = llm_result["raw_response"]
    
    # Check for truncation indicators
    is_truncated = (
      raw_content.endswith('"source":') or
      raw_content.endswith('{"source":') or
      raw_content.endswith('"evidence_snippets": [') or
      not raw_content.strip().endswith('}') or
      ('```' in raw_content and raw_content.count('```') == 1)  # Single backtick = incomplete
    )
    
    # Strategy 1: Try to extract JSON content after ```json marker
    import re
    import json
    
    json_start_marker = raw_content.find('```json')
    if json_start_marker != -1:
      # Extract everything after ```json
      json_content = raw_content[json_start_marker + 7:].strip()  # Skip '```json\n'
      
      # Remove trailing ``` if present
      if json_content.endswith('```'):
        json_content = json_content[:-3].strip()
      
      # Try to parse the JSON content
      try:
        parsed_json = json.loads(json_content)
        if isinstance(parsed_json, dict):
          status = "Parsed from markdown" + (" (complete)" if not is_truncated else " (truncated)")
          return parsed_json, status
      except json.JSONDecodeError:
        # If truncated, try to repair by finding the last complete object
        if is_truncated:
          # Find the last complete JSON object by counting braces
          brace_count = 0
          last_valid_pos = -1
          
          for i, char in enumerate(json_content):
            if char == '{':
              brace_count += 1
            elif char == '}':
              brace_count -= 1
              if brace_count == 0:
                last_valid_pos = i
          
          if last_valid_pos > 0:
            try:
              truncated_json = json_content[:last_valid_pos + 1]
              parsed_json = json.loads(truncated_json)
              if isinstance(parsed_json, dict):
                return parsed_json, "Parsed truncated JSON (recovered)"
            except json.JSONDecodeError:
              pass
    
    return None, f"Failed to parse - {'truncated response' if is_truncated else 'malformed JSON'}"
  
  # Case 3: Return original if no parsing worked
  return llm_result, "Returned original"


EVENT_TYPES: List[str] = [
  "overdue_maintenance",
  "aging_systems",
  "canceled_jobs",
  "unsold_estimates",
  "lost_customers",
]


def _render_event_type_ui(et: str, current: Dict[str, Dict[str, Any]], available_files: List[str], columns_cache: Dict[str, List[str]], events_master_default_dir: _P) -> None:
  # Common small controls
  if et == "overdue_maintenance":
    current[et]["months_threshold"] = int(st.number_input("Months threshold", min_value=1, max_value=120, value=int(current[et].get("months_threshold", 12)), key=f"{et}_months"))
  # Per-type UI
  if et == "unsold_estimates":
    st.markdown("**Source & Columns**")
    current[et]["file"] = st.selectbox(
      "Master file", options=available_files or [current[et].get("file", "Estimates.parquet")],
      index=(available_files.index(current[et]["file"]) if current[et].get("file") in available_files else 0),
      key="ue_file2"
    )
    cols = columns_cache.get(current[et]["file"], [])
    current[et]["status_column"] = st.selectbox("Estimate status column", options=cols or [current[et].get("status_column", "Estimate Status")], index=((cols.index(current[et].get("status_column", "Estimate Status")) if current[et].get("status_column", "Estimate Status") in cols else 0) if cols else 0), key="ue_status_col2")
    include_vals = current[et].get("status_include_values") or ["Dismissed", "Open"]
    current[et]["status_include_values"] = st.multiselect("Statuses to include", options=sorted(list({*include_vals, *cols})), default=include_vals, key="ue_status_include2")
    current[et]["creation_date_column"] = st.selectbox("Creation date column", options=cols or [current[et].get("creation_date_column", "Creation Date")], index=((cols.index(current[et].get("creation_date_column", "Creation Date")) if current[et].get("creation_date_column", "Creation Date") in cols else 0) if cols else 0), key="ue_creation_col2")
    current[et]["opportunity_status_column"] = st.selectbox("Opportunity status column", options=cols or [current[et].get("opportunity_status_column", "Opportunity Status")], index=((cols.index(current[et].get("opportunity_status_column", "Opportunity Status")) if current[et].get("opportunity_status_column", "Opportunity Status") in cols else 0) if cols else 0), key="ue_opp_col2")
    current[et]["opportunity_exclude_value"] = st.text_input("Exclude opportunity status", value=current[et].get("opportunity_exclude_value", "Won"), key="ue_opp_exclude2")
    current[et]["id_column"] = st.selectbox("ID column", options=cols or [current[et].get("id_column", "Estimate ID")], index=((cols.index(current[et].get("id_column", "Estimate ID")) if current[et].get("id_column", "Estimate ID") in cols else 0) if cols else 0), key="ue_id_col2")
    st.markdown("**Detection Logic (read-only)**")
    st.code(
      f"""
FILTER1 = {current[et]['status_column']} in {current[et]['status_include_values']}
FILTER2 = {current[et]['creation_date_column']} within last {int(current[et].get('months_back', 24))} months
FILTER3 = {current[et]['opportunity_status_column']} != '{current[et]['opportunity_exclude_value']}'
DEDUP = group by Customer ID + Location ID, keep most recent Creation Date
""".strip(), language="text")
    st.markdown("**Payload Columns**")
    pre_ue = current[et].get("payload_columns") or []
    current[et]["payload_columns"] = st.multiselect("Columns to include in payload", options=cols, default=[c for c in pre_ue if c in cols], key="ue_payload2")
    st.markdown("**Output Order**")
    base_order_ue = ["event_type", "entity_type", "entity_id", "detected_at", "estimate_status", "creation_date", "opportunity_status", "location_id"]
    all_out_cols_ue = base_order_ue + [c for c in current[et]["payload_columns"] if c not in base_order_ue]
    saved_order_ue = [c for c in (current[et].get("output_order") or []) if c in all_out_cols_ue]
    for c in all_out_cols_ue:
      if c not in saved_order_ue:
        saved_order_ue.append(c)
    current[et]["output_order"] = st.multiselect("Arrange output columns order", options=all_out_cols_ue, default=saved_order_ue, key="ue_output_order2")
    st.markdown("**Output File**")
    current[et]["output_filename"] = st.text_input("Output file name", value=current[et].get("output_filename", "unsold_estimates_master.parquet"), key="ue_out_name2")
  elif et == "overdue_maintenance":
    st.markdown("**Source & Columns**")
    current[et]["file"] = st.selectbox("Master file", options=available_files or [current[et].get("file", "Jobs.parquet")], index=(available_files.index(current[et]["file"]) if current[et].get("file") in available_files else 0), key="om_file2")
    cols = columns_cache.get(current[et]["file"], [])
    current[et]["filter_column"] = st.selectbox("Filter column", options=cols or [current[et].get("filter_column", "Job Type")], index=((cols.index(current[et].get("filter_column", "Job Type")) if current[et].get("filter_column", "Job Type") in cols else 0) if cols else 0), key="om_filter_col3")
    match_values = current[et].get("filter_match_values") or []
    match_values_text = "\n".join(match_values) if isinstance(match_values, list) else str(match_values) if match_values else ""
    match_values_input = st.text_area(
      "Filter: match values (one per line, case-insensitive exact match)",
      value=match_values_text,
      key="om_filter_match_values3",
      help="Only rows where the filter column exactly matches (case-insensitive) one of these values will be considered."
    )
    current[et]["filter_match_values"] = [v.strip() for v in match_values_input.strip().split("\n") if v.strip()]
    current[et]["status_column"] = st.selectbox("Status column", options=cols or [current[et].get("status_column", "Status")], index=((cols.index(current[et].get("status_column", "Status")) if current[et].get("status_column", "Status") in cols else 0) if cols else 0), key="om_status_col3", help="Column containing job status.")
    current[et]["status_value"] = st.text_input("Status value (exact match)", value=current[et].get("status_value", "Completed"), key="om_status_value3", help="Only rows with this exact status value will be considered.")
    current[et]["date_column"] = st.selectbox("Date column", options=cols or [current[et].get("date_column", "Completion Date")], index=((cols.index(current[et].get("date_column", "Completion Date")) if current[et].get("date_column", "Completion Date") in cols else 0) if cols else 0), key="om_date_col3")
    current[et]["id_column"] = st.selectbox("ID column", options=cols or [current[et].get("id_column", "Job ID")], index=((cols.index(current[et].get("id_column", "Job ID")) if current[et].get("id_column", "Job ID") in cols else 0) if cols else 0), key="om_id_col3")
    st.markdown("**Output File**")
    current[et]["output_filename"] = st.text_input("Output file name", value=current[et].get("output_filename", "overdue_maintenance_master.parquet"), key="om_out_name2")
  elif et == "canceled_jobs":
    st.markdown("**Source & Columns**")
    current[et]["file"] = st.selectbox("Master file", options=available_files or [current[et].get("file", "Jobs.parquet")], index=(available_files.index(current[et]["file"]) if current[et].get("file") in available_files else 0), key="cj_file2")
    cols = columns_cache.get(current[et]["file"], [])
    current[et]["status_column"] = st.selectbox("Status column", options=cols or [current[et].get("status_column", "Status")], index=((cols.index(current[et].get("status_column", "Status")) if current[et].get("status_column", "Status") in cols else 0) if cols else 0), key="cj_status_col2")
    current[et]["status_canceled_value"] = st.text_input("Canceled status value", value=current[et].get("status_canceled_value", "Canceled"), key="cj_status_val2")
    current[et]["date_column"] = st.selectbox("Completion date column", options=cols or [current[et].get("date_column", "Completion Date")], index=((cols.index(current[et].get("date_column", "Completion Date")) if current[et].get("date_column", "Completion Date") in cols else 0) if cols else 0), key="cj_date_col2")
    current[et]["id_column"] = st.selectbox("ID column", options=cols or [current[et].get("id_column", "Job ID")], index=((cols.index(current[et].get("id_column", "Job ID")) if current[et].get("id_column", "Job ID") in cols else 0) if cols else 0), key="cj_id_col2")
    st.markdown("**Output File**")
    current[et]["output_filename"] = st.text_input("Output file name", value=current[et].get("output_filename", "canceled_jobs_master.parquet"), key="cj_out_name2")
  elif et == "aging_systems":
    st.markdown("**Source & Columns**")
    current[et]["locations_file"] = st.selectbox("Locations master file", options=available_files or [current[et].get("locations_file", "Locations.parquet")], index=(available_files.index(current[et]["locations_file"]) if current[et].get("locations_file") in available_files else 0), key="as_locations_file2")
    current[et]["jobs_file"] = st.selectbox("Jobs master file", options=available_files or [current[et].get("jobs_file", "Jobs.parquet")], index=(available_files.index(current[et]["jobs_file"]) if current[et].get("jobs_file") in available_files else 0), key="as_jobs_file2")
    loc_cols = columns_cache.get(current[et]["locations_file"], [])
    job_cols = columns_cache.get(current[et]["jobs_file"], [])
    current[et]["location_id_column"] = st.selectbox("Location ID column", options=loc_cols or [current[et].get("location_id_column", "Location ID")], index=((loc_cols.index(current[et].get("location_id_column", "Location ID")) if current[et].get("location_id_column", "Location ID") in loc_cols else 0) if loc_cols else 0), key="as_loc_id_col2")
    default_payload = current[et].get("job_payload_fields") or ["Summary", "Job Type", "Status", "Created Date", "Completion Date", "Customer ID"]
    current[et]["job_payload_fields"] = st.multiselect("Job payload fields", options=job_cols, default=[c for c in default_payload if c in job_cols], key="as_job_payload2")
    st.markdown("**Field Mapping**")
    mapping = current[et].get("payload_field_mapping") or {}
    new_mapping = {}
    for f in current[et]["job_payload_fields"]:
      new_mapping[f] = st.text_input(f"{f} →", value=str(mapping.get(f, f)), key=f"as_map2_{f}")
    current[et]["payload_field_mapping"] = new_mapping
    st.markdown("**Output File**")
    current[et]["output_filename"] = st.text_input("Output file name", value=current[et].get("output_filename", "location_jobs_history.parquet"), key="as_out_name2")
    st.markdown("**Analysis Mode**")
    current[et]["analysis_mode"] = st.radio("Select analysis mode", ["basic", "llm"], index=(0 if (current[et].get("analysis_mode") or "basic") == "basic" else 1), key="as_mode2")
    current[et]["analysis_output_filename"] = st.text_input("Analysis output file name", value=current[et].get("analysis_output_filename", "aging_systems_basic.parquet"), key="as_analysis_out2")

  if et == "lost_customers":
    st.markdown("**Source Files & Columns**")
    current[et]["customers_file"] = st.selectbox("Customers file", options=available_files or [current[et].get("customers_file", "Customers.parquet")], index=(available_files.index(current[et]["customers_file"]) if current[et].get("customers_file") in available_files else 0), key="lc_customers_file2")
    current[et]["calls_file"] = st.selectbox("Calls file", options=available_files or [current[et].get("calls_file", "Calls.parquet")], index=(available_files.index(current[et]["calls_file"]) if current[et].get("calls_file") in available_files else 0), key="lc_calls_file2")
    
    customers_cols = columns_cache.get(current[et]["customers_file"], [])
    calls_cols = columns_cache.get(current[et]["calls_file"], [])
    
    current[et]["customer_id_column"] = st.selectbox("Customer ID column", options=customers_cols or [current[et].get("customer_id_column", "Customer ID")], index=((customers_cols.index(current[et].get("customer_id_column", "Customer ID")) if current[et].get("customer_id_column", "Customer ID") in customers_cols else 0) if customers_cols else 0), key="lc_customer_id2")
    current[et]["customer_address_column"] = st.selectbox("Customer address column", options=customers_cols or [current[et].get("customer_address_column", "Full Address")], index=((customers_cols.index(current[et].get("customer_address_column", "Full Address")) if current[et].get("customer_address_column", "Full Address") in customers_cols else 0) if customers_cols else 0), key="lc_customer_address2")
    current[et]["call_date_column"] = st.selectbox("Call date column", options=calls_cols or [current[et].get("call_date_column", "Call Date")], index=((calls_cols.index(current[et].get("call_date_column", "Call Date")) if current[et].get("call_date_column", "Call Date") in calls_cols else 0) if calls_cols else 0), key="lc_call_date2")
    
    st.markdown("**Company Configuration**")
    company_names = current[et].get("company_names", ["McCullough Heating & Air", "McCullough Heating and Air"])
    current[et]["company_names"] = st.text_area("Company names (one per line)", value="\n".join(company_names), key="lc_company_names2").strip().split("\n")
    
    st.markdown("**Analysis Settings**")
    current[et]["require_permit_data"] = st.checkbox("Require permit data for analysis", value=bool(current[et].get("require_permit_data", True)), key="lc_require_permits2")
    current[et]["min_contact_history"] = int(st.number_input("Minimum contact history (calls)", min_value=1, max_value=50, value=int(current[et].get("min_contact_history", 2)), key="lc_min_contacts2"))
    current[et]["output_filename"] = st.text_input("Output file name", value=current[et].get("output_filename", "lost_customers_analysis.parquet"), key="lc_out_name2")

  # Final common control
  current[et]["processing_limit"] = int(st.number_input("Max records (0 = no limit)", min_value=0, max_value=1_000_000, value=int(current[et].get("processing_limit", 0)), key=f"{et}_limit2"))


def _render_event_type_section(
  et: str,
  current: Dict[str, Dict[str, Any]],
  available_files: List[str],
  columns_cache: Dict[str, List[str]],
  events_master_default_dir: _P,
) -> None:
  # Top-right controls
  col_top_l, col_top_r = st.columns([3,1])
  with col_top_r:
    if et == "overdue_maintenance":
      current[et]["dedup_enabled"] = st.checkbox(
        "Dedup",
        value=bool(current[et].get("dedup_enabled", False)),
        key="om_dedup_enable_top",
        help="Remove duplicates by Customer+Location, keeping the most recent completion."
      )
    if et == "canceled_jobs":
      current[et]["dedup_enabled"] = st.checkbox(
        "Dedup",
        value=bool(current[et].get("dedup_enabled", False)),
        key="cj_dedup_enable_top",
        help="Remove duplicates by Customer+Location, keeping the most recent completion."
      )
    if et == "unsold_estimates":
      current[et]["dedup_enabled"] = st.checkbox(
        "Dedup",
        value=bool(current[et].get("dedup_enabled", False)),
        key="ue_dedup_enable_top",
        help="Remove duplicates by Customer+Location, keeping the most recent creation date."
      )

  if et == "overdue_maintenance":
    current[et]["months_threshold"] = int(st.number_input("Months threshold", min_value=1, max_value=120, value=int(current[et]["months_threshold"]), key=f"{et}_months"))

  if et == "canceled_jobs":
    current[et]["months_back"] = int(st.number_input(
      "Months back window",
      min_value=1,
      max_value=120,
      value=int(current[et]["months_back"]),
      key="cj_months_back",
      help="Only consider canceled jobs whose completion date is within this many months."
    ))

  if et == "unsold_estimates":
    st.markdown("**Source & Columns**")
    current[et]["file"] = st.selectbox(
      "Master file", options=available_files or [current[et].get("file", "Estimates.parquet")],
      index=(available_files.index(current[et]["file"]) if current[et]["file"] in available_files else 0),
      key="ue_file"
    )
    cols = columns_cache.get(current[et]["file"], [])
    current[et]["status_column"] = st.selectbox("Estimate status column", options=cols or [current[et]["status_column"]], index=(cols.index(current[et]["status_column"]) if current[et]["status_column"] in cols else 0), key="ue_status_col")
    include_vals = current[et].get("status_include_values") or ["Dismissed", "Open"]
    current[et]["status_include_values"] = st.multiselect("Statuses to include", options=sorted(list({*include_vals, *cols})), default=include_vals, key="ue_status_include", help="Values treated as unsold candidates.")
    current[et]["creation_date_column"] = st.selectbox("Creation date column", options=cols or [current[et]["creation_date_column"]], index=(cols.index(current[et]["creation_date_column"]) if current[et]["creation_date_column"] in cols else 0), key="ue_creation_col")
    current[et]["opportunity_status_column"] = st.selectbox("Opportunity status column", options=cols or [current[et]["opportunity_status_column"]], index=(cols.index(current[et]["opportunity_status_column"]) if current[et]["opportunity_status_column"] in cols else 0), key="ue_opp_col")
    current[et]["opportunity_exclude_value"] = st.text_input("Exclude opportunity status", value=current[et]["opportunity_exclude_value"], key="ue_opp_exclude")
    current[et]["id_column"] = st.selectbox("ID column", options=cols or [current[et]["id_column"]], index=(cols.index(current[et]["id_column"]) if current[et]["id_column"] in cols else 0), key="ue_id_col")
    st.markdown("**Detection Logic (read-only)**")
    st.code(
      f"""
FILTER1 = {current[et]['status_column']} in {current[et]['status_include_values']}
FILTER2 = {current[et]['creation_date_column']} within last {int(current[et]['months_back'])} months
FILTER3 = {current[et]['opportunity_status_column']} != '{current[et]['opportunity_exclude_value']}'
DEDUP = group by Customer ID + Location ID, keep most recent Creation Date
INVALIDATE A = {current[et]['summary_column']} contains '{current[et]['summary_invalidate_contains']}'
INVALIDATE B = same {current[et]['location_column']} has {current[et]['estimate_completion_date_column']} within {int(current[et]['recent_estimate_days'])} days
INVALIDATE C = in Jobs.parquet, same {current[et]['customer_column']} has Created or Scheduled within ±{int(current[et]['recent_job_days'])} days of estimate
""".strip(), language="text")
    st.markdown("**Payload Columns**")
    pre_ue = current[et].get("payload_columns") or []
    current[et]["payload_columns"] = st.multiselect("Columns to include in payload", options=cols, default=[c for c in pre_ue if c in cols], key="ue_payload")
    st.markdown("**Output Order**")
    base_order_ue = [
      "event_type", "entity_type", "entity_id", "detected_at",
      "estimate_status", "creation_date", "opportunity_status", "location_id"
    ]
    all_out_cols_ue = base_order_ue + [c for c in current[et]["payload_columns"] if c not in base_order_ue]
    saved_order_ue = [c for c in (current[et].get("output_order") or []) if c in all_out_cols_ue]
    for c in all_out_cols_ue:
      if c not in saved_order_ue:
        saved_order_ue.append(c)
    current[et]["output_order"] = st.multiselect("Arrange output columns order", options=all_out_cols_ue, default=saved_order_ue, key="ue_output_order")
    st.markdown("**Output File**")
    default_name_ue = current[et].get("output_filename") or "unsold_estimates_master.parquet"
    current[et]["output_filename"] = st.text_input("Output file name", value=default_name_ue, key="ue_out_name")
    base_check_ue = _P(current.get(et, {}).get("output_dir") or events_master_default_dir)
    desired_path_ue = base_check_ue / current[et]["output_filename"]
    if desired_path_ue.exists():
      st.warning(f"Output file '{current[et]['output_filename']}' already exists.")
      allowed_actions_ue = ["Overwrite", "Change name"]
      curr_act_ue = current[et].get("conflict_action")
      if curr_act_ue not in allowed_actions_ue:
        curr_act_ue = "Overwrite"
      current[et]["conflict_action"] = st.radio("File exists - choose action", allowed_actions_ue, index=allowed_actions_ue.index(curr_act_ue), key="ue_conflict")
      if current[et]["conflict_action"] == "Overwrite":
        current[et]["backup_before_overwrite"] = st.checkbox("Create backup before overwrite", value=bool(current[et].get("backup_before_overwrite", True)), key="ue_backup")
      else:
        ts_default_ue = desired_path_ue.with_stem(desired_path_ue.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name
        current[et]["new_filename"] = st.text_input("New file name", value=current[et].get("new_filename") or ts_default_ue, key="ue_new_name")
    st.markdown("**Invalidation & Recent Activity**")
    current[et]["summary_column"] = st.selectbox("Summary column", options=cols or [current[et]["summary_column"]], index=(cols.index(current[et]["summary_column"]) if current[et]["summary_column"] in cols else 0), key="ue_sum_col", help="Rows whose summary contains the substring are invalid.")
    current[et]["summary_invalidate_contains"] = st.text_input("Invalidate if summary contains", value=current[et]["summary_invalidate_contains"], key="ue_sum_contains")
    current[et]["location_column"] = st.selectbox("Location ID column", options=cols or [current[et]["location_column"]], index=(cols.index(current[et]["location_column"]) if current[et]["location_column"] in cols else 0), key="ue_loc_col")
    current[et]["estimate_completion_date_column"] = st.selectbox("Estimate completion date column", options=cols or [current[et]["estimate_completion_date_column"]], index=(cols.index(current[et]["estimate_completion_date_column"]) if current[et]["estimate_completion_date_column"] in cols else 0), key="ue_est_comp_col")
    current[et]["recent_estimate_days"] = int(st.number_input("Recent estimate window (days)", min_value=1, max_value=180, value=int(current[et]["recent_estimate_days"]), key="ue_recent_est"))
    current[et]["customer_column"] = st.selectbox("Customer ID column", options=cols or [current[et]["customer_column"]], index=(cols.index(current[et]["customer_column"]) if current[et]["customer_column"] in cols else 0), key="ue_cust_col")
    jobs_cols = columns_cache.get("Jobs.parquet", [])
    current[et]["jobs_file"] = st.selectbox("Jobs file", options=["Jobs.parquet"], index=0, key="ue_jobs_file")
    current[et]["job_created_date_column"] = st.selectbox("Job created date column", options=jobs_cols or [current[et]["job_created_date_column"]], index=((jobs_cols.index(current[et]["job_created_date_column"]) if current[et]["job_created_date_column"] in jobs_cols else 0) if jobs_cols else 0), key="ue_job_created")
    current[et]["job_scheduled_date_column"] = st.selectbox("Job scheduled date column", options=jobs_cols or [current[et]["job_scheduled_date_column"]], index=((jobs_cols.index(current[et]["job_scheduled_date_column"]) if current[et]["job_scheduled_date_column"] in jobs_cols else 0) if jobs_cols else 0), key="ue_job_sched")
    current[et]["recent_job_days"] = int(st.number_input("Recent job window (±days)", min_value=1, max_value=60, value=int(current[et]["recent_job_days"]), key="ue_recent_job", help="Jobs within this many days before/after the estimate invalidate the event."))
    st.markdown("**Duplicate Handling**")
    if current[et]["dedup_enabled"]:
      current[et]["dedup_customer_column"] = st.selectbox("Customer ID column", options=cols or [current[et]["dedup_customer_column"]], index=(cols.index(current[et]["dedup_customer_column"]) if current[et]["dedup_customer_column"] in cols else 0), key="ue_dedup_customer")
      current[et]["dedup_location_column"] = st.selectbox("Location ID column", options=cols or [current[et]["dedup_location_column"]], index=(cols.index(current[et]["dedup_location_column"]) if current[et]["dedup_location_column"] in cols else 0), key="ue_dedup_location")

  if et == "overdue_maintenance":
    st.markdown("**Source & Columns**")
    current[et]["file"] = st.selectbox(
      "Master file", options=available_files or [current[et].get("file", "Jobs.parquet")],
      index=(available_files.index(current[et]["file"]) if current[et]["file"] in available_files else 0),
      key="om_file"
    )
    cols = columns_cache.get(current[et]["file"], [])
    current[et]["filter_column"] = st.selectbox("Filter column", options=cols or [current[et].get("filter_column", "Job Type")], index=(cols.index(current[et].get("filter_column", "Job Type")) if current[et].get("filter_column", "Job Type") in cols else 0), key="om_filter_col2", help="Column to match against the list of values below.")
    match_values = current[et].get("filter_match_values") or []
    match_values_text = "\n".join(match_values) if isinstance(match_values, list) else str(match_values) if match_values else ""
    match_values_input = st.text_area(
      "Filter: match values (one per line, case-insensitive exact match)",
      value=match_values_text,
      key="om_filter_match_values2",
      help="Only rows where the filter column exactly matches (case-insensitive) one of these values will be considered."
    )
    current[et]["filter_match_values"] = [v.strip() for v in match_values_input.strip().split("\n") if v.strip()]
    current[et]["status_column"] = st.selectbox("Status column", options=cols or [current[et].get("status_column", "Status")], index=(cols.index(current[et].get("status_column", "Status")) if current[et].get("status_column", "Status") in cols else 0), key="om_status_col2", help="Column containing job status.")
    current[et]["status_value"] = st.text_input("Status value (exact match)", value=current[et].get("status_value", "Completed"), key="om_status_value2", help="Only rows with this exact status value will be considered.")
    current[et]["date_column"] = st.selectbox("Date column", options=cols or [current[et]["date_column"]], index=(cols.index(current[et]["date_column"]) if current[et]["date_column"] in cols else 0), key="om_date_col2", help="Date used to compute months since completion.")
    current[et]["id_column"] = st.selectbox("ID column", options=cols or [current[et]["id_column"]], index=(cols.index(current[et]["id_column"]) if current[et]["id_column"] in cols else 0), key="om_id_col2", help="Unique identifier for the entity written to the event file.")
    st.markdown("**Detection Logic (read-only)**")
    match_count = len(current[et].get("filter_match_values", []))
    st.code(
      f"""
FILTER1 = {current[et]['filter_column']} matches one of {match_count} values (case-insensitive exact match)
FILTER2 = {current[et]['status_column']} == '{current[et].get('status_value', 'Completed')}' (exact match)
DATE = parse {current[et]['date_column']} as date
MONTHS_SINCE = months_between(today, DATE)
EVENT_TRUE = FILTER1 AND FILTER2 AND MONTHS_SINCE >= {int(current[et]['months_threshold'])}
""".strip(), language="text")
    st.markdown("**Payload Columns**")
    preselect = current[et].get("payload_columns") or []
    current[et]["payload_columns"] = st.multiselect(
      "Columns to include in event payload",
      options=cols,
      default=[c for c in preselect if c in cols],
      key="om_payload_cols",
      help="Additional columns to include from the source rows when EVENT_TRUE."
    )
    st.markdown("**Output Order**")
    base_order = [
      "event_type", "entity_type", "entity_id", "detected_at",
      "months_overdue", "job_class", "completion_date"
    ]
    all_out_cols = base_order + [c for c in current[et]["payload_columns"] if c not in base_order]
    saved_order = [c for c in (current[et].get("output_order") or []) if c in all_out_cols]
    for c in all_out_cols:
      if c not in saved_order:
        saved_order.append(c)
    current[et]["output_order"] = st.multiselect(
      "Arrange output columns order",
      options=all_out_cols,
      default=saved_order,
      key="om_output_order",
      help="The order of columns in the saved Parquet file."
    )
    st.markdown("**Output File**")
    default_name = current[et].get("output_filename") or "overdue_maintenance_master.parquet"
    current[et]["output_filename"] = st.text_input(
      "Output file name",
      value=default_name,
      key="om_out_name_ui",
      help="The Parquet file that will be written in the company's parquet folder"
    )
    base_check_om = _P(current.get(et, {}).get("output_dir") or events_master_default_dir)
    desired_path = base_check_om / current[et]["output_filename"]
    if desired_path.exists():
      st.warning(f"Output file '{current[et]['output_filename']}' already exists.")
      allowed_actions = ["Overwrite", "Change name"]
      current_action = current[et].get("conflict_action")
      if current_action not in allowed_actions:
        current_action = "Overwrite"
      current[et]["conflict_action"] = st.radio(
        "File exists - choose action",
        allowed_actions,
        index=allowed_actions.index(current_action),
        key="om_conflict_action_ui"
      )
      if current[et]["conflict_action"] == "Overwrite":
        current[et]["backup_before_overwrite"] = st.checkbox(
          "Create backup before overwrite", value=bool(current[et].get("backup_before_overwrite", True)), key="om_backup_ui"
        )
      elif current[et]["conflict_action"] == "Change name":
        ts_default = desired_path.with_stem(desired_path.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name
        current[et]["new_filename"] = st.text_input(
          "New file name", value=current[et].get("new_filename") or ts_default, key="om_new_name_ui",
          help="Provide a new name to avoid overwriting the existing file"
        )
    st.markdown("**Invalidation Rules**")
    current[et]["exclude_column"] = st.selectbox(
      "Column to check",
      options=[""] + cols,
      index=(1 + cols.index(current[et].get("exclude_column")) if current[et].get("exclude_column") in cols else 0),
      key="om_exclude_col",
      help="Rows will be excluded if they match the rules below on this column."
    )
    st.markdown("**Duplicate Handling**")
    if current[et]["dedup_enabled"]:
      current[et]["dedup_customer_column"] = st.selectbox(
        "Customer ID column",
        options=cols or [current[et]["dedup_customer_column"]],
        index=(cols.index(current[et]["dedup_customer_column"]) if current[et]["dedup_customer_column"] in cols else 0),
        key="om_dedup_customer_col"
      )
      current[et]["dedup_location_column"] = st.selectbox(
        "Location ID column",
        options=cols or [current[et]["dedup_location_column"]],
        index=(cols.index(current[et]["dedup_location_column"]) if current[et]["dedup_location_column"] in cols else 0),
        key="om_dedup_location_col"
      )
    current[et]["exclude_equals"] = st.text_input(
      "Exclude if equals (exact match)",
      value=current[et].get("exclude_equals", ""),
      key="om_ex_equals",
      help="Exact, case-sensitive match. Leave blank to skip."
    )
    current[et]["exclude_contains"] = st.text_input(
      "Exclude if contains (substring, case-insensitive)",
      value=current[et].get("exclude_contains", ""),
      key="om_ex_contains",
      help="If provided, any row containing this substring will be excluded."
    )

  if et == "canceled_jobs":
    st.markdown("**Source & Columns**")
    current[et]["file"] = st.selectbox(
      "Master file", options=available_files or [current[et].get("file", "Jobs.parquet")],
      index=(available_files.index(current[et]["file"]) if current[et]["file"] in available_files else 0),
      key="cj_file"
    )
    cols = columns_cache.get(current[et]["file"], [])
    current[et]["status_column"] = st.selectbox("Status column", options=cols or [current[et]["status_column"]], index=(cols.index(current[et]["status_column"]) if current[et]["status_column"] in cols else 0), key="cj_status_col")
    current[et]["status_canceled_value"] = st.text_input("Canceled status value", value=current[et]["status_canceled_value"], key="cj_status_val")
    current[et]["date_column"] = st.selectbox("Completion date column", options=cols or [current[et]["date_column"]], index=(cols.index(current[et]["date_column"]) if current[et]["date_column"] in cols else 0), key="cj_date_col")
    current[et]["id_column"] = st.selectbox("ID column", options=cols or [current[et]["id_column"]], index=(cols.index(current[et]["id_column"]) if current[et]["id_column"] in cols else 0), key="cj_id_col")
    st.markdown("**Detection Logic (read-only)**")
    st.code(
      f"""
FILTER1 = {current[et]['status_column']} == '{current[et]['status_canceled_value']}'
FILTER2 = {current[et]['date_column']} within last {int(current[et]['months_back'])} months
INVALIDATION = For each row, find other rows with same {current[et]['location_column']}. If any has {current[et]['invalidation_date_column']} within last {int(current[et]['invalidation_days_recent'])} days AND {current[et]['status_column']} != '{current[et]['status_canceled_value']}', then exclude.
""".strip(), language="text")
    st.markdown("**Payload Columns**")
    preselect_cj = current[et].get("payload_columns") or []
    current[et]["payload_columns"] = st.multiselect(
      "Columns to include in event payload",
      options=cols,
      default=[c for c in preselect_cj if c in cols],
      key="cj_payload_cols",
      help="Additional columns to include from the source rows when detected."
    )
    st.markdown("**Output Order**")
    base_order_cj = [
      "event_type", "entity_type", "entity_id", "detected_at",
      "status", "completion_date", "location_id", "cancellation_age_months"
    ]
    all_out_cols_cj = base_order_cj + [c for c in current[et]["payload_columns"] if c not in base_order_cj]
    saved_order_cj = [c for c in (current[et].get("output_order") or []) if c in all_out_cols_cj]
    for c in all_out_cols_cj:
      if c not in saved_order_cj:
        saved_order_cj.append(c)
    current[et]["output_order"] = st.multiselect(
      "Arrange output columns order",
      options=all_out_cols_cj,
      default=saved_order_cj,
      key="cj_output_order",
      help="The order of columns in the saved Parquet file."
    )
    st.markdown("**Output File**")
    default_name_cj = current[et].get("output_filename") or "canceled_jobs_master.parquet"
    current[et]["output_filename"] = st.text_input(
      "Output file name",
      value=default_name_cj,
      key="cj_out_name_ui",
      help="The Parquet file name to write."
    )
    base_check_cj = events_master_default_dir
    desired_path_cj = base_check_cj / current[et]["output_filename"]
    if desired_path_cj.exists():
      st.warning(f"Output file '{current[et]['output_filename']}' already exists.")
      allowed_actions_cj = ["Overwrite", "Change name"]
      current_action_cj = current[et].get("conflict_action")
      if current_action_cj not in allowed_actions_cj:
        current_action_cj = "Overwrite"
      current[et]["conflict_action"] = st.radio(
        "File exists - choose action",
        allowed_actions_cj,
        index=allowed_actions_cj.index(current_action_cj),
        key="cj_conflict_action_ui"
      )
      if current[et]["conflict_action"] == "Overwrite":
        current[et]["backup_before_overwrite"] = st.checkbox(
          "Create backup before overwrite", value=bool(current[et].get("backup_before_overwrite", True)), key="cj_backup_ui"
        )
      elif current[et]["conflict_action"] == "Change name":
        ts_default_cj = desired_path_cj.with_stem(desired_path_cj.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name
        current[et]["new_filename"] = st.text_input(
          "New file name", value=current[et].get("new_filename") or ts_default_cj, key="cj_new_name_ui",
          help="Provide a new name to avoid overwriting the existing file"
        )
    st.markdown("**Invalidation Logic**")
    current[et]["location_column"] = st.selectbox(
      "Location ID column",
      options=cols or [current[et]["location_column"]],
      index=(cols.index(current[et]["location_column"]) if current[et]["location_column"] in cols else 0),
      key="cj_loc_col",
      help="Rows with the same Location ID are considered the same site; recent non-canceled work at the same site invalidates a cancel."
    )
    current[et]["invalidation_date_column"] = st.selectbox(
      "Invalidation date column",
      options=cols or [current[et]["invalidation_date_column"]],
      index=(cols.index(current[et]["invalidation_date_column"]) if current[et]["invalidation_date_column"] in cols else 0),
      key="cj_inv_date_col",
      help="Date used to detect recent non-canceled work (e.g., Completed/Completion Date)."
    )
    current[et]["invalidation_days_recent"] = int(st.number_input(
      "Invalidation window (days)",
      min_value=1,
      max_value=365,
      value=int(current[et]["invalidation_days_recent"]),
      key="cj_inv_days",
      help="If any non-canceled job at the same Location ID has this date within the last N days, mark the cancellation as invalid."
    ))
    st.markdown("**Additional Invalidation**")
    current[et]["summary_column"] = st.selectbox(
      "Summary column",
      options=[""] + cols,
      index=(([""] + cols).index(current[et].get("summary_column", "")) if current[et].get("summary_column", "") in ([""] + cols) else 0),
      key="cj_summary_col",
      help="If provided, rows whose summary contains 'test' (case-insensitive) will be excluded."
    )
    current[et]["summary_invalidate_contains"] = st.text_input(
      "Invalidate if summary contains",
      value=current[et].get("summary_invalidate_contains", "test"),
      key="cj_summary_contains",
      help="Substring to search for in the summary to invalidate rows."
    )
    st.markdown("**Duplicate Handling**")
    if current[et]["dedup_enabled"]:
      current[et]["dedup_customer_column"] = st.selectbox(
        "Customer ID column",
        options=cols or [current[et]["dedup_customer_column"]],
        index=(cols.index(current[et]["dedup_customer_column"]) if current[et]["dedup_customer_column"] in cols else 0),
        key="cj_dedup_customer_col"
      )
      current[et]["dedup_location_column"] = st.selectbox(
        "Location ID column",
        options=cols or [current[et]["dedup_location_column"]],
        index=(cols.index(current[et]["dedup_location_column"]) if current[et]["dedup_location_column"] in cols else 0),
        key="cj_dedup_location_col"
      )

  if et == "aging_systems":
    st.markdown("**Source & Columns**")
    current[et]["locations_file"] = st.selectbox(
      "Locations master file",
      options=available_files or [current[et].get("locations_file", "Locations.parquet")],
      index=(available_files.index(current[et]["locations_file"]) if current[et]["locations_file"] in available_files else 0),
      key="as_locations_file"
    )
    current[et]["jobs_file"] = st.selectbox(
      "Jobs master file",
      options=available_files or [current[et].get("jobs_file", "Jobs.parquet")],
      index=(available_files.index(current[et]["jobs_file"]) if current[et]["jobs_file"] in available_files else 0),
      key="as_jobs_file"
    )
    loc_cols = columns_cache.get(current[et]["locations_file"], [])
    job_cols = columns_cache.get(current[et]["jobs_file"], [])
    current[et]["location_id_column"] = st.selectbox(
      "Location ID column",
      options=loc_cols or [current[et]["location_id_column"]],
      index=(loc_cols.index(current[et]["location_id_column"]) if current[et]["location_id_column"] in loc_cols else 0),
      key="as_loc_id_col"
    )
    default_payload = current[et].get("job_payload_fields") or [
      "Summary", "Job Type", "Status", "Created Date", "Completion Date", "Customer ID"
    ]
    current[et]["job_payload_fields"] = st.multiselect(
      "Job payload fields",
      options=job_cols,
      default=[c for c in default_payload if c in job_cols],
      key="as_job_payload",
      help="Fields to include per job in the JSON payload."
    )
    st.markdown("**Field Mapping**")
    mapping = current[et].get("payload_field_mapping") or {}
    new_mapping = {}
    for f in current[et]["job_payload_fields"]:
      new_mapping[f] = st.text_input(f"{f} →", value=str(mapping.get(f, f)), key=f"as_map_{f}")
    current[et]["payload_field_mapping"] = new_mapping
    st.markdown("**Detection Logic (read-only)**")
    st.code(
      f"""
COLLECT LOCATIONS = all {current[et]['location_id_column']} from {current[et]['locations_file']}
MATCH JOBS = for each Location ID, find rows in {current[et]['jobs_file']}
PAYLOAD = for each matched job, include {current[et]['job_payload_fields']} as JSON
OUTPUT = one row per Location ID with an array of job payloads
""".strip(), language="text")
    st.markdown("**Output File**")
    current[et]["output_filename"] = st.text_input(
      "Output file name",
      value=current[et].get("output_filename") or "location_jobs_history.parquet",
      key="as_out_name"
    )
    base_check_as = _P(current.get(et, {}).get("output_dir") or events_master_default_dir)
    desired_path_as = base_check_as / current[et]["output_filename"]
    if desired_path_as.exists():
      st.warning(f"Output file '{current[et]['output_filename']}' already exists.")
      allowed_actions_as = ["Overwrite", "Change name"]
      curr_act_as = current[et].get("conflict_action")
      if curr_act_as not in allowed_actions_as:
        curr_act_as = "Overwrite"
      current[et]["conflict_action"] = st.radio("File exists - choose action", allowed_actions_as, index=allowed_actions_as.index(curr_act_as), key="as_conflict")
      if current[et]["conflict_action"] == "Overwrite":
        current[et]["backup_before_overwrite"] = st.checkbox("Create backup before overwrite", value=bool(current[et].get("backup_before_overwrite", True)), key="as_backup")
      else:
        ts_default_as = desired_path_as.with_stem(desired_path_as.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name
        current[et]["new_filename"] = st.text_input("New file name", value=current[et].get("new_filename") or ts_default_as, key="as_new_name")
    st.markdown("**Analysis Mode**")
    current[et]["analysis_mode"] = st.radio(
      "Select analysis mode",
      ["basic", "llm"],
      index=(0 if (current[et].get("analysis_mode") or "basic") == "basic" else 1),
      key="as_mode",
      help="Choose between a simple rules-based analysis or an LLM-driven analysis (placeholder)."
    )
    current[et]["analysis_output_filename"] = st.text_input(
      "Analysis output file name",
      value=current[et].get("analysis_output_filename") or "aging_systems_basic.parquet",
      key="as_analysis_out"
    )

  if et == "unsold_estimates":
    current[et]["months_back"] = int(st.number_input(
      "Months back window",
      min_value=1,
      max_value=120,
      value=int(current[et]["months_back"]),
      key="ue_months_back",
      help="Only consider estimates created within this many months."
    ))

  if et == "lost_customers":
    st.markdown("**Source Files & Columns**")
    current[et]["customers_file"] = st.selectbox(
      "Customers file", 
      options=available_files or [current[et].get("customers_file", "Customers.parquet")], 
      index=(available_files.index(current[et]["customers_file"]) if current[et].get("customers_file") in available_files else 0), 
      key="lc_customers_file"
    )
    current[et]["calls_file"] = st.selectbox(
      "Calls file", 
      options=available_files or [current[et].get("calls_file", "Calls.parquet")], 
      index=(available_files.index(current[et]["calls_file"]) if current[et].get("calls_file") in available_files else 0), 
      key="lc_calls_file"
    )
    
    customers_cols = columns_cache.get(current[et]["customers_file"], [])
    calls_cols = columns_cache.get(current[et]["calls_file"], [])
    
    current[et]["customer_id_column"] = st.selectbox(
      "Customer ID column", 
      options=customers_cols or [current[et].get("customer_id_column", "Customer ID")], 
      index=((customers_cols.index(current[et].get("customer_id_column", "Customer ID")) if current[et].get("customer_id_column", "Customer ID") in customers_cols else 0) if customers_cols else 0), 
      key="lc_customer_id"
    )
    current[et]["customer_address_column"] = st.selectbox(
      "Customer address column", 
      options=customers_cols or [current[et].get("customer_address_column", "Full Address")], 
      index=((customers_cols.index(current[et].get("customer_address_column", "Full Address")) if current[et].get("customer_address_column", "Full Address") in customers_cols else 0) if customers_cols else 0), 
      key="lc_customer_address"
    )
    current[et]["call_date_column"] = st.selectbox(
      "Call date column", 
      options=calls_cols or [current[et].get("call_date_column", "Call Date")], 
      index=((calls_cols.index(current[et].get("call_date_column", "Call Date")) if current[et].get("call_date_column", "Call Date") in calls_cols else 0) if calls_cols else 0), 
      key="lc_call_date"
    )
    
    st.markdown("**Company Configuration**")
    company_names = current[et].get("company_names", ["McCullough Heating & Air", "McCullough Heating and Air"])
    current[et]["company_names"] = st.text_area(
      "Company names (one per line)", 
      value="\n".join(company_names), 
      key="lc_company_names"
    ).strip().split("\n")
    
    st.markdown("**Analysis Settings**")
    current[et]["require_permit_data"] = st.checkbox(
      "Require permit data for analysis", 
      value=bool(current[et].get("require_permit_data", True)), 
      key="lc_require_permits"
    )
    current[et]["min_contact_history"] = int(st.number_input(
      "Minimum contact history (calls)", 
      min_value=1, 
      max_value=50, 
      value=int(current[et].get("min_contact_history", 2)), 
      key="lc_min_contacts"
    ))
    
    st.markdown("**Output File**")
    current[et]["output_filename"] = st.text_input(
      "Output file name", 
      value=current[et].get("output_filename", "lost_customers_analysis.parquet"), 
      key="lc_out_name"
    )

  current[et]["processing_limit"] = int(st.number_input("Max records (0 = no limit)", min_value=0, max_value=1_000_000, value=int(current[et]["processing_limit"]), key=f"{et}_limit"))
def _config_path(company: str) -> _P:
  cfg_dir = _P("config") / "events"
  cfg_dir.mkdir(parents=True, exist_ok=True)
  return cfg_dir / f"{company}_historical_events_config.json"


def _load_config(company: str) -> Dict[str, Any]:
  p = _config_path(company)
  if p.exists():
    try:
      return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
      return {}
  return {}


def _save_config(company: str, cfg: Dict[str, Any]) -> None:
  p = _config_path(company)
  p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _event_defaults() -> Dict[str, Dict[str, Any]]:
  return {
    "overdue_maintenance": {
      "enabled": False,
      "months_threshold": 12,
      "processing_limit": 0,
      "include_enrichment": False,
      "file": "Jobs.parquet",
      "filter_column": "Job Type",
      "filter_match_values": [
        "PSA Heat Inspection - 2 systems",
        "PSA Heat Inspection - 1 System",
        "1 Time Heat",
        "Complimentary 1 Time Inspection",
        "PSA Heat & Cool Inspection - 2 systems",
        "$79 Promotional Tune Up",
        "PSA Heat Inspection - 4 systems",
        "1 Time Heat & Cool",
        "PSA Heat & Cool Inspection - 1 system",
        "PSA Heat Inspection - 5 systems",
        "PSA Heat Inspection - 7 systems",
        "PSA Cool Inspection - 1 system",
        "1 Time Cool",
        "PSA Heat & Cool Inspection - 4 systems",
        "PSA Cool Inspection - 2 systems",
        "PSA Heat & Cool Inspection - 3 systems",
        "$129 Promotional Tune Up",
        "PSA Cool Inspection - 3 systems",
        "PSA Cool Inspection - 5 systems",
        "PSA Cool Inspection - 4 systems",
        "PSA Cool Inspection - 6 systems",
        "PSA Heat & Cool Inspection - 5 systems",
        "PSA Cool Inspection - 7 systems",
        "PSA Heat Inspection - 6 systems",
        "Multi-Family Tune Up",
        "$39 Furnace Tune Up",
        "$69 Heat Tune Up",
        "$79 AC Tune Up",
        "TXGas Furnace Tune up",
        "$19.95 Fall Heating Safety Check",
        "$79 Furnace Tune-up",
        "$39 Outdoor Unit Tune Up",
        "$99 Outdoor Unit Tune Up",
        "$39 Pre-Season Inspection",
        "EEF Furnace Tune Up",
        "Heating Inspection",
        "Cooling Inspection",
        "1x Tune Up",
        "AE Check Up"
      ],
      "status_column": "Status",
      "status_value": "Completed",
      "date_column": "Completion Date",
      "id_column": "Job ID",
      "payload_columns": [],
      "output_order": [],
      "exclude_column": "",
      "exclude_equals": "",
      "exclude_contains": "",
      "dedup_enabled": False,
      "dedup_customer_column": "Customer ID",
      "dedup_location_column": "Location ID",
    },
    "aging_systems": {
      "enabled": False,
      "min_age_years": 10,
      "processing_limit": 0,
      "include_enrichment": False,
      "locations_file": "Locations.parquet",
      "jobs_file": "Jobs.parquet",
      "location_id_column": "Location ID",
      "job_payload_fields": [
        "Summary", "Job Type", "Status", "Created Date", "Completion Date", "Customer ID"
      ],
      "payload_field_mapping": {
        "Summary": "Job Notes",
        "Job Type": "Job Type",
        "Status": "Job Status",
        "Created Date": "Job Created Date",
        "Completion Date": "Job Completion Date",
        "Customer ID": "Customer ID"
      },
      "output_filename": "location_jobs_history.parquet",
      "analysis_mode": "basic",
      "analysis_output_filename": "aging_systems_basic.parquet",
      "llm_analysis_output_filename": "aging_systems.parquet",
      "additional_payload_columns": [],
      "llm_output_filename": "aging_systems_llm.json",
      "llm_max_concurrent_requests": 25,
      "llm_max_tokens": 8192,
      "llm_temperature": 0.0,
      "llm_model": "deepseek-chat",
      "permit_history_output_filename": "location_permit_history.parquet",
      "profile_output_filename": "location_profile.parquet",
      "permit_data_path": "global_data/permits/permit_data.parquet",
      "permit_max_edit_distance": 2,
      "permit_fields": [
        "Permit Type Desc", "Permit Num", "Permit Class Mapped", "Permit Class", "Work Class", "Description",
        "Applied Date", "Issued Date", "Status Current", "Status Date", "Expires Date", "Completion Date",
        "Housing Units", "Original Address 1", "Original City", "Original State", "Original Zip", "Link",
        "Master Permit Num", "Location", "Contractor Trade", "Contractor Company Name", "Contractor Full Name"
      ],
      "permit_field_mapping": {
        "Permit Type Desc": "Permit Type Desc",
        "Permit Num": "Permit Num",
        "Permit Class Mapped": "Permit Class Mapped",
        "Permit Class": "Permit Class",
        "Work Class": "Work Class",
        "Description": "Description",
        "Applied Date": "Applied Date",
        "Issued Date": "Issued Date",
        "Status Current": "Status Current",
        "Status Date": "Status Date",
        "Expires Date": "Expires Date",
        "Completion Date": "Completion Date",
        "Housing Units": "Housing Units",
        "Original Address 1": "Original Address 1",
        "Original City": "Original City",
        "Original State": "Original State",
        "Original Zip": "Original Zip",
        "Link": "Link",
        "Master Permit Num": "Master Permit Num",
        "Location": "Location",
        "Contractor Trade": "Contractor Trade",
        "Contractor Company Name": "Contractor Company Name",
        "Contractor Full Name": "Contractor Full Name"
      }
    },
    "canceled_jobs": {
      "enabled": False,
      "file": "Jobs.parquet",
      "status_column": "Status",
      "status_canceled_value": "Canceled",
      "date_column": "Completion Date",
      "id_column": "Job ID",
      "months_back": 24,
      "location_column": "Location ID",
      "invalidation_date_column": "Completion Date",
      "invalidation_days_recent": 30,
      "payload_columns": [],
      "output_order": [],
      "processing_limit": 0,
      "dedup_enabled": False,
      "dedup_customer_column": "Customer ID",
      "dedup_location_column": "Location ID",
    },
    "unsold_estimates": {
      "enabled": False,
      "file": "Estimates.parquet",
      "status_column": "Estimate Status",
      "status_include_values": ["Dismissed", "Open"],
      "creation_date_column": "Creation Date",
      "months_back": 24,
      "opportunity_status_column": "Opportunity Status",
      "opportunity_exclude_value": "Won",
      "summary_column": "Estimate Summary",
      "summary_invalidate_contains": "Test Estimate",
      "location_column": "Location ID",
      "customer_column": "Customer ID",
      "estimate_completion_date_column": "Completion Date",
      "recent_estimate_days": 30,
      "jobs_file": "Jobs.parquet",
      "job_created_date_column": "Created Date",
      "job_scheduled_date_column": "Scheduled Date",
      "recent_job_days": 21,
      "id_column": "Estimate ID",
      "payload_columns": [],
      "output_order": [],
      "processing_limit": 0,
      "dedup_enabled": False,
      "dedup_customer_column": "Customer ID",
      "dedup_location_column": "Location ID",
    },
    "lost_customers": {
      "enabled": False,
      "processing_limit": 0,
      "customers_file": "Customers.parquet",
      "calls_file": "Calls.parquet",
      "customer_id_column": "Customer ID",
      "customer_address_column": "Full Address",
      "call_date_column": "Call Date",
      "company_names": ["McCullough Heating & Air", "McCullough Heating and Air"],
      "require_permit_data": True,
      "min_contact_history": 2,
      "exclude_nan_competitor": True,
      "output_filename": "lost_customers_analysis.parquet"
    },
  }


def render(company: str, cfg) -> None:
  current: Dict[str, Dict[str, Any]] = _event_defaults()
  loaded = _load_config(company) or {}
  for k, v in (loaded.get("events") or {}).items():
    if k in current and isinstance(v, dict):
      current[k].update(v)

  st.info("Configure historical scans that run against full master parquet files.")

  parquet_dir = ROOT / "companies" / company / "parquet"
  events_master_default_dir = _P(cfg.data_dir).parent / "events" / "master_files"
  available_files = [p.name for p in sorted(parquet_dir.glob("*.parquet"))]
  import pyarrow.parquet as pq
  columns_cache: Dict[str, List[str]] = {}
  for fname in available_files:
    try:
      schema = pq.read_schema(parquet_dir / fname)
      columns_cache[fname] = schema.names
    except Exception:
      columns_cache[fname] = []

  tab_main, tab_llm, tab_scheduler = st.tabs(["Event Types", "LLM Settings", "📊 Scheduler Status"])

  with tab_main:
    st.subheader("Event Types")
    for et in EVENT_TYPES:
      with st.expander(et.replace("_", " ").title(), expanded=bool(current[et]["enabled"])):
        current[et]["enabled"] = st.checkbox("Enabled", value=bool(current[et]["enabled"]), key=f"{et}_enabled")
        
        # Top-right controls
        col_top_l, col_top_r = st.columns([3,1])
        with col_top_r:
          if et == "overdue_maintenance":
            current[et]["dedup_enabled"] = st.checkbox(
              "Dedup",
              value=bool(current[et].get("dedup_enabled", False)),
              key="om_dedup_enable_top",
              help="Remove duplicates by Customer+Location, keeping the most recent completion."
            )
          if et == "canceled_jobs":
            current[et]["dedup_enabled"] = st.checkbox(
              "Dedup",
              value=bool(current[et].get("dedup_enabled", False)),
              key="cj_dedup_enable_top",
              help="Remove duplicates by Customer+Location, keeping the most recent completion."
            )
          if et == "unsold_estimates":
            current[et]["dedup_enabled"] = st.checkbox(
              "Dedup",
              value=bool(current[et].get("dedup_enabled", False)),
              key="ue_dedup_enable_top",
              help="Remove duplicates by Customer+Location, keeping the most recent creation date."
            )
        
        if et == "overdue_maintenance":
          current[et]["months_threshold"] = int(st.number_input("Months threshold", min_value=1, max_value=120, value=int(current[et]["months_threshold"]), key=f"{et}_months"))
        
        if et == "canceled_jobs":
          current[et]["months_back"] = int(st.number_input(
            "Months back window",
            min_value=1,
            max_value=120,
            value=int(current[et]["months_back"]),
            key="cj_months_back",
            help="Only consider canceled jobs whose completion date is within this many months."
          ))
        
        if et == "unsold_estimates":
          st.markdown("**Source & Columns**")
          current[et]["file"] = st.selectbox(
            "Master file", options=available_files or [current[et].get("file", "Estimates.parquet")],
            index=(available_files.index(current[et]["file"]) if current[et]["file"] in available_files else 0),
            key="ue_file"
          )
          cols = columns_cache.get(current[et]["file"], [])
          current[et]["status_column"] = st.selectbox("Estimate status column", options=cols or [current[et]["status_column"]], index=(cols.index(current[et]["status_column"]) if current[et]["status_column"] in cols else 0), key="ue_status_col")
          include_vals = current[et].get("status_include_values") or ["Dismissed", "Open"]
          current[et]["status_include_values"] = st.multiselect("Statuses to include", options=sorted(list({*include_vals, *cols})), default=include_vals, key="ue_status_include", help="Values treated as unsold candidates.")
          current[et]["creation_date_column"] = st.selectbox("Creation date column", options=cols or [current[et]["creation_date_column"]], index=(cols.index(current[et]["creation_date_column"]) if current[et]["creation_date_column"] in cols else 0), key="ue_creation_col")
          current[et]["opportunity_status_column"] = st.selectbox("Opportunity status column", options=cols or [current[et]["opportunity_status_column"]], index=(cols.index(current[et]["opportunity_status_column"]) if current[et]["opportunity_status_column"] in cols else 0), key="ue_opp_col")
          current[et]["opportunity_exclude_value"] = st.text_input("Exclude opportunity status", value=current[et]["opportunity_exclude_value"], key="ue_opp_exclude")
          current[et]["id_column"] = st.selectbox("ID column", options=cols or [current[et]["id_column"]], index=(cols.index(current[et]["id_column"]) if current[et]["id_column"] in cols else 0), key="ue_id_col")
          st.markdown("**Detection Logic (read-only)**")
          st.code(
            f"""
FILTER1 = {current[et]['status_column']} in {current[et]['status_include_values']}
FILTER2 = {current[et]['creation_date_column']} within last {int(current[et]['months_back'])} months
FILTER3 = {current[et]['opportunity_status_column']} != '{current[et]['opportunity_exclude_value']}'
DEDUP = group by Customer ID + Location ID, keep most recent Creation Date
INVALIDATE A = {current[et]['summary_column']} contains '{current[et]['summary_invalidate_contains']}'
INVALIDATE B = same {current[et]['location_column']} has {current[et]['estimate_completion_date_column']} within {int(current[et]['recent_estimate_days'])} days
INVALIDATE C = in Jobs.parquet, same {current[et]['customer_column']} has Created or Scheduled within ±{int(current[et]['recent_job_days'])} days of estimate
""".strip(), language="text")
          st.markdown("**Payload Columns**")
          pre_ue = current[et].get("payload_columns") or []
          current[et]["payload_columns"] = st.multiselect("Columns to include in payload", options=cols, default=[c for c in pre_ue if c in cols], key="ue_payload")
          st.markdown("**Output Order**")
          base_order_ue = [
            "event_type", "entity_type", "entity_id", "detected_at",
            "estimate_status", "creation_date", "opportunity_status", "location_id"
          ]
          all_out_cols_ue = base_order_ue + [c for c in current[et]["payload_columns"] if c not in base_order_ue]
          saved_order_ue = [c for c in (current[et].get("output_order") or []) if c in all_out_cols_ue]
          for c in all_out_cols_ue:
            if c not in saved_order_ue:
              saved_order_ue.append(c)
          current[et]["output_order"] = st.multiselect("Arrange output columns order", options=all_out_cols_ue, default=saved_order_ue, key="ue_output_order")
          st.markdown("**Output File**")
          default_name_ue = current[et].get("output_filename") or "unsold_estimates_master.parquet"
          current[et]["output_filename"] = st.text_input("Output file name", value=default_name_ue, key="ue_out_name")
          base_check_ue = _P(current.get(et, {}).get("output_dir") or events_master_default_dir)
          desired_path_ue = base_check_ue / current[et]["output_filename"]
          if desired_path_ue.exists():
            st.warning(f"Output file '{current[et]['output_filename']}' already exists.")
            allowed_actions_ue = ["Overwrite", "Change name"]
            curr_act_ue = current[et].get("conflict_action")
            if curr_act_ue not in allowed_actions_ue:
              curr_act_ue = "Overwrite"
            current[et]["conflict_action"] = st.radio("File exists - choose action", allowed_actions_ue, index=allowed_actions_ue.index(curr_act_ue), key="ue_conflict")
            if current[et]["conflict_action"] == "Overwrite":
              current[et]["backup_before_overwrite"] = st.checkbox("Create backup before overwrite", value=bool(current[et].get("backup_before_overwrite", True)), key="ue_backup")
            else:
              ts_default_ue = desired_path_ue.with_stem(desired_path_ue.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name
              current[et]["new_filename"] = st.text_input("New file name", value=current[et].get("new_filename") or ts_default_ue, key="ue_new_name")
          st.markdown("**Invalidation & Recent Activity**")
          current[et]["summary_column"] = st.selectbox("Summary column", options=cols or [current[et]["summary_column"]], index=(cols.index(current[et]["summary_column"]) if current[et]["summary_column"] in cols else 0), key="ue_sum_col", help="Rows whose summary contains the substring are invalid.")
          current[et]["summary_invalidate_contains"] = st.text_input("Invalidate if summary contains", value=current[et]["summary_invalidate_contains"], key="ue_sum_contains")
          current[et]["location_column"] = st.selectbox("Location ID column", options=cols or [current[et]["location_column"]], index=(cols.index(current[et]["location_column"]) if current[et]["location_column"] in cols else 0), key="ue_loc_col")
          current[et]["estimate_completion_date_column"] = st.selectbox("Estimate completion date column", options=cols or [current[et]["estimate_completion_date_column"]], index=(cols.index(current[et]["estimate_completion_date_column"]) if current[et]["estimate_completion_date_column"] in cols else 0), key="ue_est_comp_col")
          current[et]["recent_estimate_days"] = int(st.number_input("Recent estimate window (days)", min_value=1, max_value=180, value=int(current[et]["recent_estimate_days"]), key="ue_recent_est"))
          current[et]["customer_column"] = st.selectbox("Customer ID column", options=cols or [current[et]["customer_column"]], index=(cols.index(current[et]["customer_column"]) if current[et]["customer_column"] in cols else 0), key="ue_cust_col")
          # Jobs columns
          jobs_cols = columns_cache.get("Jobs.parquet", [])
          current[et]["jobs_file"] = st.selectbox("Jobs file", options=["Jobs.parquet"], index=0, key="ue_jobs_file")
          current[et]["job_created_date_column"] = st.selectbox("Job created date column", options=jobs_cols or [current[et]["job_created_date_column"]], index=((jobs_cols.index(current[et]["job_created_date_column"]) if current[et]["job_created_date_column"] in jobs_cols else 0) if jobs_cols else 0), key="ue_job_created")
          current[et]["job_scheduled_date_column"] = st.selectbox("Job scheduled date column", options=jobs_cols or [current[et]["job_scheduled_date_column"]], index=((jobs_cols.index(current[et]["job_scheduled_date_column"]) if current[et]["job_scheduled_date_column"] in jobs_cols else 0) if jobs_cols else 0), key="ue_job_sched")
          current[et]["recent_job_days"] = int(st.number_input("Recent job window (±days)", min_value=1, max_value=60, value=int(current[et]["recent_job_days"]), key="ue_recent_job", help="Jobs within this many days before/after the estimate invalidate the event."))
          st.markdown("**Duplicate Handling**")
          if current[et]["dedup_enabled"]:
            current[et]["dedup_customer_column"] = st.selectbox("Customer ID column", options=cols or [current[et]["dedup_customer_column"]], index=(cols.index(current[et]["dedup_customer_column"]) if current[et]["dedup_customer_column"] in cols else 0), key="ue_dedup_customer")
            current[et]["dedup_location_column"] = st.selectbox("Location ID column", options=cols or [current[et]["dedup_location_column"]], index=(cols.index(current[et]["dedup_location_column"]) if current[et]["dedup_location_column"] in cols else 0), key="ue_dedup_location")
          current[et]["months_back"] = int(st.number_input(
            "Months back window",
            min_value=1,
            max_value=120,
            value=int(current[et]["months_back"]),
            key="ue_months_back",
            help="Only consider estimates created within this many months."
          ))
        
        if et == "overdue_maintenance":
          st.markdown("**Source & Columns**")
          current[et]["file"] = st.selectbox(
            "Master file", options=available_files or [current[et].get("file", "Jobs.parquet")],
            index=(available_files.index(current[et]["file"]) if current[et]["file"] in available_files else 0),
            key="om_file"
          )
          cols = columns_cache.get(current[et]["file"], [])
          current[et]["filter_column"] = st.selectbox("Filter column", options=cols or [current[et].get("filter_column", "Job Type")], index=(cols.index(current[et].get("filter_column", "Job Type")) if current[et].get("filter_column", "Job Type") in cols else 0), key="om_filter_col", help="Column to match against the list of values below.")
          match_values = current[et].get("filter_match_values") or []
          match_values_text = "\n".join(match_values) if isinstance(match_values, list) else str(match_values) if match_values else ""
          match_values_input = st.text_area(
            "Filter: match values (one per line, case-insensitive exact match)",
            value=match_values_text,
            key="om_filter_match_values",
            help="Only rows where the filter column exactly matches (case-insensitive) one of these values will be considered."
          )
          current[et]["filter_match_values"] = [v.strip() for v in match_values_input.strip().split("\n") if v.strip()]
          current[et]["status_column"] = st.selectbox("Status column", options=cols or [current[et].get("status_column", "Status")], index=(cols.index(current[et].get("status_column", "Status")) if current[et].get("status_column", "Status") in cols else 0), key="om_status_col", help="Column containing job status.")
          current[et]["status_value"] = st.text_input("Status value (exact match)", value=current[et].get("status_value", "Completed"), key="om_status_value", help="Only rows with this exact status value will be considered.")
          current[et]["date_column"] = st.selectbox("Date column", options=cols or [current[et]["date_column"]], index=(cols.index(current[et]["date_column"]) if current[et]["date_column"] in cols else 0), key="om_date_col", help="Date used to compute months since completion.")
          current[et]["id_column"] = st.selectbox("ID column", options=cols or [current[et]["id_column"]], index=(cols.index(current[et]["id_column"]) if current[et]["id_column"] in cols else 0), key="om_id_col", help="Unique identifier for the entity written to the event file.")
          st.markdown("**Detection Logic (read-only)**")
          match_count = len(current[et].get("filter_match_values", []))
          st.code(
            f"""
FILTER1 = {current[et]['filter_column']} matches one of {match_count} values (case-insensitive exact match)
FILTER2 = {current[et]['status_column']} == '{current[et].get('status_value', 'Completed')}' (exact match)
DATE = parse {current[et]['date_column']} as date
MONTHS_SINCE = months_between(today, DATE)
EVENT_TRUE = FILTER1 AND FILTER2 AND MONTHS_SINCE >= {int(current[et]['months_threshold'])}
""".strip(), language="text")

          st.markdown("**Payload Columns**")
          preselect = current[et].get("payload_columns") or []
          current[et]["payload_columns"] = st.multiselect(
            "Columns to include in event payload",
            options=cols,
            default=[c for c in preselect if c in cols],
            key="om_payload_cols",
            help="Additional columns to include from the source rows when EVENT_TRUE."
          )
          st.markdown("**Output Order**")
          base_order = [
            "event_type", "entity_type", "entity_id", "detected_at",
            "months_overdue", "job_class", "completion_date"
          ]
          all_out_cols = base_order + [c for c in current[et]["payload_columns"] if c not in base_order]
          # Use saved order if present; otherwise default to full order
          saved_order = [c for c in (current[et].get("output_order") or []) if c in all_out_cols]
          # Ensure any newly allowed columns are appended to the end so they are included
          for c in all_out_cols:
            if c not in saved_order:
              saved_order.append(c)
          current[et]["output_order"] = st.multiselect(
            "Arrange output columns order",
            options=all_out_cols,
            default=saved_order,
            key="om_output_order",
            help="The order of columns in the saved Parquet file."
          )

          st.markdown("**Output File**")
          default_name = current[et].get("output_filename") or "overdue_maintenance_master.parquet"
          current[et]["output_filename"] = st.text_input(
            "Output file name",
            value=default_name,
            key="om_out_name_ui",
            help="The Parquet file that will be written in the company's parquet folder"
          )
          base_check_om = _P(current.get(et, {}).get("output_dir") or events_master_default_dir)
          desired_path = base_check_om / current[et]["output_filename"]
          if desired_path.exists():
            st.warning(f"Output file '{current[et]['output_filename']}' already exists.")
            allowed_actions = ["Overwrite", "Change name"]
            current_action = current[et].get("conflict_action")
            if current_action not in allowed_actions:
              current_action = "Overwrite"
            current[et]["conflict_action"] = st.radio(
              "File exists - choose action",
              allowed_actions,
              index=allowed_actions.index(current_action),
              key="om_conflict_action_ui"
            )
            if current[et]["conflict_action"] == "Overwrite":
              current[et]["backup_before_overwrite"] = st.checkbox(
                "Create backup before overwrite", value=bool(current[et].get("backup_before_overwrite", True)), key="om_backup_ui"
              )
            elif current[et]["conflict_action"] == "Change name":
              ts_default = desired_path.with_stem(desired_path.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name
              current[et]["new_filename"] = st.text_input(
                "New file name", value=current[et].get("new_filename") or ts_default, key="om_new_name_ui",
                help="Provide a new name to avoid overwriting the existing file"
              )

          st.markdown("**Invalidation Rules**")
          current[et]["exclude_column"] = st.selectbox(
            "Column to check",
            options=[""] + cols,
            index=(1 + cols.index(current[et]["exclude_column"]) if current[et].get("exclude_column") in cols else 0),
            key="om_exclude_col",
            help="Rows will be excluded if they match the rules below on this column."
          )
          st.markdown("**Duplicate Handling**")
          if current[et]["dedup_enabled"]:
            current[et]["dedup_customer_column"] = st.selectbox(
              "Customer ID column",
              options=cols or [current[et]["dedup_customer_column"]],
              index=(cols.index(current[et]["dedup_customer_column"]) if current[et]["dedup_customer_column"] in cols else 0),
              key="om_dedup_customer_col"
            )
            current[et]["dedup_location_column"] = st.selectbox(
              "Location ID column",
              options=cols or [current[et]["dedup_location_column"]],
              index=(cols.index(current[et]["dedup_location_column"]) if current[et]["dedup_location_column"] in cols else 0),
              key="om_dedup_location_col"
            )
          current[et]["exclude_equals"] = st.text_input(
            "Exclude if equals (exact match)",
            value=current[et].get("exclude_equals", ""),
            key="om_ex_equals",
            help="Exact, case-sensitive match. Leave blank to skip."
          )
          current[et]["exclude_contains"] = st.text_input(
            "Exclude if contains (substring, case-insensitive)",
            value=current[et].get("exclude_contains", ""),
            key="om_ex_contains",
            help="If provided, any row containing this substring will be excluded."
          )
        
        if et == "canceled_jobs":
          st.markdown("**Source & Columns**")
          current[et]["file"] = st.selectbox(
            "Master file", options=available_files or [current[et].get("file", "Jobs.parquet")],
            index=(available_files.index(current[et]["file"]) if current[et]["file"] in available_files else 0),
            key="cj_file"
          )
          cols = columns_cache.get(current[et]["file"], [])
          current[et]["status_column"] = st.selectbox("Status column", options=cols or [current[et]["status_column"]], index=(cols.index(current[et]["status_column"]) if current[et]["status_column"] in cols else 0), key="cj_status_col")
          current[et]["status_canceled_value"] = st.text_input("Canceled status value", value=current[et]["status_canceled_value"], key="cj_status_val")
          current[et]["date_column"] = st.selectbox("Completion date column", options=cols or [current[et]["date_column"]], index=(cols.index(current[et]["date_column"]) if current[et]["date_column"] in cols else 0), key="cj_date_col")
          current[et]["id_column"] = st.selectbox("ID column", options=cols or [current[et]["id_column"]], index=(cols.index(current[et]["id_column"]) if current[et]["id_column"] in cols else 0), key="cj_id_col")
          st.markdown("**Detection Logic (read-only)**")
          st.code(
            f"""
FILTER1 = {current[et]['status_column']} == '{current[et]['status_canceled_value']}'
FILTER2 = {current[et]['date_column']} within last {int(current[et]['months_back'])} months
INVALIDATION = For each row, find other rows with same {current[et]['location_column']}. If any has {current[et]['invalidation_date_column']} within last {int(current[et]['invalidation_days_recent'])} days AND {current[et]['status_column']} != '{current[et]['status_canceled_value']}', then exclude.
""".strip(), language="text")
          st.markdown("**Payload Columns**")
          preselect_cj = current[et].get("payload_columns") or []
          current[et]["payload_columns"] = st.multiselect(
            "Columns to include in event payload",
            options=cols,
            default=[c for c in preselect_cj if c in cols],
            key="cj_payload_cols",
            help="Additional columns to include from the source rows when detected."
          )
          st.markdown("**Output Order**")
          base_order_cj = [
            "event_type", "entity_type", "entity_id", "detected_at",
            "status", "completion_date", "location_id", "cancellation_age_months"
          ]
          all_out_cols_cj = base_order_cj + [c for c in current[et]["payload_columns"] if c not in base_order_cj]
          saved_order_cj = [c for c in (current[et].get("output_order") or []) if c in all_out_cols_cj]
          for c in all_out_cols_cj:
            if c not in saved_order_cj:
              saved_order_cj.append(c)
          current[et]["output_order"] = st.multiselect(
            "Arrange output columns order",
            options=all_out_cols_cj,
            default=saved_order_cj,
            key="cj_output_order",
            help="The order of columns in the saved Parquet file."
          )
          st.markdown("**Output File**")
          default_name_cj = current[et].get("output_filename") or "canceled_jobs_master.parquet"
          current[et]["output_filename"] = st.text_input(
            "Output file name",
            value=default_name_cj,
            key="cj_out_name_ui",
            help="The Parquet file name to write."
          )
          base_check_cj = events_master_default_dir
          desired_path_cj = base_check_cj / current[et]["output_filename"]
          if desired_path_cj.exists():
            st.warning(f"Output file '{current[et]['output_filename']}' already exists.")
            allowed_actions_cj = ["Overwrite", "Change name"]
            current_action_cj = current[et].get("conflict_action")
            if current_action_cj not in allowed_actions_cj:
              current_action_cj = "Overwrite"
            current[et]["conflict_action"] = st.radio(
              "File exists - choose action",
              allowed_actions_cj,
              index=allowed_actions_cj.index(current_action_cj),
              key="cj_conflict_action_ui"
            )
            if current[et]["conflict_action"] == "Overwrite":
              current[et]["backup_before_overwrite"] = st.checkbox(
                "Create backup before overwrite", value=bool(current[et].get("backup_before_overwrite", True)), key="cj_backup_ui"
              )
            elif current[et]["conflict_action"] == "Change name":
              ts_default_cj = desired_path_cj.with_stem(desired_path_cj.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name
              current[et]["new_filename"] = st.text_input(
                "New file name", value=current[et].get("new_filename") or ts_default_cj, key="cj_new_name_ui",
                help="Provide a new name to avoid overwriting the existing file"
              )
          st.markdown("**Invalidation Logic**")
          current[et]["location_column"] = st.selectbox(
            "Location ID column",
            options=cols or [current[et]["location_column"]],
            index=(cols.index(current[et]["location_column"]) if current[et]["location_column"] in cols else 0),
            key="cj_loc_col",
            help="Rows with the same Location ID are considered the same site; recent non-canceled work at the same site invalidates a cancel."
          )
          current[et]["invalidation_date_column"] = st.selectbox(
            "Invalidation date column",
            options=cols or [current[et]["invalidation_date_column"]],
            index=(cols.index(current[et]["invalidation_date_column"]) if current[et]["invalidation_date_column"] in cols else 0),
            key="cj_inv_date_col",
            help="Date used to detect recent non-canceled work (e.g., Completed/Completion Date)."
          )
          current[et]["invalidation_days_recent"] = int(st.number_input(
            "Invalidation window (days)",
            min_value=1,
            max_value=365,
            value=int(current[et]["invalidation_days_recent"]),
            key="cj_inv_days",
            help="If any non-canceled job at the same Location ID has this date within the last N days, mark the cancellation as invalid."
          ))
          # Additional invalidation: Summary contains "test"
          st.markdown("**Additional Invalidation**")
          current[et]["summary_column"] = st.selectbox(
            "Summary column",
            options=[""] + cols,
            index=(([""] + cols).index(current[et].get("summary_column", "")) if current[et].get("summary_column", "") in ([""] + cols) else 0),
            key="cj_summary_col",
            help="If provided, rows whose summary contains 'test' (case-insensitive) will be excluded."
          )
          current[et]["summary_invalidate_contains"] = st.text_input(
            "Invalidate if summary contains",
            value=current[et].get("summary_invalidate_contains", "test"),
            key="cj_summary_contains",
            help="Substring to search for in the summary to invalidate rows."
          )
          st.markdown("**Duplicate Handling**")
          if current[et]["dedup_enabled"]:
            current[et]["dedup_customer_column"] = st.selectbox(
              "Customer ID column",
              options=cols or [current[et]["dedup_customer_column"]],
              index=(cols.index(current[et]["dedup_customer_column"]) if current[et]["dedup_customer_column"] in cols else 0),
              key="cj_dedup_customer_col"
            )
            current[et]["dedup_location_column"] = st.selectbox(
              "Location ID column",
              options=cols or [current[et]["dedup_location_column"]],
              index=(cols.index(current[et]["dedup_location_column"]) if current[et]["dedup_location_column"] in cols else 0),
              key="cj_dedup_location_col"
            )
        
        if et == "aging_systems":
          st.markdown("**Source & Columns**")
          current[et]["locations_file"] = st.selectbox(
            "Locations master file",
            options=available_files or [current[et].get("locations_file", "Locations.parquet")],
            index=(available_files.index(current[et]["locations_file"]) if current[et]["locations_file"] in available_files else 0),
            key="as_locations_file"
          )
          current[et]["jobs_file"] = st.selectbox(
            "Jobs master file",
            options=available_files or [current[et].get("jobs_file", "Jobs.parquet")],
            index=(available_files.index(current[et]["jobs_file"]) if current[et]["jobs_file"] in available_files else 0),
            key="as_jobs_file"
          )
          # columns cache
          loc_cols = columns_cache.get(current[et]["locations_file"], [])
          job_cols = columns_cache.get(current[et]["jobs_file"], [])
          current[et]["location_id_column"] = st.selectbox(
            "Location ID column",
            options=loc_cols or [current[et]["location_id_column"]],
            index=(loc_cols.index(current[et]["location_id_column"]) if current[et]["location_id_column"] in loc_cols else 0),
            key="as_loc_id_col"
          )
          default_payload = current[et].get("job_payload_fields") or [
            "Summary", "Job Type", "Status", "Created Date", "Completion Date", "Customer ID"
          ]
          current[et]["job_payload_fields"] = st.multiselect(
            "Job payload fields",
            options=job_cols,
            default=[c for c in default_payload if c in job_cols],
            key="as_job_payload",
            help="Fields to include per job in the JSON payload."
          )
          st.markdown("**Field Mapping**")
          mapping = current[et].get("payload_field_mapping") or {}
          new_mapping = {}
          for f in current[et]["job_payload_fields"]:
            new_mapping[f] = st.text_input(f"{f} →", value=str(mapping.get(f, f)), key=f"as_map_{f}")
          current[et]["payload_field_mapping"] = new_mapping
          st.markdown("**Detection Logic (read-only)**")
          st.code(
            f"""
COLLECT LOCATIONS = all {current[et]['location_id_column']} from {current[et]['locations_file']}
MATCH JOBS = for each Location ID, find rows in {current[et]['jobs_file']}
PAYLOAD = for each matched job, include {current[et]['job_payload_fields']} as JSON
OUTPUT = one row per Location ID with an array of job payloads
""".strip(), language="text")
          st.markdown("**Output File**")
          current[et]["output_filename"] = st.text_input(
            "Output file name",
            value=current[et].get("output_filename") or "location_jobs_history.parquet",
            key="as_out_name"
          )
          # Pre-run existence check for output file (placed under Output File section)
          base_check_as = _P(current.get(et, {}).get("output_dir") or events_master_default_dir)
          desired_path_as = base_check_as / current[et]["output_filename"]
          if desired_path_as.exists():
            st.warning(f"Output file '{current[et]['output_filename']}' already exists.")
            allowed_actions_as = ["Overwrite", "Change name"]
            curr_act_as = current[et].get("conflict_action")
            if curr_act_as not in allowed_actions_as:
              curr_act_as = "Overwrite"
            current[et]["conflict_action"] = st.radio("File exists - choose action", allowed_actions_as, index=allowed_actions_as.index(curr_act_as), key="as_conflict")
            if current[et]["conflict_action"] == "Overwrite":
              current[et]["backup_before_overwrite"] = st.checkbox("Create backup before overwrite", value=bool(current[et].get("backup_before_overwrite", True)), key="as_backup")
            else:
              ts_default_as = desired_path_as.with_stem(desired_path_as.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name
              current[et]["new_filename"] = st.text_input("New file name", value=current[et].get("new_filename") or ts_default_as, key="as_new_name")
          st.markdown("**Analysis Mode**")
          current[et]["analysis_mode"] = st.radio(
            "Select analysis mode",
            ["basic", "llm"],
            index=(0 if (current[et].get("analysis_mode") or "basic") == "basic" else 1),
            key="as_mode",
            help="Choose between a simple rules-based analysis or an LLM-driven analysis."
          )
          current[et]["analysis_output_filename"] = st.text_input(
            "Analysis output file name",
            value=current[et].get("analysis_output_filename") or "aging_systems_basic.parquet",
            key="as_analysis_out"
          )
          current[et]["llm_analysis_output_filename"] = st.text_input(
            "LLM analysis output file name",
            value=current[et].get("llm_analysis_output_filename") or "aging_systems.parquet",
            key="as_llm_analysis_out",
            help="Filename for the LLM analysis results (aging_systems.parquet by default)"
          )
          
          st.markdown("**Additional Payload Columns**")
          # Get available columns from locations file for payload selection
          loc_cols = columns_cache.get(current[et]["locations_file"], [])
          current[et]["additional_payload_columns"] = st.multiselect(
            "Additional columns to include in output",
            options=loc_cols,
            default=current[et].get("additional_payload_columns", []),
            key="as_additional_payload",
            help="Select additional columns from locations data to include in the aging systems output (e.g., most_likely_next_job, contractor_market_share)"
          )
          
          st.markdown("**Permit Integration**")
          current[et]["permit_data_path"] = st.text_input(
            "Permit parquet path",
            value=current[et].get("permit_data_path") or "global_data/permits/permit_data.parquet",
            key="as_permit_path",
            help="Path to the global permits parquet dataset."
          )
          current[et]["permit_max_edit_distance"] = int(st.number_input(
            "Address match distance (max edits)",
            min_value=0,
            max_value=5,
            value=int(current[et].get("permit_max_edit_distance", 2)),
            key="as_permit_max_edits",
            help="Maximum Levenshtein edits allowed on the street core for a fuzzy match."
          ))
          
          # Permit fields selection
          perm_cols = []
          try:
            from pyarrow import parquet as pq
            ppath = _P(current[et]["permit_data_path"])
            if ppath.exists():
              perm_cols = list(pq.ParquetFile(ppath).schema.names)
          except Exception:
            perm_cols = []
          
          base_fields = current[et].get("permit_fields") or []
          opts = perm_cols or base_fields
          current[et]["permit_fields"] = st.multiselect(
            "Permit payload fields",
            options=opts,
            default=[c for c in base_fields if c in opts] or base_fields,
            key="as_permit_fields",
            help="Select permit columns to include in the payload."
          )
          
          st.markdown("**Permit Field Mapping**")
          pf = current[et].get("permit_fields") or []
          pmap = current[et].get("permit_field_mapping") or {}
          new_pmap = {}
          for f in pf:
            new_pmap[f] = st.text_input(f"{f} →", value=str(pmap.get(f, f)), key=f"as_permit_map_{f}")
          current[et]["permit_field_mapping"] = new_pmap
          
          st.markdown("**Additional Output Files**")
          current[et]["permit_history_output_filename"] = st.text_input(
            "Permit history output file",
            value=current[et].get("permit_history_output_filename") or "location_permit_history.parquet",
            key="as_permit_hist_out"
          )
          current[et]["profile_output_filename"] = st.text_input(
            "Profile output file",
            value=current[et].get("profile_output_filename") or "location_profile.parquet", 
            key="as_profile_out"
          )
        
        if et == "lost_customers":
          st.markdown("**Source Files & Columns**")
          current[et]["customers_file"] = st.selectbox(
            "Customers file",
            options=available_files or [current[et].get("customers_file", "Customers.parquet")],
            index=(available_files.index(current[et]["customers_file"]) if current[et].get("customers_file") in available_files else 0),
            key="lc_customers_file"
          )
          current[et]["calls_file"] = st.selectbox(
            "Calls file",
            options=available_files or [current[et].get("calls_file", "Calls.parquet")],
            index=(available_files.index(current[et]["calls_file"]) if current[et].get("calls_file") in available_files else 0),
            key="lc_calls_file"
          )
          
          customers_cols = columns_cache.get(current[et]["customers_file"], [])
          calls_cols = columns_cache.get(current[et]["calls_file"], [])
          
          current[et]["customer_id_column"] = st.selectbox(
            "Customer ID column",
            options=customers_cols or [current[et].get("customer_id_column", "Customer ID")],
            index=((customers_cols.index(current[et].get("customer_id_column", "Customer ID")) if current[et].get("customer_id_column", "Customer ID") in customers_cols else 0) if customers_cols else 0),
            key="lc_customer_id"
          )
          current[et]["customer_address_column"] = st.selectbox(
            "Customer address column",
            options=customers_cols or [current[et].get("customer_address_column", "Full Address")],
            index=((customers_cols.index(current[et].get("customer_address_column", "Full Address")) if current[et].get("customer_address_column", "Full Address") in customers_cols else 0) if customers_cols else 0),
            key="lc_customer_address"
          )
          current[et]["call_date_column"] = st.selectbox(
            "Call date column",
            options=calls_cols or [current[et].get("call_date_column", "Call Date")],
            index=((calls_cols.index(current[et].get("call_date_column", "Call Date")) if current[et].get("call_date_column", "Call Date") in calls_cols else 0) if calls_cols else 0),
            key="lc_call_date"
          )
          
          st.markdown("**Company Configuration**")
          company_names = current[et].get("company_names", ["McCullough Heating & Air", "McCullough Heating and Air"])
          current[et]["company_names"] = st.text_area(
            "Company names (one per line)",
            value="\n".join(company_names),
            key="lc_company_names",
            help="Enter your company names, one per line. These will be excluded when identifying competitors."
          ).strip().split("\n")
          
          st.markdown("**Analysis Settings**")
          current[et]["require_permit_data"] = st.checkbox(
            "Require permit data for analysis",
            value=bool(current[et].get("require_permit_data", True)),
            key="lc_require_permits",
            help="If enabled, customers without matching permit data will be skipped."
          )
          current[et]["min_contact_history"] = int(st.number_input(
            "Minimum contact history (calls)",
            min_value=1,
            max_value=50,
            value=int(current[et].get("min_contact_history", 2)),
            key="lc_min_contacts",
            help="Minimum number of calls required for a customer to be analyzed."
          ))
          current[et]["exclude_nan_competitor"] = st.checkbox(
            "Exclude lost customers with no competitor identified",
            value=bool(current[et].get("exclude_nan_competitor", True)),
            key="lc_exclude_nan",
            help="If enabled, lost customers without a specific competitor identified will be excluded from results."
          )
          
          st.markdown("**Detection Logic (read-only)**")
          exclude_text = "EXCLUDE = lost customers with no competitor identified" if current[et]['exclude_nan_competitor'] else "INCLUDE = all lost customers (even without specific competitor)"
          st.code(
            f"""
COLLECT CUSTOMERS = all {current[et]['customer_id_column']} from {current[et]['customers_file']}
MATCH CALLS = for each Customer ID, find rows in {current[et]['calls_file']}
CALCULATE = first_contact_date (oldest {current[et]['call_date_column']})
CALCULATE = last_contact_date (newest {current[et]['call_date_column']})
MATCH PERMITS = match {current[et]['customer_address_column']} to permit addresses
ANALYZE = check if competitor permits issued after last_contact_date
{exclude_text}
OUTPUT = lost customers with competitor details
""".strip(), language="text")
          
          st.markdown("**Output File**")
          current[et]["output_filename"] = st.text_input(
            "Output file name",
            value=current[et].get("output_filename", "lost_customers_analysis.parquet"),
            key="lc_out_name",
            help="Name for the output parquet file containing lost customer analysis results."
          )
        
        current[et]["processing_limit"] = int(st.number_input("Max records (0 = no limit)", min_value=0, max_value=1_000_000, value=int(current[et]["processing_limit"]), key=f"{et}_limit"))
        
        # Add automation section for each event type
        st.markdown("---")
        st.markdown("### 🤖 Automation")
        
        if current[et]["enabled"]:
          # Use checkbox to show/hide automation instead of nested expander
          show_automation = st.checkbox(f"Configure scheduling for {et.replace('_', ' ')}", key=f"show_automation_{et}")
          
          if show_automation:
            st.info(f"Configure automatic scanning for {et.replace('_', ' ')} events")
            
            # Render schedule configuration
            schedule_config = render_schedule_config(
              key_prefix=f"event_{et}",
              default_interval_minutes=120  # 2 hours default
            )
            
            # Create scheduled task button
            if st.button(f"📅 Create Schedule", type="primary", key=f"create_{et}_schedule"):
              task_config = {
                'event_type': et,
                'scan_all_events': False
              }
              
              success = create_scheduled_task(
                task_type=TaskType.HISTORICAL_EVENT_SCAN,
                company=company,
                task_name=f"{et.replace('_', ' ').title()} Scan - {company}",
                task_description=f"Automated scan for {et.replace('_', ' ')} events",
                schedule_config=schedule_config,
                task_config_overrides=task_config
              )
              
              if success:
                st.success(f"✅ Scheduled {et.replace('_', ' ')} scan created!")
                st.rerun()
              else:
                st.error("Failed to create scheduled task")
            
            # Show existing schedules for this event type
            st.markdown("**Existing Schedules:**")
            from datahound.scheduler import DataHoundScheduler
            scheduler = DataHoundScheduler(Path.cwd())
            tasks = scheduler.get_tasks_by_company(company)
            event_tasks = [t for t in tasks if t.task_config.event_type == et]
            
            if event_tasks:
              for task in event_tasks:
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                  st.write(f"📅 {task.name}")
                with col2:
                  if task.next_run:
                    st.caption(f"Next: {task.next_run.strftime('%m/%d %H:%M')}")
                with col3:
                  if st.button("🗑️", key=f"delete_{task.task_id}"):
                    scheduler.delete_task(task.task_id)
                    st.rerun()
            else:
              st.caption("No schedules configured for this event type")
        else:
          st.info("⚠️ Enable this event type to configure automation")

    # Option to schedule all events together
    st.markdown("---")
    with st.expander("🔄 Schedule All Events Together"):
      st.markdown("### Batch Event Scheduling")
      st.info("Create a single scheduled task that runs all enabled event scans at once.")
      
      # Show which events are enabled
      enabled_events = [et for et in EVENT_TYPES if current[et]["enabled"]]
      if enabled_events:
        st.write(f"**Enabled events:** {', '.join([e.replace('_', ' ').title() for e in enabled_events])}")
        
        all_schedule_config = render_schedule_config(
          key_prefix="all_events",
          default_interval_minutes=240  # 4 hours default for all events
        )
        
        if st.button("📅 Schedule All Events", type="primary", key="create_all_events_schedule"):
          task_config = {
            'scan_all_events': True,
            'event_type': None
          }
          
          success = create_scheduled_task(
            task_type=TaskType.HISTORICAL_EVENT_SCAN,
            company=company,
            task_name=f"All Events Scan - {company}",
            task_description="Automated scan for all enabled event types",
            schedule_config=all_schedule_config,
            task_config_overrides=task_config
          )
          
          if success:
            st.success("✅ Scheduled all events scan created successfully!")
            st.rerun()
          else:
            st.error("Failed to create scheduled task")
      else:
        st.warning("No events are currently enabled! Enable at least one event type to use batch scheduling.")
    
    col1, col2 = st.columns(2)
    with col1:
      save_clicked = st.button("Save Config", type="primary", key="save_hist_cfg")
    with col2:
      st.caption("Use Export/Import below")

  if save_clicked:
    # Sanitize output order for overdue_maintenance: keep only base + payload, and include any missing
    try:
      om = current.get("overdue_maintenance", {})
      base_order = [
        "event_type", "entity_type", "entity_id", "detected_at",
        "months_overdue", "job_class", "completion_date"
      ]
      payload = [c for c in (om.get("payload_columns") or [])]
      allowed = base_order + [c for c in payload if c not in base_order]
      # Keep the order chosen by the user (already in current), then prune/append
      existing_order = [c for c in (current.get("overdue_maintenance", {}).get("output_order") or [])]
      pruned = [c for c in existing_order if c in allowed]
      # append any allowed not present to ensure full coverage
      for c in allowed:
        if c not in pruned:
          pruned.append(c)
      current["overdue_maintenance"]["output_order"] = pruned
    except Exception:
      pass
    # Sanitize output order for canceled_jobs
      # Sanitize output order for unsold_estimates
      try:
        ue = current.get("unsold_estimates", {})
        base_order_ue = [
          "event_type", "entity_type", "entity_id", "detected_at",
          "estimate_status", "creation_date", "opportunity_status", "location_id"
        ]
        payload_ue = [c for c in (ue.get("payload_columns") or [])]
        allowed_ue = base_order_ue + [c for c in payload_ue if c not in base_order_ue]
        existing_order_ue = [c for c in (current.get("unsold_estimates", {}).get("output_order") or [])]
        pruned_ue = [c for c in existing_order_ue if c in allowed_ue]
        for c in allowed_ue:
          if c not in pruned_ue:
            pruned_ue.append(c)
        current["unsold_estimates"]["output_order"] = pruned_ue
      except Exception:
        pass
    try:
      cj = current.get("canceled_jobs", {})
      base_order_cj = [
        "event_type", "entity_type", "entity_id", "detected_at",
        "status", "completion_date", "location_id", "cancellation_age_months"
      ]
      payload_cj = [c for c in (cj.get("payload_columns") or [])]
      allowed_cj = base_order_cj + [c for c in payload_cj if c not in base_order_cj]
      existing_order_cj = [c for c in (current.get("canceled_jobs", {}).get("output_order") or [])]
      pruned_cj = [c for c in existing_order_cj if c in allowed_cj]
      for c in allowed_cj:
        if c not in pruned_cj:
          pruned_cj.append(c)
      current["canceled_jobs"]["output_order"] = pruned_cj
    except Exception:
      pass
    _save_config(company, {"events": current})
    st.success("Saved historical events config")

  col_exp1, col_exp2 = st.columns(2)
  with col_exp1:
    st.download_button("Export JSON", data=json.dumps({"events": current}, indent=2), file_name=f"{company}_historical_events_config.json", mime="application/json")
  with col_exp2:
    uploaded = st.file_uploader("Import JSON", type="json", key="hist_import")
    if uploaded and st.button("Apply Imported Config"):
      try:
        new_cfg = json.load(uploaded)
        _save_config(company, new_cfg)
        st.success("Imported configuration. Reload to see values.")
      except Exception as e:
        st.error(str(e))

  st.divider()
  st.subheader("Run")
  persist_on_run = st.checkbox("Write output file (Parquet)", value=True)
  st.caption(f"Default output dir: `{(events_master_default_dir).as_posix()}`")
  custom_output_dir = st.text_input("Output directory (optional)", value=current.get("overdue_maintenance", {}).get("output_dir", (events_master_default_dir.as_posix() if 'overdue_maintenance' in current else "")), key="om_out_dir")
  # save back to current config in-memory so run uses it
  if "overdue_maintenance" in current:
    current["overdue_maintenance"]["output_dir"] = custom_output_dir.strip()
  include_sample = st.checkbox("Include sample row", value=True)
  if st.button("Run Scans", type="primary"):
    written: List[str] = []
    # Progress UI
    enabled_order = [et for et in EVENT_TYPES if (isinstance(current.get(et), dict) and current.get(et, {}).get("enabled"))]
    total_events = max(1, len(enabled_order))
    events_completed = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    term_box = st.empty()
    ui_lines: List[str] = []
    # initialize terminal window
    term_box.markdown(
      """
<div style="height:320px;overflow-y:auto;background:#0b1021;color:#e5e7eb;padding:8px;border-radius:6px;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;font-size:12px;line-height:1.4;white-space:pre;"></div>
""",
      unsafe_allow_html=True,
    )
    def ui_log(msg: str) -> None:
      ui_lines.append(msg)
      content = "\n".join(ui_lines[-800:])
      safe = html.escape(content)
      term_box.markdown(
        f"""
<div style="height:320px;overflow-y:auto;background:#0b1021;color:#e5e7eb;padding:8px;border-radius:6px;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;font-size:12px;line-height:1.4;white-space:pre;">{safe}</div>
""",
        unsafe_allow_html=True,
      )
      # tiny yield to allow UI to repaint during long loops
      try:
        time.sleep(0.002)
      except Exception:
        pass
    def update_progress(detail: str = "") -> None:
      pct = int((events_completed / total_events) * 100)
      progress_bar.progress(min(max(pct, 0), 100))
      if detail:
        status_text.text(detail)
    update_progress("Starting scans...")
    # choose base output directory
    cfg_out_dir_om = (current.get("overdue_maintenance", {}).get("output_dir") or "").strip()
    base_default_om = (_P(cfg_out_dir_om) if cfg_out_dir_om else events_master_default_dir)
    base_default_om.mkdir(parents=True, exist_ok=True)
    cfg_out_dir_cj = (current.get("canceled_jobs", {}).get("output_dir") or "").strip()
    base_default_cj = (_P(cfg_out_dir_cj) if cfg_out_dir_cj else events_master_default_dir)
    base_default_cj.mkdir(parents=True, exist_ok=True)
    # logging setup → all logs under data/<company>/logs with structured subdirs
    logs_root = _P(cfg.data_dir).parent / "logs"
    event_log_dir = logs_root / "historical" / "overdue_maintenance"
    event_log_dir.mkdir(parents=True, exist_ok=True)
    scan_log_path = event_log_dir / "scan.jsonl"
    def _log(action: str, details: Dict[str, Any]) -> None:
      try:
        with open(scan_log_path, "a", encoding="utf-8") as f:
          rec = {"ts": datetime.now(UTC).isoformat(), "company": company, "action": action}
          rec.update(details or {})
          f.write(json.dumps(rec) + "\n")
      except Exception:
        pass
    for et, cfg in current.items():
      if not isinstance(cfg, dict) or not cfg.get("enabled"):
        continue
      ui_log(f"[{et}] Starting...")
      if et == "overdue_maintenance":
        try:
          update_progress("Running overdue_maintenance")
          src = parquet_dir / cfg.get("file", "Jobs.parquet")
          if not src.exists():
            st.error(f"Source file not found: {src}")
            continue
          df = pd.read_parquet(src, engine="pyarrow")
          ui_log(f"[{et}] Loaded source {src.name} rows={len(df)}")
          _log("scan_start", {
            "event_type": "overdue_maintenance",
            "source_file": str(src),
            "rows_source": int(len(df)),
            "filter_column": cfg.get("filter_column"),
            "filter_match_values": cfg.get("filter_match_values", []),
            "status_column": cfg.get("status_column"),
            "status_value": cfg.get("status_value"),
            "date_column": cfg.get("date_column"),
            "id_column": cfg.get("id_column"),
            "months_threshold": int(cfg.get("months_threshold", 12)),
            "processing_limit": int(cfg.get("processing_limit", 0) or 0),
            "output_filename": cfg.get("output_filename") or "overdue_maintenance_master.parquet",
            "conflict_action": cfg.get("conflict_action", "Overwrite"),
          })
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
          # Filter by Job Type match values
          if fcol in df.columns:
            if match_values_lower:
              col_values_lower = df[fcol].astype(str).str.lower().str.strip()
              filtered = df[col_values_lower.isin(match_values_lower)].copy()
            else:
              st.warning(f"No match values configured; skipping filter.")
              filtered = df.copy()
          else:
            st.warning(f"Filter column '{fcol}' not found in {src.name}; skipping filter.")
            filtered = df.copy()
          # Filter by Status column
          if status_col in filtered.columns:
            status_mask = filtered[status_col].astype(str).str.strip() == status_value
            filtered = filtered[status_mask].copy()
          else:
            st.warning(f"Status column '{status_col}' not found in {src.name}; skipping status filter.")
          _log("filter_applied", {"before": int(len(df)), "after": int(len(filtered))})
          # Compute months since date
          if date_col in filtered.columns:
            dt = pd.to_datetime(filtered[date_col], errors="coerce")
            now = pd.Timestamp.utcnow().tz_localize(None)
            months = ((now.year - dt.dt.year) * 12 + (now.month - dt.dt.month)).astype("Int64")
            filtered["months_since"] = months
            overdue = filtered[(filtered["months_since"] >= months_threshold)]
          else:
            st.warning(f"Date column '{date_col}' not found; no results.")
            overdue = filtered.iloc[0:0]
          _log("threshold_applied", {"threshold_months": months_threshold, "after": int(len(overdue))})
          # Apply invalidation rules
          ex_col = cfg.get("exclude_column") or ""
          ex_eq = cfg.get("exclude_equals") or ""
          ex_contains = cfg.get("exclude_contains") or ""
          if ex_col and ex_col in overdue.columns:
            mask = pd.Series([True] * len(overdue), index=overdue.index)
            removed_eq = 0
            removed_contains = 0
            before_inv = int(len(overdue))
            if ex_eq:
              eq_mask = overdue[ex_col].astype(str) != ex_eq
              removed_eq = int((~eq_mask).sum())
              mask &= eq_mask
            if ex_contains:
              cont_mask = ~overdue[ex_col].astype(str).str.lower().str.contains(str(ex_contains).lower(), na=False)
              removed_contains = int((~cont_mask).sum())
              mask &= cont_mask
            overdue = overdue[mask]
            _log("invalidation_applied", {"removed_eq": removed_eq, "removed_contains": removed_contains, "before": before_inv, "after": int(len(overdue))})
          # Apply processing limit
          limit = int(cfg.get("processing_limit", 0) or 0)
          if limit > 0 and not overdue.empty:
            overdue = overdue.head(limit)
            _log("limit_applied", {"limit": limit, "after": int(len(overdue))})
          # Build events
          if not overdue.empty:
            payload_cols = [c for c in (cfg.get("payload_columns") or []) if c in overdue.columns]
            base_cols = {
              "event_type": "overdue_maintenance",
              "entity_type": "job",
              "entity_id": overdue[id_col].astype(str) if id_col in overdue.columns else overdue.index.astype(str),
              "detected_at": datetime.now(UTC).isoformat(),
              "months_overdue": overdue.get("months_since", pd.Series([], dtype="Int64")),
            }
            # optional commonly useful fields
            if fcol in overdue.columns:
              base_cols["job_class"] = overdue[fcol].astype(str)
            if date_col in overdue.columns:
              base_cols["completion_date"] = overdue[date_col].astype(str)
            events = pd.DataFrame(base_cols)
            for col in payload_cols:
              events[col] = overdue[col].astype(str)
            # Deduplication (last step before save)
            if bool(cfg.get("dedup_enabled", False)):
              cust_col = cfg.get("dedup_customer_column", "Customer ID")
              loc_col_dedup = cfg.get("dedup_location_column", "Location ID")
              if cust_col in overdue.columns and loc_col_dedup in overdue.columns and date_col in overdue.columns:
                od = pd.to_datetime(overdue[date_col], errors="coerce")
                dedup_key = overdue[cust_col].astype(str) + "||" + overdue[loc_col_dedup].astype(str)
                order_df = pd.DataFrame({
                  "key": dedup_key,
                  "comp_date": od,
                }, index=overdue.index)
                order_df = order_df.sort_values(["key", "comp_date"], ascending=[True, False])
                keep_labels = order_df.groupby("key").head(1).index.tolist()
                before_len = int(len(events))
                events = events.loc[keep_labels]
                _log("dedup_applied", {"kept": int(len(events)), "removed": int(before_len - int(len(events)))})
            # Determine output path and handle conflicts using previously chosen options
            desired_name = cfg.get("output_filename") or "overdue_maintenance_master.parquet"
            conflict_action = cfg.get("conflict_action", "Overwrite")
            new_name_cfg = cfg.get("new_filename")
            backup_before = bool(cfg.get("backup_before_overwrite", True))
            out = base_default_om / desired_name
            if out.exists():
              if conflict_action == "Change name":
                alt_name = new_name_cfg or (out.with_stem(out.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name)
                out = base_default_om / alt_name
              elif conflict_action == "Overwrite":
                if backup_before:
                  try:
                    backup_path = out.with_name(out.stem + f"_backup_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.parquet")
                    out.replace(backup_path)
                    st.info(f"Backup created: {backup_path.name}")
                  except Exception:
                    pass
              _log("output_decision", {"exists": True, "action": conflict_action, "target": str(out)})
            else:
              _log("output_decision", {"exists": False, "target": str(out)})
            # Reorder output columns per user preference
            order = [c for c in (cfg.get("output_order") or []) if c in events.columns]
            if order:
              events = events[[c for c in order if c in events.columns]]
            events.to_parquet(out, index=False)
            written.append(out.name)
            _log("write_complete", {"rows_written": int(len(events)), "output_file": str(out)})
            st.success(f"Overdue maintenance events: {len(events)}")
            st.dataframe(events.head(100), width='stretch')
            ui_log(f"[overdue_maintenance] Wrote output {out.name} rows={len(events)}")
          else:
            st.info("No overdue maintenance events found.")
        except Exception as e:
          st.error(f"Overdue maintenance scan error: {e}")
        finally:
          events_completed += 1
          update_progress(f"Completed overdue_maintenance")
      elif et == "canceled_jobs":
        try:
          update_progress("Running canceled_jobs")
          src = parquet_dir / cfg.get("file", "Jobs.parquet")
          if not src.exists():
            st.error(f"Source file not found: {src}")
            continue
          df = pd.read_parquet(src, engine="pyarrow")
          # event-specific logs directory
          cj_log_dir = logs_root / "historical" / "canceled_jobs"
          cj_log_dir.mkdir(parents=True, exist_ok=True)
          cj_log_path = cj_log_dir / "scan.jsonl"
          def _log_cj(action: str, details: Dict[str, Any]) -> None:
            try:
              with open(cj_log_path, "a", encoding="utf-8") as f:
                rec = {"ts": datetime.now(UTC).isoformat(), "company": company, "action": action}
                rec.update(details or {})
                f.write(json.dumps(rec) + "\n")
            except Exception:
              pass
          _log_cj("scan_start", {
            "event_type": "canceled_jobs",
            "source_file": str(src),
            "rows_source": int(len(df)),
            "status_column": cfg.get("status_column"),
            "status_canceled_value": cfg.get("status_canceled_value"),
            "date_column": cfg.get("date_column"),
            "months_back": int(cfg.get("months_back", 24)),
            "id_column": cfg.get("id_column"),
            "location_column": cfg.get("location_column"),
            "invalidation_date_column": cfg.get("invalidation_date_column"),
            "invalidation_days_recent": int(cfg.get("invalidation_days_recent", 30)),
            "processing_limit": int(cfg.get("processing_limit", 0) or 0),
            "output_filename": cfg.get("output_filename") or "canceled_jobs_master.parquet",
            "conflict_action": cfg.get("conflict_action", "Overwrite"),
          })
          status_col = cfg.get("status_column", "Status")
          canceled_val = str(cfg.get("status_canceled_value", "Canceled"))
          date_col = cfg.get("date_column", "Completion Date")
          id_col = cfg.get("id_column", "Job ID")
          months_back = int(cfg.get("months_back", 24))
          loc_col = cfg.get("location_column", "Location ID")
          inv_date_col = cfg.get("invalidation_date_column", date_col)
          inv_days = int(cfg.get("invalidation_days_recent", 30))
          # Filter 1: status == canceled
          if status_col in df.columns:
            status_norm = df[status_col].astype(str).str.strip().str.lower()
            canceled_mask = status_norm == canceled_val.strip().lower()
            filtered = df[canceled_mask].copy()
          else:
            st.warning(f"Status column '{status_col}' not found in {src.name}; skipping.")
            filtered = df.iloc[0:0]
          _log_cj("status_filter_applied", {"before": int(len(df)), "after": int(len(filtered))})
          # Filter 2: date within last months_back months
          if not filtered.empty and date_col in filtered.columns:
            dt = pd.to_datetime(filtered[date_col], errors="coerce")
            now = pd.Timestamp.utcnow().tz_localize(None)
            months_since = ((now.year - dt.dt.year) * 12 + (now.month - dt.dt.month)).astype("Int64")
            recent = filtered[(months_since <= months_back)].copy()
            try:
              recent["cancellation_age_months"] = months_since.loc[recent.index]
            except Exception:
              pass
          else:
            recent = filtered.iloc[0:0]
          _log_cj("date_window_applied", {"months_back": months_back, "before": int(len(filtered)), "after": int(len(recent))})
          # Invalidation: if same location has a non-canceled job within last inv_days days
          invalidated = 0
          if not recent.empty and loc_col in df.columns and inv_date_col in df.columns and status_col in df.columns:
            df_status_norm = df[status_col].astype(str).str.strip().str.lower()
            df_dates = pd.to_datetime(df[inv_date_col], errors="coerce")
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=inv_days)
            non_canceled_recent = (df_status_norm != canceled_val.strip().lower()) & (df_dates >= cutoff)
            # map location -> has_non_canceled_recent
            has_recent_map = df.groupby(df[loc_col].astype(str))[[inv_date_col]].apply(lambda g: bool(non_canceled_recent.loc[g.index].any()))
            has_recent_map = has_recent_map.rename("has_non_canceled_recent")
            recent_loc = recent[loc_col].astype(str)
            flag = recent_loc.map(has_recent_map)
            flag = flag.fillna(False)
            before_inv = int(len(recent))
            recent = recent[~flag]
            invalidated = before_inv - int(len(recent))
          _log_cj("invalidation_applied", {"removed_due_to_recent_non_canceled": int(invalidated), "after": int(len(recent))})
          # Apply processing limit
          limit = int(cfg.get("processing_limit", 0) or 0)
          if limit > 0 and not recent.empty:
            recent = recent.head(limit)
            _log_cj("limit_applied", {"limit": limit, "after": int(len(recent))})
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
            # Additional invalidation: summary contains substring
            sum_col = (cfg.get("summary_column") or "").strip()
            sum_contains = (cfg.get("summary_invalidate_contains") or "").strip().lower()
            if sum_col and sum_col in recent.columns and sum_contains:
              before_sum = int(len(events))
              mask_sum = ~recent[sum_col].astype(str).str.lower().str.contains(sum_contains, na=False)
              # align mask to events index
              mask_sum = mask_sum.loc[events.index]
              events = events.loc[mask_sum]
              _log_cj("summary_invalidation_applied", {"removed": int(before_sum - int(len(events))), "after": int(len(events))})
            # Deduplication (last step before save)
            if bool(cfg.get("dedup_enabled", False)):
              cust_col = cfg.get("dedup_customer_column", "Customer ID")
              loc_col_dedup = cfg.get("dedup_location_column", "Location ID")
              # Use only rows that survived to events
              if not events.empty and cust_col in recent.columns and loc_col_dedup in recent.columns and date_col in recent.columns:
                current_rows = recent.loc[events.index]
                rd = pd.to_datetime(current_rows[date_col], errors="coerce")
                dedup_key = current_rows[cust_col].astype(str) + "||" + current_rows[loc_col_dedup].astype(str)
                order_df = pd.DataFrame({
                  "key": dedup_key,
                  "comp_date": rd,
                }, index=current_rows.index)
                order_df = order_df.sort_values(["key", "comp_date"], ascending=[True, False])
                keep_labels = order_df.groupby("key").head(1).index.tolist()
                before_len = int(len(events))
                # ensure safe selection only using intersection
                keep_labels = list(events.index.intersection(keep_labels))
                events = events.loc[keep_labels]
                _log_cj("dedup_applied", {"kept": int(len(events)), "removed": int(before_len - int(len(events)))})
            # Determine output path and handle conflicts
            desired_name = cfg.get("output_filename") or "canceled_jobs_master.parquet"
            conflict_action = cfg.get("conflict_action", "Overwrite")
            new_name_cfg = cfg.get("new_filename")
            backup_before = bool(cfg.get("backup_before_overwrite", True))
            out = base_default_cj / desired_name
            if out.exists():
              if conflict_action == "Change name":
                alt_name = new_name_cfg or (out.with_stem(out.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name)
                out = base_default_cj / alt_name
              elif conflict_action == "Overwrite":
                if backup_before:
                  try:
                    backup_path = out.with_name(out.stem + f"_backup_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.parquet")
                    out.replace(backup_path)
                    st.info(f"Backup created: {backup_path.name}")
                  except Exception:
                    pass
              _log_cj("output_decision", {"exists": True, "action": conflict_action, "target": str(out)})
            else:
              _log_cj("output_decision", {"exists": False, "target": str(out)})
            # Reorder columns per preference
            order = [c for c in (cfg.get("output_order") or []) if c in events.columns]
            if order:
              events = events[[c for c in order if c in events.columns]]
            events.to_parquet(out, index=False)
            written.append(out.name)
            _log_cj("write_complete", {"rows_written": int(len(events)), "output_file": str(out)})
            st.success(f"Canceled jobs events: {len(events)}")
            st.dataframe(events.head(100), width='stretch')
            ui_log(f"[canceled_jobs] Wrote output {out.name} rows={len(events)}")
          else:
            st.info("No canceled jobs found.")
        except Exception as e:
          st.error(f"Canceled jobs scan error: {e}")
        finally:
          events_completed += 1
          update_progress(f"Completed canceled_jobs")
      elif et == "unsold_estimates":
        try:
          update_progress("Running unsold_estimates")
          src = parquet_dir / cfg.get("file", "Estimates.parquet")
          if not src.exists():
            st.error(f"Source file not found: {src}")
            continue
          df = pd.read_parquet(src, engine="pyarrow")
          # logs
          ue_log_dir = logs_root / "historical" / "unsold_estimates"
          ue_log_dir.mkdir(parents=True, exist_ok=True)
          ue_log_path = ue_log_dir / "scan.jsonl"
          def _log_ue(action: str, details: Dict[str, Any]) -> None:
            try:
              with open(ue_log_path, "a", encoding="utf-8") as f:
                rec = {"ts": datetime.now(UTC).isoformat(), "company": company, "action": action}
                rec.update(details or {})
                f.write(json.dumps(rec) + "\n")
            except Exception:
              pass
          _log_ue("scan_start", {
            "event_type": "unsold_estimates",
            "source_file": str(src),
            "rows_source": int(len(df)),
          })
          status_col = cfg.get("status_column", "Estimate Status")
          include_vals = [str(v) for v in (cfg.get("status_include_values") or ["Dismissed", "Open"])]
          creation_col = cfg.get("creation_date_column", "Creation Date")
          months_back = int(cfg.get("months_back", 24))
          opp_col = cfg.get("opportunity_status_column", "Opportunity Status")
          opp_exclude = str(cfg.get("opportunity_exclude_value", "Won"))
          id_col = cfg.get("id_column", "Estimate ID")
          loc_col = cfg.get("location_column", "Location ID")
          cust_col = cfg.get("customer_column", "Customer ID")
          est_comp_col = cfg.get("estimate_completion_date_column", "Completion Date")
          recent_est_days = int(cfg.get("recent_estimate_days", 30))
          sum_col = cfg.get("summary_column", "Estimate Summary")
          sum_contains = str(cfg.get("summary_invalidate_contains", "Test Estimate")).lower()
          jobs_created_col = cfg.get("job_created_date_column", "Created Date")
          jobs_sched_col = cfg.get("job_scheduled_date_column", "Scheduled Date")
          recent_job_days = int(cfg.get("recent_job_days", 21))
          # Filter 1: status in include
          mask_status = df[status_col].astype(str).isin(include_vals) if status_col in df.columns else pd.Series(False, index=df.index)
          f1 = df.loc[mask_status].copy()
          _log_ue("status_filter_applied", {"before": int(len(df)), "after": int(len(f1))})
          ui_log(f"[unsold_estimates] Status filter: before={len(df)} after={len(f1)}")
          # Filter 2: creation within months_back
          if not f1.empty and creation_col in f1.columns:
            dt = pd.to_datetime(f1[creation_col], errors="coerce")
            now = pd.Timestamp.utcnow().tz_localize(None)
            months_since = ((now.year - dt.dt.year) * 12 + (now.month - dt.dt.month)).astype("Int64")
            f2 = f1.loc[months_since <= months_back].copy()
          else:
            f2 = f1.iloc[0:0]
          _log_ue("creation_window_applied", {"months_back": months_back, "before": int(len(f1)), "after": int(len(f2))})
          ui_log(f"[unsold_estimates] Creation window months_back={months_back} after={len(f2)}")
          # Filter 3: opportunity != exclude
          if not f2.empty and opp_col in f2.columns:
            mask_opp = f2[opp_col].astype(str).str.strip().str.lower() != opp_exclude.strip().lower()
            f3 = f2.loc[mask_opp].copy()
          else:
            f3 = f2
          _log_ue("opportunity_filter_applied", {"exclude": opp_exclude, "before": int(len(f2)), "after": int(len(f3))})
          ui_log(f"[unsold_estimates] Opportunity exclude='{opp_exclude}' after={len(f3)}")
          # Invalidate by summary contains
          if not f3.empty and sum_col in f3.columns and sum_contains:
            before_sum = int(len(f3))
            f3 = f3.loc[~f3[sum_col].astype(str).str.lower().str.contains(sum_contains, na=False)].copy()
            _log_ue("summary_invalidation_applied", {"removed": int(before_sum - int(len(f3))), "after": int(len(f3))})
            ui_log(f"[unsold_estimates] Summary invalidation removed={before_sum - int(len(f3))}")
          # Recent estimate activity by location within window
          f_est = f3
          if not f_est.empty and loc_col in df.columns and est_comp_col in df.columns:
            all_dates = pd.to_datetime(df[est_comp_col], errors="coerce")
            by_loc_recent = {}
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=recent_est_days)
            for loc, idxs in df.groupby(df[loc_col].astype(str)).groups.items():
              has_recent = bool((all_dates.loc[list(idxs)] >= cutoff).any())
              by_loc_recent[str(loc)] = has_recent
            mask_recent_est = ~f_est[loc_col].astype(str).map(by_loc_recent).fillna(False)
            before_est = int(len(f_est))
            f_est = f_est.loc[mask_recent_est].copy()
            _log_ue("recent_estimate_invalidation_applied", {"removed": int(before_est - int(len(f_est))), "after": int(len(f_est))})
          # Recent job activity by customer within ±days window
          f_jobs = f_est
          if not f_jobs.empty and cust_col in f_jobs.columns:
            jobs_src = parquet_dir / "Jobs.parquet"
            if jobs_src.exists():
              jdf = pd.read_parquet(jobs_src, engine="pyarrow")
              created = pd.to_datetime(jdf.get(jobs_created_col), errors="coerce") if jobs_created_col in jdf.columns else None
              scheduled = pd.to_datetime(jdf.get(jobs_sched_col), errors="coerce") if jobs_sched_col in jdf.columns else None
              est_dt = pd.to_datetime(f_jobs[creation_col], errors="coerce") if creation_col in f_jobs.columns else None
              before_jobs = int(len(f_jobs))
              if created is not None or scheduled is not None:
                valid_idx = []
                for idx, row in f_jobs.iterrows():
                  cid = str(row.get(cust_col, ""))
                  if not cid:
                    valid_idx.append(idx)
                    continue
                  jmask = jdf[cust_col].astype(str) == cid if cust_col in jdf.columns else pd.Series(False, index=jdf.index)
                  j_idxs = jdf.index[jmask]
                  e_date = est_dt.loc[idx] if est_dt is not None else None
                  if pd.isna(e_date):
                    valid_idx.append(idx)
                    continue
                  low = e_date - pd.Timedelta(days=recent_job_days)
                  high = e_date + pd.Timedelta(days=recent_job_days)
                  win_hit = False
                  if created is not None:
                    hit = ((created.loc[j_idxs] >= low) & (created.loc[j_idxs] <= high)).any()
                    win_hit = win_hit or bool(hit)
                  if scheduled is not None and not win_hit:
                    hit = ((scheduled.loc[j_idxs] >= low) & (scheduled.loc[j_idxs] <= high)).any()
                    win_hit = win_hit or bool(hit)
                  if not win_hit:
                    valid_idx.append(idx)
                f_jobs = f_jobs.loc[valid_idx].copy()
                _log_ue("recent_job_invalidation_applied", {"removed": int(before_jobs - int(len(f_jobs))), "after": int(len(f_jobs))})
          # Dedup
          f_final = f_jobs
          if not f_final.empty and bool(cfg.get("dedup_enabled", False)):
            ccol = cfg.get("dedup_customer_column", "Customer ID")
            lcol = cfg.get("dedup_location_column", "Location ID")
            if ccol in f_final.columns and lcol in f_final.columns and creation_col in f_final.columns:
              cd = pd.to_datetime(f_final[creation_col], errors="coerce")
              key = f_final[ccol].astype(str) + "||" + f_final[lcol].astype(str)
              ord_df = pd.DataFrame({"key": key, "cdate": cd}, index=f_final.index).sort_values(["key", "cdate"], ascending=[True, False])
              keep = ord_df.groupby("key").head(1).index
              f_final = f_final.loc[keep].copy()
              _log_ue("dedup_applied", {"kept": int(len(f_final))})
          # Limit
          limit = int(cfg.get("processing_limit", 0) or 0)
          if limit > 0 and not f_final.empty:
            f_final = f_final.head(limit).copy()
            _log_ue("limit_applied", {"limit": limit, "after": int(len(f_final))})
          # Build events
          if not f_final.empty:
            payload_cols = [c for c in (cfg.get("payload_columns") or []) if c in f_final.columns]
            base_cols = {
              "event_type": "unsold_estimates",
              "entity_type": "estimate",
              "entity_id": f_final[id_col].astype(str) if id_col in f_final.columns else f_final.index.astype(str),
              "detected_at": datetime.now(UTC).isoformat(),
              "estimate_status": f_final[status_col].astype(str) if status_col in f_final.columns else "",
              "creation_date": f_final[creation_col].astype(str) if creation_col in f_final.columns else "",
              "opportunity_status": f_final[opp_col].astype(str) if opp_col in f_final.columns else "",
              "location_id": f_final[loc_col].astype(str) if loc_col in f_final.columns else "",
            }
            events = pd.DataFrame(base_cols)
            for col in payload_cols:
              events[col] = f_final[col].astype(str)
            # Output path
            desired_name = cfg.get("output_filename") or "unsold_estimates_master.parquet"
            conflict_action = cfg.get("conflict_action", "Overwrite")
            new_name_cfg = cfg.get("new_filename")
            backup_before = bool(cfg.get("backup_before_overwrite", True))
            cfg_out_dir_ue = (cfg.get("output_dir") or "").strip()
            out_base = (_P(cfg_out_dir_ue) if cfg_out_dir_ue else events_master_default_dir)
            out = out_base / desired_name
            if out.exists():
              if conflict_action == "Change name":
                alt_name = new_name_cfg or (out.with_stem(out.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name)
                out = out_base / alt_name
              elif conflict_action == "Overwrite":
                if backup_before:
                  try:
                    backup_path = out.with_name(out.stem + f"_backup_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.parquet")
                    out.replace(backup_path)
                    st.info(f"Backup created: {backup_path.name}")
                  except Exception:
                    pass
              _log_ue("output_decision", {"exists": True, "action": conflict_action, "target": str(out)})
            else:
              _log_ue("output_decision", {"exists": False, "target": str(out)})
            # Reorder per user order
            order = [c for c in (cfg.get("output_order") or []) if c in events.columns]
            if order:
              events = events[[c for c in order if c in events.columns]]
            events.to_parquet(out, index=False)
            written.append(out.name)
            _log_ue("write_complete", {"rows_written": int(len(events)), "output_file": str(out)})
            st.success(f"Unsold estimates events: {len(events)}")
            st.dataframe(events.head(100), width='stretch')
            ui_log(f"[unsold_estimates] Wrote output {out.name} rows={len(events)}")
          else:
            st.info("No unsold estimates found.")
        except Exception as e:
          st.error(f"Unsold estimates scan error: {e}")
        finally:
          events_completed += 1
          update_progress(f"Completed unsold_estimates")
      elif et == "aging_systems":
        try:
          update_progress("Running aging_systems")
          loc_src = parquet_dir / cfg.get("locations_file", "Locations.parquet")
          jobs_src = parquet_dir / cfg.get("jobs_file", "Jobs.parquet")
          # logging setup for aging_systems
          as_log_dir = logs_root / "historical" / "aging_systems"
          as_log_dir.mkdir(parents=True, exist_ok=True)
          as_log_path = as_log_dir / "scan.jsonl"
          def _log_as(action: str, details: Dict[str, Any]) -> None:
            try:
              with open(as_log_path, "a", encoding="utf-8") as f:
                rec = {"ts": datetime.now(UTC).isoformat(), "company": company, "action": action}
                rec.update(details or {})
                f.write(json.dumps(rec) + "\n")
            except Exception:
              pass
          if not loc_src.exists() or not jobs_src.exists():
            if not loc_src.exists():
              st.error(f"Locations source not found: {loc_src}")
            if not jobs_src.exists():
              st.error(f"Jobs source not found: {jobs_src}")
            continue
          ldf = pd.read_parquet(loc_src, engine="pyarrow")
          jdf = pd.read_parquet(jobs_src, engine="pyarrow")
          loc_col = cfg.get("location_id_column", "Location ID")
          payload_fields = [c for c in (cfg.get("job_payload_fields") or []) if c in (jdf.columns if not jdf.empty else [])]
          _log_as("scan_start", {
            "event_type": "aging_systems",
            "locations_file": str(loc_src),
            "jobs_file": str(jobs_src),
            "rows_locations": int(len(ldf)),
            "rows_jobs": int(len(jdf)),
            "payload_fields": payload_fields,
            "processing_limit": int(cfg.get("processing_limit", 0) or 0)
          })
          # build mapping location -> jobs payload list
          if loc_col not in ldf.columns or loc_col not in jdf.columns:
            st.error("Location ID column not found in one of the source files.")
            continue
          # prebuild jobs by location
          jobs_groups = jdf.groupby(jdf[loc_col].astype(str))
          out_rows = []
          # apply max-records limit on number of locations processed
          loc_ids_all = ldf[loc_col].astype(str).dropna().unique().tolist()
          limit = int(cfg.get("processing_limit", 0) or 0)
          if limit > 0:
            _log_as("limit_applied", {"limit": limit, "before": int(len(loc_ids_all)), "after": int(min(limit, len(loc_ids_all)))})
            loc_ids_all = loc_ids_all[:limit]
          for idx_loc, loc_id in enumerate(loc_ids_all, start=1):
            try:
              idxs = jobs_groups.groups.get(loc_id)
              if idxs is None or len(idxs) == 0:
                out_rows.append({"location_id": loc_id, "jobs": "[]"})
                continue
              subset = jdf.loc[list(idxs)]
              pl = []
              for _, r in subset.iterrows():
                item_raw = {k: (str(r.get(k)) if r.get(k) is not None and not pd.isna(r.get(k)) else "") for k in payload_fields}
                # apply mapping
                mapped = {}
                fmap = cfg.get("payload_field_mapping") or {}
                for k, v in item_raw.items():
                  mapped[fmap.get(k, k)] = v
                item = mapped
                pl.append(item)
              out_rows.append({"location_id": loc_id, "jobs": json.dumps(pl, ensure_ascii=False)})
              if idx_loc % 200 == 0:
                ui_log(f"[aging_systems] Built jobs for {idx_loc}/{len(loc_ids_all)} locations")
                status_text.text(f"Aging Systems: {idx_loc}/{len(loc_ids_all)} locations processed")
            except Exception:
              out_rows.append({"location_id": loc_id, "jobs": "[]"})
          events = pd.DataFrame(out_rows)
          # output path
          desired_name = cfg.get("output_filename") or "location_jobs_history.parquet"
          conflict_action = cfg.get("conflict_action", "Overwrite")
          new_name_cfg = cfg.get("new_filename")
          backup_before = bool(cfg.get("backup_before_overwrite", True))
          cfg_out_dir_as = (cfg.get("output_dir") or "").strip()
          out_base = (_P(cfg_out_dir_as) if cfg_out_dir_as else events_master_default_dir)
          out = out_base / desired_name
          if out.exists():
            if conflict_action == "Change name":
              alt_name = new_name_cfg or (out.with_stem(out.stem + "_" + datetime.now(UTC).strftime('%Y%m%d_%H%M%S')).name)
              out = out_base / alt_name
            elif conflict_action == "Overwrite":
              if backup_before:
                try:
                  backup_path = out.with_name(out.stem + f"_backup_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.parquet")
                  out.replace(backup_path)
                  st.info(f"Backup created: {backup_path.name}")
                except Exception:
                  pass
            _log_as("output_decision", {"exists": True, "action": conflict_action, "target": str(out)})
          else:
            _log_as("output_decision", {"exists": False, "target": str(out)})
          order = ["location_id", "jobs"]
          events = events[[c for c in order if c in events.columns]]
          events.to_parquet(out, index=False)
          written.append(out.name)
          _log_as("write_complete", {"rows_written": int(len(events)), "output_file": str(out)})
          st.success(f"Aging systems: wrote {len(events)} locations")
          st.dataframe(events.head(100), width='stretch')
          # Analysis step
          mode = (cfg.get("analysis_mode") or "basic").lower()
          hist_parquet = out
          if hist_parquet.exists():
            hist_df = pd.read_parquet(hist_parquet, engine="pyarrow")
            if mode == "basic":
              _log_as("analysis_start", {"mode": mode, "input_file": str(hist_parquet)})
              # compute basic system age (whole years)
              ages = []
              import math
              for _, row in hist_df.iterrows():
                loc_id = row.get("location_id")
                try:
                  jobs_json = row.get("jobs") or "[]"
                  jobs = json.loads(jobs_json)
                except Exception:
                  jobs = []
                years_candidates: list[int] = []
                now = pd.Timestamp.utcnow().tz_localize(None)
                for j in jobs:
                  jtype = str(j.get("Job Type", ""))
                  jstatus = str(j.get("Job Status", j.get("Status", "")))
                  if ("install" in jtype.lower()) and (jstatus.strip().lower() == "completed"):
                    cdate = j.get("Job Completion Date", j.get("Completion Date"))
                    try:
                      dt = pd.to_datetime(cdate, errors="coerce")
                      if not pd.isna(dt):
                        # full-year difference: subtract one if anniversary hasn't occurred yet this year
                        years = (now.year - dt.year)
                        if (now.month, now.day) < (dt.month, dt.day):
                          years -= 1
                        years = max(0, int(years))
                        years_candidates.append(years)
                    except Exception:
                      pass
                if years_candidates:
                  age_years = int(min(years_candidates))
                else:
                  age_years = None
                ages.append({"location_id": loc_id, "system_age_years": age_years})
              age_df = pd.DataFrame(ages)
              # write analysis output
              out_base = (_P(cfg.get("output_dir") or "") or events_master_default_dir)
              if not isinstance(out_base, _P):
                out_base = events_master_default_dir
              out_path = (_P(cfg.get("output_dir")) if cfg.get("output_dir") else events_master_default_dir) / (cfg.get("analysis_output_filename") or "aging_systems_basic.parquet")
              out_path.parent.mkdir(parents=True, exist_ok=True)
              age_df.to_parquet(out_path, index=False)
              written.append(out_path.name)
              _log_as("analysis_write_complete", {"rows_written": int(len(age_df)), "output_file": str(out_path)})
              st.success(f"Aging systems (basic analysis): wrote {len(age_df)} rows")
              st.dataframe(age_df.head(100), width='stretch')
            else:
              # Build per-location ordered job histories (in-memory) and write permit history parquet only
              _log_as("analysis_start", {"mode": mode, "input_file": str(hist_parquet)})
              # Use events_master_default_dir consistently for aging systems outputs
              output_base_dir = events_master_default_dir
              out_json = output_base_dir / (cfg.get("llm_output_filename") or "aging_systems_llm.json")
              records_map: Dict[str, List[Dict[str, Any]]] = {}
              for i_hist, row in enumerate(hist_df.iterrows(), start=1):
                _, row = row
                loc_id = row.get("location_id")
                try:
                  jobs = json.loads(row.get("jobs") or "[]")
                except Exception:
                  jobs = []
                # normalize and order fields per spec
                ordered = []
                for j in jobs:
                  created = j.get("Job Created Date", j.get("Created Date"))
                  completed = j.get("Job Completion Date", j.get("Completion Date"))
                  try:
                    sort_key = pd.to_datetime(completed, errors="coerce")
                  except Exception:
                    sort_key = pd.NaT
                  ordered.append({
                    "Job Created Date": (str(created) if created is not None and not pd.isna(created) else ""),
                    "Job Completion Date": (str(completed) if completed is not None and not pd.isna(completed) else ""),
                    "Customer ID": str(j.get("Customer ID") or ""),
                    "Job Status": str(j.get("Job Status", j.get("Status")) or ""),
                    "Job Type": str(j.get("Job Type") or ""),
                    "Job Notes": str(j.get("Job Notes", j.get("Summary")) or ""),
                    "_sort": sort_key
                  })
                # sort by completion date oldest->newest
                dfj = pd.DataFrame(ordered)
                if not dfj.empty:
                  dfj = dfj.sort_values(by=["_sort"], ascending=[True])
                if "_sort" in dfj.columns:
                  dfj = dfj.drop(columns=["_sort"])
                records_map[str(loc_id)] = dfj.to_dict(orient="records") if not dfj.empty else []
                if i_hist % 200 == 0:
                  ui_log(f"[aging_systems] Prepared LLM jobs for {i_hist}/{len(hist_df)} locations")
                  status_text.text(f"Aging Systems (LLM): {i_hist}/{len(hist_df)} locations prepared")
              # Skip writing LLM JSON (deprecated); we will only produce permit history parquet
              # Enrich with permits and also save a separate permit history parquet
              ui_log(f"[aging_systems] Starting permit enrichment phase")
              try:
                # load permits
                permit_path = _P(cfg.get("permit_data_path") or "global_data/permits/permit_data.parquet")
                if permit_path.exists():
                  pdf = pd.read_parquet(permit_path, engine="pyarrow")
                else:
                  pdf = pd.DataFrame()
                # load locations for address
                loc_src2 = parquet_dir / cfg.get("locations_file", "Locations.parquet")
                ldf2 = pd.read_parquet(loc_src2, engine="pyarrow") if loc_src2.exists() else pd.DataFrame()
                addr_by_loc = {}
                loc_id_col = cfg.get("location_id_column", "Location ID")
                addr_col = "Customer Address"
                if not ldf2.empty and loc_id_col in ldf2.columns and addr_col in ldf2.columns:
                  addr_by_loc = {str(r[loc_id_col]): str(r[addr_col]) for _, r in ldf2[[loc_id_col, addr_col]].dropna().iterrows()}
                _log_as("llm_sources_loaded", {
                  "permit_path": str(permit_path),
                  "permit_exists": bool(permit_path.exists()),
                  "permit_rows": int(len(pdf)) if not pdf.empty else 0,
                  "permit_columns": (list(map(str, pdf.columns)) if not pdf.empty else []),
                  "locations_path": str(loc_src2),
                  "locations_exists": bool(loc_src2.exists()),
                  "locations_rows": int(len(ldf2)) if not ldf2.empty else 0,
                })
                # simple address normalization
                def _clean_addr(a: str) -> str:
                  a = (a or "").upper().strip()
                  a = a.replace(" USA", "").replace(",", " ")
                  a = re.sub(r"\s+", " ", a)
                  return a
                def _norm_addr(a: str) -> str:
                  try:
                    import usaddress
                    parts = dict(usaddress.parse(a or ""))
                    num = parts.get("AddressNumber", "").strip()
                    street = (parts.get("StreetName", "") + " " + parts.get("StreetNamePostType", "")).strip()
                    city = parts.get("PlaceName", "").strip()
                    state = parts.get("StateName", "").strip()
                    zipc = parts.get("ZipCode", "").strip()
                    core = " ".join([num, street]).strip()
                    tail = " ".join([city, state, zipc]).strip()
                    out = (core + " " + tail).strip()
                    return _clean_addr(out)
                  except Exception:
                    return _clean_addr(a)
                # enhanced legacy fuzzy matching
                from datahound.events.fuzzy_matching import (
                  build_enhanced_permit_index,
                  enhanced_address_match,
                  prepare_location_addresses_enhanced
                )
                max_edits_cfg = int(cfg.get("permit_max_edit_distance", 2))
                pmap = cfg.get("permit_field_mapping") or {}
                desired_fields = cfg.get("permit_fields") or []
                enriched = {}
                # Build index once
                exact_map, exact_count_map, name_idx, init_idx, all_idx = build_enhanced_permit_index(pdf, max_edits=max_edits_cfg)
                # Prepare location addresses (normalized)
                loc_df_small = ldf2[[loc_id_col, addr_col]].rename(columns={loc_id_col: "Location ID", addr_col: "Customer Address"}) if not ldf2.empty else pd.DataFrame()
                loc_addr_map = prepare_location_addresses_enhanced(loc_df_small)
                # Build quick lookup of permits by Permit Num for full-field join
                perm_col_map = {c.lower(): c for c in (list(pdf.columns) if not pdf.empty else [])}
                pn_col = None
                for cand in ["permit num", "permit_id", "permit number", "project id"]:
                  if cand in perm_col_map:
                    pn_col = perm_col_map[cand]
                    break
                if not pn_col:
                  _log_as("llm_join_missing_col", {"reason": "permit_num_column_not_found"})
                # dedicated debug log for matching
                match_debug_path = as_log_dir / "matching_debug.jsonl"
                def _dbg(tag: str, payload: Dict[str, Any]) -> None:
                  try:
                    with open(match_debug_path, "a", encoding="utf-8") as f:
                      rec = {"ts": datetime.now(UTC).isoformat(), "company": company, "tag": tag}
                      rec.update(payload or {})
                      f.write(json.dumps(rec) + "\n")
                  except Exception:
                    pass
                for i_loc, (loc_id, jobs) in enumerate(records_map.items(), start=1):
                  raw_address = addr_by_loc.get(str(loc_id), "")
                  _log_as("llm_match_start", {"location_id": str(loc_id), "raw_address": raw_address})
                  def _tap(tag: str, payload: Dict[str, Any]) -> None:
                    _dbg(tag, {"location_id": str(loc_id), **payload})
                    if tag == "loc_norm":
                      # store prepared location address for later row output
                      try:
                        enriched.setdefault(loc_id, {}).setdefault("_match_meta", {}).setdefault("loc_norm", payload)
                      except Exception:
                        pass
                  result = enhanced_address_match(raw_address, exact_map, exact_count_map, name_idx, init_idx, all_idx, max_edits=max_edits_cfg, debug=True, debug_cb=_tap)
                  matches: List[Dict[str, Any]] = []
                  if result.get("match_type") != "NO_MATCH":
                    _log_as("llm_match_results", {"location_id": str(loc_id), "match_type": result.get("match_type"), "score": result.get("score", 0.0), "count": int(result.get("match_count", 0))})
                    # Join back to full permits by Permit Num to fill selected fields
                    added = 0
                    for s in result.get("matched_permits", [])[:200]:
                      permit_id = s.get("permit_id") or s.get("Permit Num") or s.get("permit_num")
                      if pn_col and permit_id:
                        rows = pdf[pdf[pn_col].astype(str) == str(permit_id)] if not pdf.empty else pd.DataFrame()
                        if not rows.empty:
                          for _, prow in rows.iterrows():
                            item = {}
                            for k in desired_fields:
                              if k in pdf.columns:
                                val = prow.get(k)
                                item[pmap.get(k, k)] = None if pd.isna(val) else str(val)
                              else:
                                item[pmap.get(k, k)] = ""
                            matches.append(item)
                            added += 1
                        else:
                          _log_as("llm_join_not_found", {"location_id": str(loc_id), "permit_id": str(permit_id)})
                      else:
                        _log_as("llm_join_skipped", {"location_id": str(loc_id), "reason": "missing_permit_id_or_column"})
                    _log_as("llm_join_added_count", {"location_id": str(loc_id), "added": int(added)})
                  # sort permits by Applied Date (mapped) before saving
                  applied_key = pmap.get("Applied Date", "Applied Date")
                  if matches:
                    try:
                      dfm = pd.DataFrame(matches)
                      if applied_key in dfm.columns:
                        dfm["_applied"] = pd.to_datetime(dfm[applied_key], errors="coerce")
                        dfm = dfm.sort_values(["_applied", applied_key], ascending=[True, True])
                        dfm = dfm.drop(columns=["_applied"]) if "_applied" in dfm.columns else dfm
                        matches = dfm.to_dict(orient="records")
                    except Exception:
                      try:
                        import pandas as pd_local  # Local import to avoid scope issues
                        def _pdate(x):
                          try:
                            return pd_local.to_datetime(x, errors="coerce")
                          except Exception:
                            return pd_local.NaT
                        matches = sorted(matches, key=lambda r: _pdate(r.get(applied_key)))
                      except Exception:
                        pass
                  if i_loc % 200 == 0:
                    ui_log(f"[aging_systems] Matched permits for {i_loc}/{len(records_map)} locations")
                    status_text.text(f"Aging Systems (permits): {i_loc}/{len(records_map)} locations processed")
                  # keep just best meta for row-level columns
                  prev_loc_norm = None
                  try:
                    prev_loc_norm = enriched.get(loc_id, {}).get("_match_meta", {}).get("loc_norm")
                  except Exception:
                    prev_loc_norm = None
                  enriched[loc_id] = {
                    "jobs": jobs,
                    "permits": matches,
                    "_match_meta": {
                      "raw_address": raw_address,
                      "best_address": result.get("best_address", ""),
                      "score": result.get("score"),
                      "distance": result.get("distance"),
                      "loc_norm": prev_loc_norm or {},
                      "best_norm": result.get("best_norm", ""),
                    }
                  }
                # keep jobs JSON as-is; write independent permit history parquet
                rows = []
                for loc_id, v in enriched.items():
                  info = v or {}
                  meta = info.get("_match_meta", {})
                  # compute prepared addresses
                  loc_norm = (meta.get("loc_norm", {}) or {})
                  loc_prep = str((loc_norm.get("normalized") or "")).lower()
                  best_norm = str(meta.get("best_norm", "")).lower()
                  perms_list = info.get("permits", []) or []
                  rows.append({
                    "location_id": loc_id,
                    "permits": json.dumps(perms_list, ensure_ascii=False),
                    "permits_found": int(len(perms_list)),
                    "location_address_raw": meta.get("raw_address", ""),
                    "location_address_prepared": loc_prep,
                    "permit_address_raw": meta.get("best_address", ""),
                    "permit_address_prepared": best_norm,
                    "edit_count": meta.get("distance", None),
                    "similarity_score": meta.get("score", None),
                  })
                permit_hist_name = cfg.get("permit_history_output_filename") or "location_permit_history.parquet"
                out_permit_parquet = output_base_dir / permit_hist_name
                pd.DataFrame(rows).to_parquet(out_permit_parquet, index=False)
                _log_as("llm_enrichment_complete", {"locations": int(len(rows)), "permit_parquet": str(out_permit_parquet)})
                ui_log(f"[aging_systems] Created permit history file: {out_permit_parquet.name}")
                st.caption(f"Permit history Parquet: `{out_permit_parquet.as_posix()}`")
              except Exception as ex:
                ui_log(f"[aging_systems] Permit enrichment failed: {str(ex)}")
                _log_as("llm_enrichment_error", {"error": str(ex)})
              # If both history parquet files exist, compose location_profile.parquet
              try:
                jobs_hist = out
                # Ensure permit parquet path is defined even if previous section failed
                permit_hist_name = cfg.get("permit_history_output_filename") or "location_permit_history.parquet"
                permits_hist = output_base_dir / permit_hist_name
                _log_as("profile_check", {
                  "jobs_hist_path": str(jobs_hist),
                  "permits_hist_path": str(permits_hist),
                  "jobs_hist_exists": bool(jobs_hist.exists()),
                  "permits_hist_exists": bool(permits_hist.exists())
                })
                ui_log(f"[aging_systems] Checking for profile generation: jobs={jobs_hist.exists()}, permits={permits_hist.exists()}")
                
                # Generate profiles if jobs history exists, with or without permits
                if jobs_hist.exists():
                  ui_log(f"[aging_systems] Jobs history exists - starting profile generation")
                  prof_rows = []
                  # Load jobs history (required)
                  jh = pd.read_parquet(jobs_hist, engine="pyarrow")
                  jobs_by_loc = {str(r["location_id"]): (json.loads(r["jobs"]) if isinstance(r.get("jobs"), str) else []) for _, r in jh.iterrows()}
                  
                  # Load permits history (optional)
                  permits_by_loc = {}
                  if permits_hist.exists():
                    ui_log(f"[aging_systems] Loading permit history")
                    ph = pd.read_parquet(permits_hist, engine="pyarrow")
                    permits_by_loc = {str(r["location_id"]): (json.loads(r["permits"]) if isinstance(r.get("permits"), str) else []) for _, r in ph.iterrows()}
                  else:
                    ui_log(f"[aging_systems] No permit history found - proceeding with jobs only")
                  all_locs = set(jobs_by_loc.keys()) | set(permits_by_loc.keys())
                  for loc in all_locs:
                    prof_rows.append({
                      "location_id": loc,
                      "job_history": json.dumps(jobs_by_loc.get(loc, []), ensure_ascii=False),
                      "permit_history": json.dumps(permits_by_loc.get(loc, []), ensure_ascii=False),
                    })
                  profile_name = cfg.get("profile_output_filename") or "location_profile.parquet"
                  out_profile = output_base_dir / profile_name
                  pd.DataFrame(prof_rows).to_parquet(out_profile, index=False)
                  _log_as("profile_write_complete", {"rows_written": int(len(prof_rows)), "profile_file": str(out_profile)})
                  st.caption(f"Location profile Parquet: `{out_profile.as_posix()}`")

                  # Emit YAML profiles per location under token budget
                  ui_log(f"[aging_systems] Starting YAML generation for {len(all_locs)} locations")
                  _log_as("yaml_generation_start", {"total_locations": len(all_locs)})
                  
                  def _to_iso(d: str) -> str:
                    try:
                      import pandas as pd_iso  # Local import to avoid scope issues
                      dt = pd_iso.to_datetime(d, errors="coerce")
                      if pd_iso.isna(dt):
                        return ""
                      return dt.strftime("%Y-%m-%d")
                    except Exception:
                      return ""
                  def _clean_text(s: str) -> str:
                    if s is None:
                      return ""
                    txt = str(s).strip()
                    txt = re.sub(r"\s+", " ", txt)
                    return txt
                  def estimate_tokens(s: str) -> int:
                    # rough estimate ~4 chars per token
                    return max(1, (len(s) + 3) // 4)
                  def build_yaml(loc_id: str, jobs: list, permits: list) -> str:
                    # normalize and sort
                    import pandas as pd_sort  # Local import for nested functions
                    def sort_jobs(jl: list) -> list:
                      def key_fn(j):
                        # Try multiple field name variations for completion date, then created date
                        completed = j.get("Job Completion Date") or j.get("completion_date") or j.get("Completion Date") or ""
                        created = j.get("Job Created Date") or j.get("created_date") or j.get("Created Date") or ""
                        # Use completion date first, fall back to created date
                        date_str = completed or created
                        date_val = pd_sort.to_datetime(date_str, errors="coerce") if date_str else pd_sort.NaT
                        # Handle NaT/None values by using a very early date for sorting
                        return date_val if pd_sort.notna(date_val) else pd_sort.Timestamp.min
                      return sorted(jl, key=key_fn)
                    def sort_permits(pl: list) -> list:
                      def key_fp(p):
                        date_val = pd_sort.to_datetime(p.get("Applied Date") or p.get("applied_date"), errors="coerce")
                        # Handle NaT/None values by using a very early date for sorting
                        return date_val if pd_sort.notna(date_val) else pd_sort.Timestamp.min
                      return sorted(pl, key=key_fp)
                    jobs_s = sort_jobs(jobs)
                    permits_s = sort_permits(permits)
                    # aggregates
                    total_jobs = len(jobs_s)
                    total_permits = len(permits_s)
                    first_job = _to_iso(jobs_s[0].get("Job Created Date") or jobs_s[0].get("created_date") or jobs_s[0].get("Created Date") if jobs_s else "")
                    last_job = _to_iso(jobs_s[-1].get("Job Completion Date") or jobs_s[-1].get("completion_date") or jobs_s[-1].get("Completion Date") if jobs_s else "")
                    first_permit = _to_iso(permits_s[0].get("Applied Date") if permits_s else "")
                    last_permit = _to_iso(permits_s[-1].get("Issued Date") if permits_s else "")
                    # header
                    lines = []
                    lines.append(f"location_id: {loc_id}")
                    lines.append("aggregates:")
                    lines.append(f"  total_jobs: {total_jobs}")
                    lines.append(f"  total_permits: {total_permits}")
                    lines.append(f"  first_job_date: {first_job}")
                    lines.append(f"  last_job_date: {last_job}")
                    lines.append(f"  first_permit_date: {first_permit}")
                    lines.append(f"  last_permit_date: {last_permit}")
                    # permits
                    lines.append("permit_history:")
                    for p in permits_s:
                      lines.append("  - permit_number: \"%s\"" % _clean_text(p.get("Permit Num") or p.get("permit_number") or ""))
                      lines.append("    type_desc: %s" % _clean_text(p.get("Permit Type Desc") or p.get("type_desc") or ""))
                      lines.append("    class_mapped: %s" % _clean_text(p.get("Permit Class Mapped") or p.get("class_mapped") or ""))
                      lines.append("    work_class: %s" % _clean_text(p.get("Work Class") or p.get("work_class") or ""))
                      lines.append("    description: \"%s\"" % _clean_text(p.get("Description") or p.get("description") or ""))
                      lines.append("    applied_date: %s" % _to_iso(p.get("Applied Date") or p.get("applied_date") or ""))
                      lines.append("    issued_date: %s" % _to_iso(p.get("Issued Date") or p.get("issued_date") or ""))
                      lines.append("    status: %s" % _clean_text(p.get("Status Current") or p.get("status") or ""))
                      lines.append("    contractor: \"%s\"" % _clean_text(p.get("Contractor Company Name") or p.get("contractor") or ""))
                      link = _clean_text(p.get("Link") or p.get("link") or "")
                      if link:
                        lines.append("    link: \"%s\"" % link)
                    # jobs
                    lines.append("job_history:")
                    for j in jobs_s:
                      # Handle multiple possible field name variations
                      job_type = _clean_text(j.get("Job Type") or j.get("job_type") or j.get("Type") or "")
                      status = _clean_text(j.get("Job Status") or j.get("status") or j.get("Status") or "")
                      created = _to_iso(j.get("Job Created Date") or j.get("created_date") or j.get("Created Date") or "")
                      completed = _to_iso(j.get("Job Completion Date") or j.get("completion_date") or j.get("Completion Date") or "")
                      customer = _clean_text(j.get("Customer ID") or j.get("customer_id") or "")
                      notes = _clean_text(j.get("Job Notes") or j.get("notes") or j.get("Summary") or "")
                      
                      lines.append("  - job_type: %s" % job_type)
                      lines.append("    status: %s" % status)
                      lines.append("    created_date: %s" % created)
                      lines.append("    completion_date: %s" % completed)
                      lines.append("    customer_id: \"%s\"" % customer)
                      if notes:
                        if len(notes) > 280:
                          notes = notes[:280] + "…"
                        lines.append("    notes: |")
                        # indent wrapped notes
                        for chunk in re.sub(r"\r?\n", " ", notes).split("\n"):
                          lines.append("      " + chunk)
                    return "\n".join(lines)

                  profiles_dir = output_base_dir / "profiles_yaml"
                  profiles_dir.mkdir(parents=True, exist_ok=True)
                  TOKEN_LIMIT = 32700
                  yaml_generated = 0
                  for i_loc, loc in enumerate(all_locs, 1):
                    try:
                      jobs = jobs_by_loc.get(loc, [])
                      permits = permits_by_loc.get(loc, [])
                      
                      # trim jobs until token budget is met
                      import pandas as pd_trim  # Local import for nested function
                      def sort_jobs_for_trim(jl: list) -> list:
                        def key_fn(j):
                          date_val = pd_trim.to_datetime(j.get("Job Completion Date") or j.get("Job Created Date"), errors="coerce")
                          # Handle NaT/None values by using a very early date for sorting
                          return date_val if pd_trim.notna(date_val) else pd_trim.Timestamp.min
                        return sorted(jl, key=key_fn)
                      jobs_trim = sort_jobs_for_trim(list(jobs))
                      try:
                        yaml_text = build_yaml(loc, jobs_trim, permits)
                      except Exception as yaml_err:
                        ui_log(f"[aging_systems] build_yaml failed for {loc}: {str(yaml_err)}")
                        continue
                      tok = estimate_tokens(yaml_text)
                      removed = 0
                      while tok > TOKEN_LIMIT and jobs_trim:
                        # drop oldest
                        jobs_trim = jobs_trim[1:]
                        removed += 1
                        try:
                          yaml_text = build_yaml(loc, jobs_trim, permits)
                          tok = estimate_tokens(yaml_text)
                        except Exception as trim_err:
                          ui_log(f"[aging_systems] build_yaml trim failed for {loc}: {str(trim_err)}")
                          break
                      
                      # write file
                      out_yaml = profiles_dir / f"{loc}.yaml"
                      with open(out_yaml, "w", encoding="utf-8") as f:
                        f.write(yaml_text)
                      _log_as("profile_yaml_write", {"location_id": loc, "tokens": int(tok), "jobs_removed": int(removed), "path": str(out_yaml)})
                      yaml_generated += 1
                      
                      # Progress update every 50 locations
                      if i_loc % 50 == 0:
                        ui_log(f"[aging_systems] Generated YAML for {i_loc}/{len(all_locs)} locations")
                        status_text.text(f"Aging Systems (YAML): {i_loc}/{len(all_locs)} profiles generated")
                        
                    except Exception as e:
                      ui_log(f"[aging_systems] Failed to generate YAML for location {loc}: {str(e)}")
                      _log_as("profile_yaml_error", {"location_id": loc, "error": str(e)})
                  
                  ui_log(f"[aging_systems] YAML generation complete: {yaml_generated}/{len(all_locs)} files created")

                  # Trigger LLM analysis if YAML profiles exist
                  profiles_dir = output_base_dir / "profiles_yaml"
                  if profiles_dir.exists():
                    yaml_files = list(profiles_dir.glob("*.yaml"))
                    if yaml_files:
                      ui_log(f"[aging_systems] Starting LLM analysis on {len(yaml_files)} YAML profiles...")
                      
                      # Set up LLM configuration
                      llm_cfg = LLMConfig(
                        api_key=None,
                        model=str(cfg.get("llm_model") or "deepseek-chat"),
                        max_tokens=int(cfg.get("llm_max_tokens", 8192)),
                        temperature=float(cfg.get("llm_temperature", 0.0)),
                        max_concurrent_requests=int(cfg.get("llm_max_concurrent_requests", 25)),
                      )
                      analyzer = LLMAnalyzer(llm_cfg)
                      
                      if analyzer.is_available():
                        # Prepare requests for all YAML files
                        now_iso = datetime.now(UTC).strftime('%Y-%m-%d')
                        system_prompt = build_aging_systems_system_prompt("McCullough Heating & Air", now_iso)
                        
                        requests = []
                        file_metadata = []
                        
                        for yaml_file in yaml_files:
                          try:
                            loc_id = yaml_file.stem
                            text = yaml_file.read_text(encoding="utf-8")
                            requests.append({
                              "id": loc_id,
                              "system_prompt": system_prompt,
                              "user_prompt": text
                            })
                            file_metadata.append({
                              "location_id": loc_id,
                              "yaml_path": yaml_file.as_posix()
                            })
                          except Exception:
                            continue
                        
                        if requests:
                          ui_log(f"[aging_systems] Processing {len(requests)} locations with LLM analysis...")
                          
                          # Process requests concurrently
                          results = analyzer.analyze_texts_concurrent(requests)
                          
                          # Process results and create aging_systems.parquet
                          parquet_data = []
                          successful_llm = 0
                          
                          for result, metadata in zip(results, file_metadata):
                            if result["success"] and result.get("result"):
                              llm_result = result["result"]
                              parsed_response, parse_status = _parse_llm_response(llm_result)
                              
                              if parsed_response and isinstance(parsed_response, dict):
                                company = parsed_response.get("company", "")
                                analysis_date = parsed_response.get("analysis_date", "")
                                answers = parsed_response.get("answers", {})
                                
                                parquet_row = {
                                  "location_id": metadata["location_id"],
                                  "processing_timestamp": datetime.now(UTC).isoformat(),
                                  "duration_ms": result["duration_ms"],
                                  "success": result["success"],
                                  "llm_response_raw": json.dumps(llm_result, ensure_ascii=False),
                                  "company": company,
                                  "analysis_date": analysis_date,
                                  "last_system_install_date": _extract_answer_value(answers.get("last_system_install_date")),
                                  "go_to_contractor": _extract_answer_value(answers.get("go_to_contractor")),
                                  "estimated_current_system_age_years": _extract_answer_value(answers.get("estimated_current_system_age_years")),
                                  "most_likely_next_job": _extract_answer_value(answers.get("most_likely_next_job")),
                                  "last_permit_issue_date": _extract_answer_value(answers.get("last_permit_issue_date")),
                                  "last_job_date": _extract_answer_value(answers.get("last_job_date")),
                                }
                                parquet_data.append(parquet_row)
                                successful_llm += 1
                          
                          # Save LLM results to configurable filename
                          if parquet_data:
                            llm_output_filename = cfg.get("llm_analysis_output_filename") or "aging_systems.parquet"
                            aging_systems_parquet = output_base_dir / llm_output_filename
                            df_llm_results = pd.DataFrame(parquet_data)
                            
                            # Add additional payload columns if configured
                            additional_columns = cfg.get("additional_payload_columns", [])
                            if additional_columns and not ldf.empty:
                              # Get location data for additional columns
                              loc_id_col = cfg.get("location_id_column", "Location ID")
                              if loc_id_col in ldf.columns:
                                location_data = ldf.set_index(ldf[loc_id_col].astype(str))
                                
                                # Add additional columns to each row
                                for idx, row in df_llm_results.iterrows():
                                  location_id = str(row.get("location_id", ""))
                                  if location_id in location_data.index:
                                    location_row = location_data.loc[location_id]
                                    for col in additional_columns:
                                      if col in location_row.index:
                                        df_llm_results.at[idx, col] = str(location_row[col]) if pd.notna(location_row[col]) else ""
                            
                            df_llm_results.to_parquet(aging_systems_parquet, index=False, engine="pyarrow")
                            written.append(aging_systems_parquet.name)
                            _log_as("llm_analysis_complete", {
                              "llm_results_file": str(aging_systems_parquet),
                              "successful_analyses": successful_llm,
                              "total_requests": len(requests),
                              "additional_columns_added": additional_columns
                            })
                            ui_log(f"[aging_systems] LLM analysis complete: {successful_llm}/{len(requests)} successful")
                            st.success(f"LLM Analysis: {successful_llm}/{len(requests)} locations analyzed")
                            st.dataframe(df_llm_results.head(100), width='stretch')
                          else:
                            ui_log(f"[aging_systems] No successful LLM results to save")
                      else:
                        ui_log(f"[aging_systems] LLM not available - skipping analysis")
                        _log_as("llm_analysis_skipped", {"reason": "llm_not_available"})
                    else:
                      ui_log(f"[aging_systems] No YAML profiles found - skipping LLM analysis")
                      _log_as("llm_analysis_skipped", {"reason": "no_yaml_profiles"})
                  else:
                      ui_log(f"[aging_systems] Profiles directory not found - skipping LLM analysis")
                      _log_as("llm_analysis_skipped", {"reason": "no_profiles_directory"})
                else:
                  ui_log(f"[aging_systems] Jobs history file missing - skipping profile generation")
                  _log_as("profile_skipped", {"jobs_hist_exists": bool(jobs_hist.exists()), "permits_hist_exists": bool(permits_hist.exists())})
              except Exception as ex:
                _log_as("profile_error", {"error": str(ex)})
        except Exception as e:
          st.error(f"Aging systems scan error: {e}")
        finally:
          events_completed += 1
          update_progress(f"Completed aging_systems")
      elif et == "lost_customers":
        try:
          update_progress("Running lost_customers")
          ui_log(f"[{et}] Starting lost customers analysis...")
          
          # Set up logging for lost_customers
          lc_log_dir = logs_root / "historical" / "lost_customers"
          lc_log_dir.mkdir(parents=True, exist_ok=True)
          lc_log_path = lc_log_dir / "scan.jsonl"
          
          def _log_lc(action: str, details: Dict[str, Any]) -> None:
            try:
              with open(lc_log_path, "a", encoding="utf-8") as f:
                rec = {"ts": datetime.now(UTC).isoformat(), "company": company, "action": action}
                rec.update(details or {})
                f.write(json.dumps(rec) + "\n")
            except Exception:
              pass
          
          # Load data files
          customers_src = parquet_dir / current[et].get("customers_file", "Customers.parquet")
          calls_src = parquet_dir / current[et].get("calls_file", "Calls.parquet")
          
          if not customers_src.exists():
            st.error(f"Customers source not found: {customers_src}")
            continue
          if not calls_src.exists():
            st.error(f"Calls source not found: {calls_src}")
            continue
            
          customers_df = pd.read_parquet(customers_src, engine="pyarrow")
          calls_df = pd.read_parquet(calls_src, engine="pyarrow")
          
          ui_log(f"[{et}] Loaded customers: {len(customers_df)} rows")
          ui_log(f"[{et}] Loaded calls: {len(calls_df)} rows")
          
          # Get processing configuration
          processing_limit = int(current[et].get("processing_limit", 0) or 0)
          
          _log_lc("scan_start", {
            "event_type": "lost_customers",
            "customers_file": str(customers_src),
            "calls_file": str(calls_src),
            "rows_customers": int(len(customers_df)),
            "rows_calls": int(len(calls_df)),
            "processing_limit": processing_limit,
            "customer_id_column": current[et].get("customer_id_column", "Customer ID"),
            "address_column": current[et].get("customer_address_column", "Full Address"),
            "call_date_column": current[et].get("call_date_column", "Call Date"),
            "company_names": current[et].get("company_names", [])
          })
          
          # Create intermediate output directory for memory management
          intermediate_dir = events_master_default_dir / "lost_customers_intermediate"
          intermediate_dir.mkdir(parents=True, exist_ok=True)
          
          ui_log(f"[{et}] Processing customers with limit: {processing_limit if processing_limit > 0 else 'unlimited'}")
          
          # Step 1: Extract customer basic data (ID + Address)
          ui_log(f"[{et}] Step 1: Extracting customer addresses...")
          customer_addresses = _extract_customer_addresses(
            customers_df, current[et], processing_limit, ui_log, _log_lc
          )
          
          # Save customer addresses to intermediate parquet
          addresses_file = intermediate_dir / "customer_addresses.parquet"
          customer_addresses.to_parquet(addresses_file, index=False)
          ui_log(f"[{et}] Saved {len(customer_addresses)} customer addresses to {addresses_file.name}")
          
          # Clear customers memory
          del customers_df
          
          # Step 2: Extract call data for each customer
          ui_log(f"[{et}] Step 2: Processing call history...")
          customer_calls = _extract_customer_calls(
            calls_df, addresses_file, current[et], ui_log, _log_lc
          )
          
          # Save customer calls to intermediate parquet
          calls_file = intermediate_dir / "customer_calls.parquet"
          customer_calls.to_parquet(calls_file, index=False)
          ui_log(f"[{et}] Saved {len(customer_calls)} customer call records to {calls_file.name}")
          
          # Clear calls memory
          del calls_df
          
          # Step 3: Process permit matching
          ui_log(f"[{et}] Step 3: Matching addresses to permits...")
          permit_data_path = _P("global_data/permits/permit_data.csv")
          if not permit_data_path.exists():
            st.error(f"Permit data not found: {permit_data_path}")
            continue
            
          permit_matches = _process_permit_matching_pipeline(
            calls_file, permit_data_path, current[et], ui_log, _log_lc
          )
          
          # Save permit matches to intermediate parquet
          matches_file = intermediate_dir / "permit_matches.parquet"
          permit_matches.to_parquet(matches_file, index=False)
          ui_log(f"[{et}] Saved {len(permit_matches)} permit matches to {matches_file.name}")
          
          # Step 4: Analyze lost customers
          ui_log(f"[{et}] Step 4: Analyzing lost customer patterns...")
          events_df = _analyze_lost_customers_pipeline(
            matches_file, current[et], ui_log, _log_lc
          )
          
          # Use the events_df as the final result
          events = events_df
          
          ui_log(f"[{et}] Analysis complete. Found {len(events)} lost customers")
          _log_lc("analysis_complete", {
            "events_found": len(events),
            "intermediate_files_created": 3
          })
          
          # Save output file
          desired_name = current[et].get("output_filename") or "lost_customers_analysis.parquet"
          cfg_out_dir_lc = (current[et].get("output_dir") or "").strip()
          out_base = (_P(cfg_out_dir_lc) if cfg_out_dir_lc else events_master_default_dir)
          out = out_base / desired_name
          
          out_base.mkdir(parents=True, exist_ok=True)
          events.to_parquet(out, index=False)
          written.append(out.name)
          
          _log_lc("write_complete", {
            "rows_written": int(len(events)),
            "output_file": str(out),
            "lost_customers_found": len(events)
          })
          
          ui_log(f"[{et}] Wrote {len(events)} lost customer records to {out.name}")
          
          # Get processing stats from the intermediate files for display
          contacts_file = intermediate_dir / "customer_calls.parquet"
          if contacts_file.exists():
            contacts_processed = len(pd.read_parquet(contacts_file))
          else:
            contacts_processed = 0
          
          st.success(f"Lost customers: analyzed {contacts_processed} customers, found {len(events)} lost customers")
          
          if len(events) > 0:
            st.dataframe(events.head(100), width='stretch')
          else:
            st.info("No lost customers found based on current criteria.")
        
        except Exception as e:
          ui_log(f"[{et}] ERROR: {str(e)}")
          st.error(f"Lost customers scan failed: {str(e)}")
          _log_lc("scan_error", {"error": str(e)})
        finally:
          events_completed += 1
          update_progress(f"Completed lost_customers")
      elif persist_on_run:
        base_dir = events_master_default_dir
        out = base_dir / f"{et}_master.parquet"
        cols = ["event_type", "entity_type", "entity_id", "detected_at", "details"]
        df = pd.DataFrame({c: pd.Series(dtype="string") for c in cols})
        if include_sample:
          df = pd.DataFrame([{ "event_type": et, "entity_type": "", "entity_id": "", "detected_at": datetime.now(UTC).isoformat(), "details": "" }])
        df.to_parquet(out, index=False)
        written.append(out.name)
    st.success("Scans completed")
    if written:
      st.info(f"Saved as: {written}")

  # LLM Settings Tab
  with tab_llm:
    st.subheader("LLM Settings")
    # Aging Systems - full LLM UI
    with st.expander("Aging Systems", expanded=True):
      cfg_as = current.get("aging_systems", {})
      base_out_as = (_P(cfg_as.get("output_dir")) if cfg_as.get("output_dir") else events_master_default_dir)
      profiles_dir_as = base_out_as / "profiles_yaml"
      st.caption(f"Profiles directory: `{profiles_dir_as.as_posix()}`")
      yaml_exists_as = profiles_dir_as.exists() and any(profiles_dir_as.glob("*.yaml"))
      now_iso_global = datetime.now(UTC).strftime('%Y-%m-%d')
      built_prompt_global = build_aging_systems_system_prompt("McCullough Heating & Air", now_iso_global)
      show_prompt = st.checkbox("Show system prompt (built)", value=False, key="as_show_prompt")
      if show_prompt:
        st.code(built_prompt_global, language="markdown")
      
      st.markdown("**LLM Configuration**")
      col_llm1, col_llm2 = st.columns(2)
      with col_llm1:
        max_tokens = st.number_input(
          "Max tokens per response",
          min_value=1000,
          max_value=8192,
          value=min(int(cfg_as.get("llm_max_tokens", 8192)), 8192),
          key="llm_max_tokens_tab",
          help="Maximum tokens for LLM response. Higher values allow more complete responses but cost more."
        )
        cfg_as["llm_max_tokens"] = max_tokens
        
        temperature = st.slider(
          "Temperature",
          min_value=0.0,
          max_value=1.0,
          value=float(cfg_as.get("llm_temperature", 0.0)),
          step=0.1,
          key="llm_temperature_tab",
          help="Controls randomness. 0.0 = deterministic, 1.0 = creative"
        )
        cfg_as["llm_temperature"] = temperature
      
      with col_llm2:
        st.metric("Token Limit", f"{max_tokens}", help="Current token limit per response")
        st.metric("Temperature", f"{temperature:.1f}", help="Current creativity setting")
      
      st.markdown("**Concurrent Processing**")
      col_conc1, col_conc2 = st.columns(2)
      with col_conc1:
        max_concurrent = st.number_input(
          "Max concurrent requests",
          min_value=1,
          max_value=50,
          value=int(cfg_as.get("llm_max_concurrent_requests", 10)),
          key="llm_max_concurrent_tab",
          help="Number of concurrent API requests to DeepSeek. Higher values = faster processing but more API load."
        )
        cfg_as["llm_max_concurrent_requests"] = max_concurrent
      with col_conc2:
        st.metric("Current Setting", f"{max_concurrent} requests", help="This controls how many YAML files are processed simultaneously")
      
      use_custom_global = st.checkbox("Override system prompt", value=False, key="llm_use_custom_global_tab")
      custom_prompt_global = None
      if use_custom_global:
        custom_prompt_global = st.text_area(
          "Custom system prompt",
          value=str(cfg_as.get("llm_system_prompt") or ""),
          height=200,
          key="llm_custom_prompt_global_tab",
        )
      st.markdown("**YAML Preview**")
      if not yaml_exists_as:
        st.info("No YAML profiles found. Generate profiles first in the Event Types tab.")
      yaml_files = sorted(profiles_dir_as.glob("*.yaml")) if yaml_exists_as else []
      uploaded_yaml = st.file_uploader("Or upload a YAML profile", type=["yaml","yml"], key="llm_up_yaml_tab")
      selection_mode = st.radio("Source", ["Choose from directory", "Use uploaded file"], index=0, key="llm_src_mode_tab")
      sel_yaml_text = ""
      sel_loc_id = ""
      sel_name = ""
      if selection_mode == "Choose from directory" and yaml_files:
        options = [p.name for p in yaml_files]
        sel_name = st.selectbox("YAML file", options=options, key="llm_sel_yaml_global_tab")
        if sel_name:
          p = profiles_dir_as / sel_name
          try:
            sel_yaml_text = p.read_text(encoding="utf-8")
            sel_loc_id = p.stem
          except Exception:
            sel_yaml_text = ""
        if sel_yaml_text:
          st.markdown(f"**Preview: {sel_name}**")
          st.code(sel_yaml_text, language="yaml")
      elif selection_mode == "Use uploaded file" and uploaded_yaml is not None:
        try:
          sel_yaml_text = uploaded_yaml.read().decode("utf-8", errors="ignore")
          st.markdown("**Preview: uploaded.yaml**")
          st.code(sel_yaml_text, language="yaml")
        except Exception:
          sel_yaml_text = ""
      col_run1, col_run2 = st.columns(2)
      with col_run1:
        run_one = st.button("Run on Selected", type="primary", key="llm_run_one_global_tab")
      with col_run2:
        run_all = st.button("Run on All in Directory", key="llm_run_all_global_tab")
      if run_one or run_all:
        cfg_llm = cfg_as
        llm_cfg = LLMConfig(
          api_key=None,
          model=str(cfg_llm.get("llm_model") or "deepseek-chat"),
          max_tokens=int(cfg_llm.get("llm_max_tokens", 8192)),
          temperature=float(cfg_llm.get("llm_temperature", 0.0)),
          max_concurrent_requests=int(cfg_llm.get("llm_max_concurrent_requests", 10)),
        )
        analyzer = LLMAnalyzer(llm_cfg)
        if not analyzer.is_available():
          st.error("DeepSeek API not configured. Set API key in config/global.json.")
        else:
          logs_root2 = _P(cfg.data_dir).parent / "logs"
          llm_log_path2 = (logs_root2 / "historical" / "aging_systems" / "llm_interactions.jsonl")
          llm_log_path2.parent.mkdir(parents=True, exist_ok=True)
          system_prompt_use = (custom_prompt_global if use_custom_global and custom_prompt_global else built_prompt_global)
          to_process = []
          if run_all and yaml_files:
            to_process = yaml_files
          elif run_one and sel_yaml_text:
            if selection_mode == "Choose from directory":
              to_process = [profiles_dir_as / (sel_loc_id + ".yaml")]
            else:
              to_process = [None]
          processed = 0
          if to_process:
            # Prepare requests for concurrent processing
            requests = []
            file_metadata = []
            
            for p in to_process:
              try:
                if p is None:
                  loc_id = "uploaded"
                  text = sel_yaml_text
                  yaml_path = "uploaded"
                else:
                  loc_id = p.stem
                  text = p.read_text(encoding="utf-8")
                  yaml_path = p.as_posix()
                
                requests.append({
                  "id": loc_id,
                  "system_prompt": system_prompt_use,
                  "user_prompt": text
                })
                file_metadata.append({
                  "location_id": loc_id,
                  "yaml_path": yaml_path
                })
              except Exception:
                continue
            
            if requests:
              # Show progress info
              total_files = len(requests)
              max_concurrent = llm_cfg.max_concurrent_requests
              st.info(f"Processing {total_files} file(s) with max {max_concurrent} concurrent requests...")
              
              # Create progress bar and status
              progress_bar = st.progress(0)
              status_text = st.empty()
              
              # Create detailed output window
              st.markdown("**Processing Details**")
              output_container = st.container()
              with output_container:
                output_placeholder = st.empty()
              
              # Track processing details
              processing_log = []
              
              def progress_callback(completed: int, total: int) -> None:
                progress = completed / total if total > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"Processed {completed}/{total} files ({progress:.1%}) - {total-completed} remaining")
                
                # Add progress update to log
                processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] Progress: {completed}/{total} files completed ({progress:.1%})")
                
                # Update detailed output with recent entries
                log_text = "\n".join(processing_log[-12:])  # Show last 12 entries
                if log_text:
                  output_placeholder.code(log_text, language="text")
              
              # Process requests concurrently
              start_time = time.perf_counter()
              processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] Starting concurrent processing with {max_concurrent} workers...")
              results = analyzer.analyze_texts_concurrent(requests, progress_callback)
              total_duration = time.perf_counter() - start_time
              processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] Concurrent processing completed in {total_duration:.1f}s")
              
              # Process results and prepare for parquet output
              successful = 0
              failed = 0
              parquet_data = []
              
              processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] Processing {len(results)} results...")
              
              for result, metadata in zip(results, file_metadata):
                try:
                  # Log to JSONL file
                  with open(llm_log_path2, "a", encoding="utf-8") as f:
                    log_entry = {
                      "ts": datetime.now(UTC).isoformat(),
                      "company": company,
                      "location_id": metadata["location_id"],
                      "yaml_path": metadata["yaml_path"],
                      "duration_ms": result["duration_ms"],
                      "success": result["success"],
                      "result_meta": (list(result["result"].keys()) if isinstance(result.get("result"), dict) else []),
                      "result": result["result"],
                    }
                    if "error" in result:
                      log_entry["error"] = result["error"]
                    f.write(json.dumps(log_entry) + "\n")
                  
                  # Prepare data for parquet
                  if result["success"] and result.get("result"):
                    llm_result = result["result"]
                    parquet_row = {
                      "location_id": metadata["location_id"],
                      "processing_timestamp": datetime.now(UTC).isoformat(),
                      "duration_ms": result["duration_ms"],
                      "success": result["success"],
                      "llm_response_raw": json.dumps(llm_result, ensure_ascii=False),
                    }
                    
                    # Parse the LLM response to extract structured data
                    parsed_response, parse_status = _parse_llm_response(llm_result)
                    
                    if parsed_response and isinstance(parsed_response, dict):
                      # Extract main fields
                      company = parsed_response.get("company", "")
                      analysis_date = parsed_response.get("analysis_date", "")
                      answers = parsed_response.get("answers", {})
                      
                      # Extract structured answers
                      parquet_row.update({
                        "company": company,
                        "analysis_date": analysis_date,
                        "last_system_install_date": _extract_answer_value(answers.get("last_system_install_date")),
                        "go_to_contractor": _extract_answer_value(answers.get("go_to_contractor")),
                        "estimated_current_system_age_years": _extract_answer_value(answers.get("estimated_current_system_age_years")),
                        "most_likely_next_job": _extract_answer_value(answers.get("most_likely_next_job")),
                        "last_permit_issue_date": _extract_answer_value(answers.get("last_permit_issue_date")),
                        "last_job_date": _extract_answer_value(answers.get("last_job_date")),
                      })
                      
                      # Count non-null extracted values
                      extracted_count = len([v for k, v in parquet_row.items() if k not in ['location_id', 'processing_timestamp', 'duration_ms', 'success', 'llm_response_raw'] and v is not None and v != ''])
                      
                      if "truncated" in parse_status.lower():
                        processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] ⚠ {metadata['location_id']}: {parse_status} - extracted {extracted_count} fields ({result['duration_ms']}ms)")
                      else:
                        processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] ✓ {metadata['location_id']}: {parse_status} - extracted {extracted_count} fields ({result['duration_ms']}ms)")
                    else:
                      # Fallback for unparseable responses
                      parquet_row.update({
                        "company": "",
                        "analysis_date": "",
                        "last_system_install_date": None,
                        "go_to_contractor": None,
                        "estimated_current_system_age_years": None,
                        "most_likely_next_job": None,
                        "last_permit_issue_date": None,
                        "last_job_date": None,
                      })
                      processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] ✗ {metadata['location_id']}: {parse_status} ({result['duration_ms']}ms)")
                    
                    parquet_data.append(parquet_row)
                    successful += 1
                  else:
                    failed += 1
                    error_msg = result.get("error", "Unknown error")
                    processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] ✗ {metadata['location_id']}: {error_msg}")
                    
                except Exception as e:
                  failed += 1
                  processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] ✗ {metadata['location_id']}: Processing error - {str(e)}")
              
              # Save results to parquet file
              parquet_file_saved = False
              parquet_path = None
              
              if parquet_data:
                try:
                  # Determine output path using configurable filename
                  output_base = base_out_as
                  llm_output_filename = cfg_as.get("llm_analysis_output_filename") or "aging_systems.parquet"
                  parquet_path = output_base / llm_output_filename
                  
                  # Create DataFrame and add additional payload columns
                  df_results = pd.DataFrame(parquet_data)
                  
                  # Add additional payload columns if configured
                  additional_columns = cfg_as.get("additional_payload_columns", [])
                  if additional_columns:
                    # Need to get location data for additional columns
                    # This would require loading the locations file again or passing it through
                    # For now, log that additional columns were requested
                    ui_log(f"[aging_systems] Additional payload columns requested: {additional_columns}")
                  
                  df_results.to_parquet(parquet_path, index=False, engine="pyarrow")
                  
                  parquet_file_saved = True
                  processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] ✓ Saved {len(parquet_data)} records to {parquet_path.name}")
                  
                except Exception as e:
                  processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] ✗ Failed to save parquet: {str(e)}")
              else:
                processing_log.append(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] No successful results to save to parquet")
              
              # Update final progress display
              final_log = "\n".join(processing_log[-15:])
              output_placeholder.code(final_log, language="text")
              
              # Show completion summary
              avg_time_per_file = total_duration / len(results) if results else 0
              st.success(f"Concurrent LLM processing complete!")
              
              col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
              with col_sum1:
                st.metric("Files Processed", f"{len(results)}")
              with col_sum2:
                st.metric("Successful", f"{successful}")
              with col_sum3:
                st.metric("Failed", f"{failed}")
              with col_sum4:
                st.metric("Avg Time/File", f"{avg_time_per_file:.1f}s")
              
              # Show file outputs
              st.markdown("**Output Files**")
              col_out1, col_out2 = st.columns(2)
              with col_out1:
                st.caption(f"📊 **Parquet Results**: `{parquet_path.name if parquet_path else 'Not created'}`")
                if parquet_file_saved and parquet_path:
                  st.caption(f"   Path: `{parquet_path.as_posix()}`")
                  st.caption(f"   Records: {len(parquet_data)}")
              with col_out2:
                st.caption(f"📋 **JSONL Log**: `{llm_log_path2.name}`")
                st.caption(f"   Path: `{llm_log_path2.as_posix()}`")
                st.caption(f"   Entries: {len(results)}")
              
              st.caption(f"Total time: {total_duration:.1f}s | Concurrency: {max_concurrent}")
              processed = len(results)

    # Placeholders for other event types
    with st.expander("Overdue Maintenance", expanded=False):
      st.caption("LLM logic placeholder. Configure once logic exists.")
      st.text_area("System prompt", value="", key="llm_om_prompt", height=120)
    with st.expander("Canceled Jobs", expanded=False):
      st.caption("LLM logic placeholder. Configure once logic exists.")
      st.text_area("System prompt", value="", key="llm_cj_prompt", height=120)
    with st.expander("Unsold Estimates", expanded=False):
      st.caption("LLM logic placeholder. Configure once logic exists.")
      st.text_area("System prompt", value="", key="llm_ue_prompt", height=120)
    with st.expander("Lost Customers", expanded=False):
      st.caption("LLM logic placeholder. Configure once logic exists.")
      st.text_area("System prompt", value="", key="llm_lc_prompt", height=120)

  # Standalone LLM section removed; now under LLM Settings tab


def _extract_customer_addresses(customers_df: pd.DataFrame, config: Dict[str, Any], 
                               processing_limit: int, ui_log, log_func) -> pd.DataFrame:
  """Step 1: Extract customer ID and Full Address from Customer.parquet"""
  
  customer_id_col = config.get("customer_id_column", "Customer ID")
  address_col = config.get("customer_address_column", "Full Address")
  
  # Get customers to process
  all_customer_ids = customers_df[customer_id_col].astype(str).dropna().unique().tolist()
  if processing_limit > 0:
    customer_ids = all_customer_ids[:processing_limit]
  else:
    customer_ids = all_customer_ids
  
  ui_log(f"Extracting addresses for {len(customer_ids)} customers (of {len(all_customer_ids)} total)")
  
  address_data = []
  skipped_no_address = 0
  
  for i, customer_id in enumerate(customer_ids):
    if i % 1000 == 0 and i > 0:
      ui_log(f"Processed {i}/{len(customer_ids)} customer addresses...")
    
    # Get customer row
    customer_row = customers_df[customers_df[customer_id_col].astype(str) == customer_id]
    if customer_row.empty:
      continue
      
    customer_data = customer_row.iloc[0]
    full_address = str(customer_data[address_col]) if pd.notna(customer_data[address_col]) else ""
    
    if not full_address:
      skipped_no_address += 1
      continue
    
    address_data.append({
      "customer_id": customer_id,
      "full_address": full_address,
      "customer_name": str(customer_data.get("Customer Name", "")) if "Customer Name" in customer_data.index else "",
      "phone": str(customer_data.get("Phone Number", "")) if "Phone Number" in customer_data.index else ""
    })
  
  log_func("extract_addresses_complete", {
    "customers_processed": len(customer_ids),
    "addresses_extracted": len(address_data),
    "skipped_no_address": skipped_no_address
  })
  
  return pd.DataFrame(address_data)


def _extract_customer_calls(calls_df: pd.DataFrame, addresses_file: _P, 
                           config: Dict[str, Any], ui_log, log_func) -> pd.DataFrame:
  """Step 2: Extract call data for each customer from Calls.parquet"""
  
  # Load customer addresses
  addresses_df = pd.read_parquet(addresses_file)
  customer_ids = addresses_df["customer_id"].tolist()
  
  customer_id_col = config.get("customer_id_column", "Customer ID")
  call_date_col = config.get("call_date_column", "Call Date")
  
  ui_log(f"Extracting call history for {len(customer_ids)} customers...")
  
  call_data = []
  skipped_no_calls = 0
  
  for i, customer_id in enumerate(customer_ids):
    if i % 1000 == 0 and i > 0:
      ui_log(f"Processed calls for {i}/{len(customer_ids)} customers...")
    
    # Get customer info
    customer_info = addresses_df[addresses_df["customer_id"] == customer_id].iloc[0]
    
    # Get all calls for this customer
    customer_calls = calls_df[calls_df[customer_id_col].astype(str) == customer_id]
    
    if customer_calls.empty:
      skipped_no_calls += 1
      continue
    
    # Parse call dates
    call_dates = []
    for _, call_row in customer_calls.iterrows():
      date_value = call_row[call_date_col]
      if pd.notna(date_value):
        try:
          parsed_date = pd.to_datetime(date_value, errors='coerce')
          if pd.notna(parsed_date):
            call_dates.append(parsed_date.date())
        except Exception:
          continue
    
    if not call_dates:
      skipped_no_calls += 1
      continue
    
    call_dates.sort()
    first_contact = call_dates[0]
    last_contact = call_dates[-1]
    
    call_data.append({
      "customer_id": customer_id,
      "customer_name": customer_info["customer_name"],
      "phone": customer_info["phone"],
      "full_address": customer_info["full_address"],
      "first_contact_date": first_contact.isoformat(),
      "last_contact_date": last_contact.isoformat(),
      "total_calls": len(call_dates)
    })
  
  log_func("extract_calls_complete", {
    "customers_processed": len(customer_ids),
    "calls_extracted": len(call_data),
    "skipped_no_calls": skipped_no_calls
  })
  
  return pd.DataFrame(call_data)


def _process_permit_matching_pipeline(calls_file: _P, permit_data_path: _P, 
                                     config: Dict[str, Any], ui_log, log_func) -> pd.DataFrame:
  """Process permit matching in chunks to manage memory"""
  from datahound.events.address_utils import normalize_address_street, extract_street_from_full_address
  
  # Load customer call data
  calls_df = pd.read_parquet(calls_file)
  ui_log(f"Loaded {len(calls_df)} customer call records for permit matching")
  
  # Build address lookup for faster matching
  customer_addresses = {}
  for _, row in calls_df.iterrows():
    customer_id = row["customer_id"]
    address = row["full_address"]
    
    # Normalize customer address for matching
    try:
      customer_street = extract_street_from_full_address(address)
      normalized_customer = normalize_address_street(customer_street)
      if normalized_customer:
        customer_addresses[normalized_customer] = row.to_dict()
    except Exception:
      continue
  
  ui_log(f"Built address index for {len(customer_addresses)} normalized addresses")
  
  # Log first few customer addresses for debugging
  if customer_addresses:
    sample_addresses = list(customer_addresses.keys())[:3]
    for i, addr in enumerate(sample_addresses):
      ui_log(f"Sample customer address {i+1}: '{addr}'")
  
  # Load permit data in chunks and find matches
  chunk_size = 50000
  permit_matches = []
  total_permits_processed = 0
  
  ui_log(f"Loading permit data in chunks of {chunk_size}...")
  
  for chunk_num, permit_chunk in enumerate(pd.read_csv(permit_data_path, chunksize=chunk_size, dtype=str, low_memory=False)):
    ui_log(f"Processing permit chunk {chunk_num + 1} ({len(permit_chunk)} permits)...")
    
    # Find address column in permits - check actual column names
    permit_address_col = None
    available_cols = list(permit_chunk.columns)
    
    # Log available columns for first chunk to help debug
    if chunk_num == 0:
      ui_log(f"Available permit columns: {available_cols[:10]}...")  # Show first 10
    
    for col in available_cols:
      if 'location' in col.lower() or 'address' in col.lower():
        permit_address_col = col
        break
    
    if not permit_address_col:
      ui_log(f"No address column found in permit chunk {chunk_num + 1}")
      continue
    
    if chunk_num == 0:
      ui_log(f"Using permit address column: {permit_address_col}")
    
    chunk_matches = 0
    
    # Process permits in this chunk
    for _, permit_row in permit_chunk.iterrows():
      permit_address = str(permit_row[permit_address_col]) if pd.notna(permit_row[permit_address_col]) else ""
      
      if not permit_address:
        continue
      
      try:
        permit_street = extract_street_from_full_address(permit_address)
        normalized_permit = normalize_address_street(permit_street)
        
        # Check if this permit address matches any customer address
        if normalized_permit in customer_addresses:
          customer_data = customer_addresses[normalized_permit]
          chunk_matches += 1
          
          # Get contractor info
          contractor = str(permit_row.get("Contractor Company Name", "")) if "Contractor Company Name" in permit_row.index else ""
          
          # Create match record
          match_data = {
            "customer_id": customer_data["customer_id"],
            "customer_name": customer_data["customer_name"],
            "phone": customer_data["phone"],
            "full_address": customer_data["full_address"],
            "first_contact_date": customer_data["first_contact_date"],
            "last_contact_date": customer_data["last_contact_date"],
            "total_calls": customer_data["total_calls"],
            "permit_contractor": contractor,
            "permit_applied_date": str(permit_row.get("Applied Date", "")) if "Applied Date" in permit_row.index else "",
            "permit_issued_date": str(permit_row.get("Issued Date", "")) if "Issued Date" in permit_row.index else "",
            "permit_description": str(permit_row.get("Description", "")) if "Description" in permit_row.index else "",
            "permit_address": permit_address,
            "normalized_customer_address": normalized_permit,  # They match, so use the same
            "normalized_permit_address": normalized_permit
          }
          permit_matches.append(match_data)
          
          # Log first few matches for debugging
          if len(permit_matches) <= 3:
            ui_log(f"Match {len(permit_matches)}: Customer {customer_data['customer_id']} -> {contractor}")
          
      except Exception as e:
        # Log address normalization failures for debugging
        if chunk_matches == 0 and len(permit_matches) < 5:  # Only log first few failures
          log_func("address_normalization_error", {
            "permit_address": permit_address,
            "error": str(e),
            "chunk": chunk_num + 1
          })
        continue
    
    ui_log(f"Chunk {chunk_num + 1}: Found {chunk_matches} matches")
    total_permits_processed += len(permit_chunk)
  
  log_func("permit_matching_complete", {
    "customers_with_calls": len(calls_df),
    "total_permits_processed": total_permits_processed,
    "permit_matches_found": len(permit_matches),
    "unique_customers_matched": len(set(match["customer_id"] for match in permit_matches))
  })
  
  return pd.DataFrame(permit_matches)


def _analyze_lost_customers_pipeline(matches_file: _P, config: Dict[str, Any], ui_log, log_func) -> pd.DataFrame:
  """Analyze permit matches to identify lost customers with detailed logging"""
  
  # Load permit matches
  matches_df = pd.read_parquet(matches_file)
  ui_log(f"Analyzing {len(matches_df)} permit matches...")
  
  if matches_df.empty:
    log_func("analysis_empty", {"reason": "no_permit_matches"})
    return pd.DataFrame(columns=[
      "event_type", "entity_type", "entity_id", "severity", "detected_at", "rule_name",
      "customer_id", "first_contact_date", "last_contact_date", "competitor_used",
      "shopper_customer", "lost_customer", "customer_name", "phone", "address"
    ])
  
  company_names = config.get("company_names", ["McCullough Heating & Air", "McCullough Heating and Air"])
  ui_log(f"Using company names for exclusion: {company_names}")
  
  lost_customers = []
  detailed_analysis_log = []
  
  # Create detailed analysis log file for inspection
  analysis_log_file = matches_file.parent / "detailed_customer_analysis.jsonl"
  
  # Group by customer to analyze each customer's permits
  customer_groups = matches_df.groupby("customer_id")
  ui_log(f"Analyzing {len(customer_groups)} unique customers with permit matches")
  
  for customer_num, (customer_id, customer_matches) in enumerate(customer_groups):
    customer_data = customer_matches.iloc[0]  # Get customer info
    
    analysis_record = {
      "customer_id": customer_id,
      "customer_name": customer_data["customer_name"],
      "total_permits": len(customer_matches),
      "analysis_timestamp": pd.Timestamp.now().isoformat(),
      "customer_number": customer_num + 1
    }
    
    # Parse contact dates
    try:
      first_contact = pd.to_datetime(customer_data["first_contact_date"]).date()
      last_contact = pd.to_datetime(customer_data["last_contact_date"]).date()
      analysis_record["first_contact_date"] = first_contact.isoformat()
      analysis_record["last_contact_date"] = last_contact.isoformat()
      analysis_record["contact_span_days"] = (last_contact - first_contact).days
    except Exception as e:
      analysis_record["error"] = f"Failed to parse contact dates: {e}"
      analysis_record["first_contact_raw"] = customer_data["first_contact_date"]
      analysis_record["last_contact_raw"] = customer_data["last_contact_date"]
      detailed_analysis_log.append(analysis_record)
      continue
    
    # Analyze each permit for this customer
    company_permits = []
    competitor_permits = []
    invalid_permits = []
    permit_details = []
    
    for permit_idx, (_, permit_row) in enumerate(customer_matches.iterrows()):
      contractor = str(permit_row["permit_contractor"]).strip()
      
      permit_analysis = {
        "permit_index": permit_idx + 1,
        "contractor_raw": contractor,
        "applied_date_raw": str(permit_row["permit_applied_date"]),
        "issued_date_raw": str(permit_row["permit_issued_date"]),
        "description": str(permit_row["permit_description"])
      }
      
      if not contractor:
        permit_analysis["skip_reason"] = "no_contractor"
        invalid_permits.append(permit_analysis)
        continue
      
      # Get permit date (prefer applied, fallback to issued)
      permit_date = None
      date_value = permit_row["permit_applied_date"]
      
      if not date_value or pd.isna(date_value) or str(date_value).strip() == "":
        date_value = permit_row["permit_issued_date"]
        permit_analysis["date_source"] = "issued_date"
      else:
        permit_analysis["date_source"] = "applied_date"
      
      permit_analysis["date_value_used"] = str(date_value)
      
      if date_value and not pd.isna(date_value) and str(date_value).strip():
        try:
          permit_date_ts = pd.to_datetime(date_value, errors='coerce')
          if pd.notna(permit_date_ts):
            permit_date = permit_date_ts.date()
            permit_analysis["parsed_date"] = permit_date.isoformat()
            permit_analysis["days_from_first_contact"] = (permit_date - first_contact).days
            permit_analysis["days_from_last_contact"] = (permit_date - last_contact).days
          else:
            permit_analysis["date_parse_error"] = f"Failed to parse date: {date_value}"
            permit_date = None
        except Exception as e:
          permit_analysis["date_parse_error"] = str(e)
          permit_date = None
      
      if not permit_date:
        permit_analysis["skip_reason"] = "no_valid_date"
        invalid_permits.append(permit_analysis)
        continue
      
      # Check if this is our company or competitor
      company_match_details = []
      is_our_company = False
      
      for company_name in company_names:
        match_check = {
          "company_name": company_name,
          "contractor": contractor,
          "company_in_contractor": company_name.lower() in contractor.lower(),
          "contractor_in_company": contractor.lower() in company_name.lower()
        }
        company_match_details.append(match_check)
        
        if company_name.lower() in contractor.lower():
          is_our_company = True
      
      permit_analysis.update({
        "is_our_company": is_our_company,
        "company_match_details": company_match_details,
        "permit_date_vs_first_contact": "after" if permit_date > first_contact else "before",
        "permit_date_vs_last_contact": "after" if permit_date > last_contact else "before"
      })
      
      permit_info = {
        "contractor": contractor,
        "date": permit_date,
        "is_our_company": is_our_company,
        "analysis": permit_analysis
      }
      
      if is_our_company:
        company_permits.append(permit_info)
        permit_analysis["classification"] = "company_permit"
      else:
        competitor_permits.append(permit_info)
        permit_analysis["classification"] = "competitor_permit"
      
      permit_details.append(permit_analysis)
    
    # Log permit classification results
    analysis_record.update({
      "company_permits_count": len(company_permits),
      "competitor_permits_count": len(competitor_permits),
      "invalid_permits_count": len(invalid_permits),
      "permit_details": permit_details,
      "company_permits_summary": [{"contractor": p["contractor"], "date": p["date"].isoformat()} for p in company_permits],
      "competitor_permits_summary": [{"contractor": p["contractor"], "date": p["date"].isoformat()} for p in competitor_permits]
    })
    
    # Determine lost customer status with detailed logic
    lost_customer = False
    competitor_used = None
    shopper_customer = False
    decision_log = []
    
    # Check for lost customer: competitor permit after our last contact
    lost_customer_checks = []
    for comp_permit in competitor_permits:
      permit_date = comp_permit["date"]
      
      # Ensure we have valid dates for comparison
      if not permit_date or pd.isna(permit_date):
        decision_check = {
          "check_type": "lost_customer_test",
          "competitor": comp_permit["contractor"],
          "permit_date": "invalid_date",
          "last_contact_date": last_contact.isoformat(),
          "permit_after_last_contact": False,
          "result": "invalid_permit_date",
          "error": "permit_date is NaT or None"
        }
        lost_customer_checks.append(decision_check)
        continue
      
      try:
        decision_check = {
          "check_type": "lost_customer_test",
          "competitor": comp_permit["contractor"],
          "permit_date": permit_date.isoformat(),
          "last_contact_date": last_contact.isoformat(),
          "permit_after_last_contact": permit_date > last_contact,
          "days_after_last_contact": (permit_date - last_contact).days
        }
        
        if permit_date > last_contact:
          lost_customer = True
          competitor_used = comp_permit["contractor"]
          decision_check["result"] = "LOST_CUSTOMER_IDENTIFIED"
          decision_check["competitor_used"] = competitor_used
          lost_customer_checks.append(decision_check)
          break
        else:
          decision_check["result"] = "permit_before_last_contact"
          lost_customer_checks.append(decision_check)
          
      except Exception as e:
        decision_check = {
          "check_type": "lost_customer_test",
          "competitor": comp_permit["contractor"],
          "permit_date": str(permit_date),
          "last_contact_date": last_contact.isoformat(),
          "comparison_error": str(e),
          "result": "date_comparison_failed"
        }
        lost_customer_checks.append(decision_check)
    
    decision_log.extend(lost_customer_checks)
    
    # Check for shopper customer: competitor work after first contact, but our most recent work
    if competitor_permits and company_permits:
      try:
        # Filter out permits with invalid dates for comparison
        valid_company_permits = [p for p in company_permits if p["date"] and not pd.isna(p["date"])]
        valid_competitor_permits = [p for p in competitor_permits if p["date"] and not pd.isna(p["date"])]
        
        latest_company = max(valid_company_permits, key=lambda x: x["date"]) if valid_company_permits else None
        latest_competitor = max(valid_competitor_permits, key=lambda x: x["date"]) if valid_competitor_permits else None
        
        competitor_after_first = False
        try:
          competitor_after_first = any(comp["date"] > first_contact for comp in valid_competitor_permits if comp["date"] and not pd.isna(comp["date"]))
        except Exception:
          competitor_after_first = False
        
        shopper_check = {
          "check_type": "shopper_customer_test",
          "has_competitor_permits": bool(competitor_permits),
          "has_company_permits": bool(company_permits),
          "valid_competitor_permits": len(valid_competitor_permits),
          "valid_company_permits": len(valid_company_permits),
          "latest_company_date": latest_company["date"].isoformat() if latest_company else None,
          "latest_competitor_date": latest_competitor["date"].isoformat() if latest_competitor else None,
          "competitor_after_first_contact": competitor_after_first
        }
        
        if latest_company and latest_competitor:
          try:
            company_more_recent = latest_company["date"] > latest_competitor["date"]
            shopper_check["company_more_recent"] = company_more_recent
            
            if competitor_after_first and company_more_recent:
              shopper_customer = True
              shopper_check["result"] = "SHOPPER_CUSTOMER_IDENTIFIED"
            else:
              shopper_check["result"] = "not_shopper_pattern"
          except Exception as e:
            shopper_check["result"] = "date_comparison_failed"
            shopper_check["comparison_error"] = str(e)
        else:
          shopper_check["result"] = "insufficient_valid_dates"
        
        decision_log.append(shopper_check)
        
      except Exception as e:
        shopper_check = {
          "check_type": "shopper_customer_test",
          "error": str(e),
          "result": "shopper_analysis_failed"
        }
        decision_log.append(shopper_check)
    
    # Final decision summary
    final_decision = {
      "lost_customer": lost_customer,
      "competitor_used": competitor_used,
      "shopper_customer": shopper_customer,
      "decision_reasoning": {
        "lost_logic": "Found competitor permit after last contact date" if lost_customer else "No competitor permits after last contact",
        "shopper_logic": "Competitor work after first contact but company most recent" if shopper_customer else "No shopper pattern detected"
      }
    }
    
    # Log detailed analysis for this customer
    analysis_record.update({
      "decision_log": decision_log,
      "final_decision": final_decision
    })
    
    detailed_analysis_log.append(analysis_record)
    
    # Write detailed log entry to file for inspection
    with open(analysis_log_file, "a", encoding="utf-8") as f:
      f.write(json.dumps(analysis_record, indent=2) + "\n")
    
    # Create event if lost customer
    if lost_customer:
      from datetime import datetime, UTC
      
      # Check exclusion rule for NaN competitor
      exclude_nan_competitor = config.get("exclude_nan_competitor", True)
      should_exclude = exclude_nan_competitor and (not competitor_used or competitor_used.strip() == "" or str(competitor_used).lower() in ['nan', 'none', 'null'])
      
      # Log exclusion decision
      exclusion_check = {
        "exclude_nan_competitor_enabled": exclude_nan_competitor,
        "competitor_used_raw": competitor_used,
        "competitor_used_is_empty": not competitor_used or competitor_used.strip() == "",
        "competitor_used_is_nan": str(competitor_used).lower() in ['nan', 'none', 'null'],
        "should_exclude": should_exclude,
        "exclusion_result": "EXCLUDED" if should_exclude else "INCLUDED"
      }
      
      # Add exclusion check to analysis record
      analysis_record["exclusion_check"] = exclusion_check
      
      if should_exclude:
        ui_log(f"EXCLUDED: Customer {customer_id} - no specific competitor identified (competitor_used: '{competitor_used}')")
        analysis_record["final_decision"]["excluded"] = True
        analysis_record["final_decision"]["exclusion_reason"] = "no_specific_competitor"
      else:
        # Determine severity
        today = pd.Timestamp.now().date()
        months_since = (today.year - last_contact.year) * 12 + (today.month - last_contact.month)
        if today.day < last_contact.day:
          months_since -= 1
        
        if competitor_used:
          if months_since >= 24:
            severity = "critical"
          elif months_since >= 12:
            severity = "high"
          else:
            severity = "medium"
        else:
          severity = "medium"
        
        lost_customers.append({
          "event_type": "lost_customers",
          "entity_type": "customer",
          "entity_id": customer_id,
          "severity": severity,
          "detected_at": datetime.now(UTC).isoformat(),
          "rule_name": "Lost Customers Analysis",
          "customer_id": customer_id,
          "first_contact_date": first_contact.isoformat(),
          "last_contact_date": last_contact.isoformat(),
          "competitor_used": competitor_used or "",
          "shopper_customer": shopper_customer,
          "lost_customer": True,
          "customer_name": customer_data["customer_name"],
          "phone": customer_data["phone"],
          "address": customer_data["full_address"],
          "permits_analyzed": len(customer_matches),
          "months_since_last_contact": months_since
        })
        
        ui_log(f"LOST CUSTOMER FOUND: {customer_id} ({customer_data['customer_name']}) - {competitor_used}")
        analysis_record["final_decision"]["included"] = True
  
  # Calculate exclusion statistics
  total_lost_identified = sum(1 for record in detailed_analysis_log if record.get("final_decision", {}).get("lost_customer", False))
  excluded_count = sum(1 for record in detailed_analysis_log if record.get("exclusion_check", {}).get("should_exclude", False))
  
  # Log summary of detailed analysis
  ui_log(f"Detailed analysis saved to: {analysis_log_file}")
  ui_log(f"Lost customers identified: {total_lost_identified}, Excluded due to NaN competitor: {excluded_count}, Final output: {len(lost_customers)}")
  
  log_func("lost_customer_analysis_complete", {
    "customers_analyzed": len(customer_groups),
    "lost_customers_identified": total_lost_identified,
    "excluded_nan_competitor": excluded_count,
    "lost_customers_final_output": len(lost_customers),
    "exclusion_filter_enabled": config.get("exclude_nan_competitor", True),
    "detailed_analysis_file": str(analysis_log_file),
    "analysis_records": len(detailed_analysis_log)
  })
  
  return pd.DataFrame(lost_customers)


  # Scheduler Status Tab
  with tab_scheduler:
    st.markdown("### 📊 Scheduler Status")
    render_scheduler_status(key_context="event_detection")
    
    st.markdown("---")
    st.markdown("### 🗂️ Event Scan Task Management")
    
    # Show all event scan tasks for this company
    from datahound.scheduler import DataHoundScheduler
    scheduler = DataHoundScheduler(Path.cwd())
    all_tasks = scheduler.get_all_tasks()
    event_tasks = [t for t in all_tasks if t.task_config.task_type.value == 'historical_event_scan' and t.task_config.company == company]
    
    if event_tasks:
      # Separate individual and batch tasks
      individual_tasks = [t for t in event_tasks if not (hasattr(t.task_config, 'scan_all_events') and t.task_config.scan_all_events)]
      batch_tasks = [t for t in event_tasks if hasattr(t.task_config, 'scan_all_events') and t.task_config.scan_all_events]
      
      col1, col2 = st.columns(2)
      
      with col1:
        st.markdown("#### 📊 Individual Event Tasks")
        if individual_tasks:
          for task in individual_tasks:
            with st.container():
              st.write(f"**{task.task_config.event_type.replace('_', ' ').title()}**")
              st.caption(f"Status: {task.status.value} | Runs: {task.run_count}")
              if task.next_run:
                st.caption(f"Next: {task.next_run.strftime('%m/%d %H:%M')}")
              
              task_col1, task_col2 = st.columns(2)
              with task_col1:
                if st.button("⏸️ Pause", key=f"pause_individual_{task.task_id}"):
                  scheduler.pause_task(task.task_id)
                  st.rerun()
              with task_col2:
                if st.button("🗑️ Delete", key=f"delete_individual_{task.task_id}"):
                  scheduler.delete_task(task.task_id)
                  st.rerun()
              st.markdown("---")
        else:
          st.info("No individual event tasks")
      
      with col2:
        st.markdown("#### 🔄 Batch Event Tasks")
        if batch_tasks:
          st.warning("⚠️ Batch tasks will duplicate individual tasks!")
          for task in batch_tasks:
            with st.container():
              st.write(f"**{task.name}**")
              st.caption(f"Status: {task.status.value} | Runs: {task.run_count}")
              if task.next_run:
                st.caption(f"Next: {task.next_run.strftime('%m/%d %H:%M')}")
              
              if st.button("🗑️ Delete Batch Task", key=f"delete_batch_{task.task_id}", type="primary"):
                scheduler.delete_task(task.task_id)
                st.success("Batch task deleted!")
                st.rerun()
              st.markdown("---")
        else:
          st.success("✅ No duplicate batch tasks")
    else:
      st.info("No event scan tasks found for this company")


def main() -> None:
  st.title("Historical Events")
  st.caption("Home / Historical Events")
  company, cfg = select_company_config()
  render(company, cfg)


if __name__ == "__main__":
  main()
