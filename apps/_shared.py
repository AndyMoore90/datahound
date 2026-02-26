from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import json
import sys
import os


def ensure_root_on_path() -> None:
  root = Path(__file__).resolve().parents[1]
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  try:
    from datahound.env import load_env_fallback
    load_env_fallback(root)
  except Exception:
    pass


def list_companies() -> List[str]:
  base = Path("companies")
  if not base.exists():
    return []
  names = []
  for p in base.iterdir():
    if p.is_dir() and (p / "config.json").exists():
      names.append(p.name)
  return sorted(names)


def get_config_path(company: str) -> Path:
  return Path("companies") / company / "config.json"


def read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
  if not path.exists():
    return []
  lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
  items: List[Dict[str, Any]] = []
  for line in lines[-limit:]:
    try:
      items.append(json.loads(line))
    except Exception:
      continue
  return items


def load_all_changes(logs_dir: Path | list[Path], limit_per_file: int = 50000) -> pd.DataFrame:
  dirs = logs_dir if isinstance(logs_dir, list) else [logs_dir]
  frames: List[pd.DataFrame] = []
  for base in dirs:
    files = [
      base / "job_changes_log.jsonl",
      base / "customer_changes_log.jsonl",
      base / "membership_changes_log.jsonl",
      base / "call_changes_log.jsonl",
      base / "estimate_changes_log.jsonl",
      base / "invoice_changes_log.jsonl",
      base / "location_changes_log.jsonl",
    ]
    for p in files:
      rows = read_jsonl(p, int(limit_per_file))
      if not rows:
        continue
      df = pd.DataFrame(rows)
      for col in ["ts","company","file_type","change_type","id","column","old","new","master","prepared"]:
        if col not in df.columns:
          df[col] = None
      try:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.tz_localize(None)
      except Exception:
        pass
      df["file_type"] = df["file_type"].astype("string")
      df["change_type"] = df["change_type"].astype("string")
      df["column"] = df["column"].astype("string")
      df["prepared"] = df["prepared"].astype("string")
      frames.append(df)
  if not frames:
    return pd.DataFrame(columns=["ts","company","file_type","change_type","id","column","old","new","master","prepared"])
  all_df = pd.concat(frames, ignore_index=True)
  all_df.sort_values("ts", inplace=True)
  return all_df


def select_company_config():
  # Lazy import to avoid circulars in non-Streamlit contexts
  import streamlit as st
  from datahound.download.types import load_config
  
  companies = list_companies()
  if not companies:
    st.error("No companies found. Add a company folder with config.json under companies/.")
    st.stop()
  
  # Initialize persistent company selection across all pages
  if 'datahound_selected_company' not in st.session_state:
    st.session_state.datahound_selected_company = companies[0]
  
  # Ensure selected company is still valid (in case companies list changed)
  if st.session_state.datahound_selected_company not in companies:
    st.session_state.datahound_selected_company = companies[0]
  
  # Company selector with persistent session state
  def on_company_change():
    st.session_state.datahound_selected_company = st.session_state.company_selector_global
    # Force rerun to update all components
    st.rerun()
  
  company = st.sidebar.selectbox(
    "Company", 
    companies, 
    index=companies.index(st.session_state.datahound_selected_company),
    key="company_selector_global",
    on_change=on_company_change
  )
  
  # Ensure session state is always in sync
  st.session_state.datahound_selected_company = company
  
  cfg = load_config(get_config_path(company))
  return company, cfg


def load_professional_styling():
  import streamlit as st
  
  css_file = Path(__file__).parent / "assets" / "style.css"
  
  if css_file.exists():
    with open(css_file, "r") as f:
      css_content = f.read()
      
    # Inject custom CSS
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    
    # Add professional fonts
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
  else:
    # Fallback minimal styling
    st.markdown("""
    <style>
    .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stButton > button { border-radius: 8px; border: 1px solid #D1D5DB; transition: all 0.15s; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .stMetric { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)


def setup_professional_page(title: str, icon: str = "⚡", layout: str = "wide"):
  import streamlit as st
  
  st.set_page_config(
    page_title=f"DataHound Pro - {title}",
    page_icon=icon,
    layout=layout,
    initial_sidebar_state="expanded"
  )
  
  load_professional_styling()


