import json
import shutil
import sys
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, List

import streamlit as st
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._shared import (
    ensure_root_on_path, select_company_config, list_companies,
    get_config_path, read_jsonl, load_all_changes,
)

ensure_root_on_path()

from datahound.download.types import (
    load_config, save_config, load_global_config, save_global_config,
    GmailConfig, DownloadConfig, new_company_config_template,
)

try:
    from apps.components.ui_components import (
        inject_custom_css, dh_page_header, dh_alert, dh_path_with_copy,
    )
    UI = True
except ImportError:
    UI = False


# ---------------------------------------------------------------------------
# Companies tab
# ---------------------------------------------------------------------------

def _render_companies() -> None:
    companies = list_companies()
    company = st.selectbox("Company", companies, key="admin_company_select")
    cfg = load_config(get_config_path(company))

    st.markdown("#### Edit Company")
    with st.form("edit_company_form"):
        st.text_input("Company", value=cfg.company, disabled=True)
        data_dir_str = st.text_input("Data dir", value=str(cfg.data_dir))
        mark = st.checkbox("Mark as read by default", value=cfg.mark_as_read)
        ext_choices = [".xlsx", ".xls", ".csv"]
        allowed = st.multiselect("Allowed extensions", ext_choices,
                                 default=[e for e in cfg.allowed_extensions if e in ext_choices])
        scopes_str = st.text_area("Gmail scopes (comma-separated)", value=",".join(cfg.gmail.scopes))
        creds_path_str = st.text_input("Credentials path", value=str(cfg.gmail.credentials_path))
        token_path_str = st.text_input("Token path", value=str(cfg.gmail.token_path))
        links_str = st.text_area("Link prefixes (one per line)", value="\n".join(cfg.gmail.link_prefixes))
        rows = [{"file_type": k, "query": v} for k, v in cfg.gmail.query_by_type.items()]
        edited_rows = st.data_editor(rows, num_rows="dynamic", width="stretch", key="admin_queries_editor")

        col_a, col_b = st.columns(2)
        with col_a:
            save_btn = st.form_submit_button("Save Config", type="primary")
        with col_b:
            mkdirs_btn = st.form_submit_button("Create Missing Dirs")

    if save_btn:
        try:
            scopes = [s.strip() for s in scopes_str.split(",") if s.strip()]
            links = [l.strip() for l in links_str.splitlines() if l.strip()]
            qmap: Dict[str, str] = {}
            rows_iter = edited_rows if isinstance(edited_rows, list) else edited_rows.to_dict(orient="records")
            for row in rows_iter:
                ft = (row.get("file_type") or "").strip()
                q = (row.get("query") or "").strip()
                if ft and q:
                    qmap[ft] = q
            new_cfg = DownloadConfig(
                company=cfg.company, data_dir=Path(data_dir_str),
                gmail=GmailConfig(
                    scopes=scopes, credentials_path=Path(creds_path_str),
                    token_path=Path(token_path_str), query_by_type=qmap, link_prefixes=links,
                ),
                allowed_extensions=allowed or [".xlsx", ".xls", ".csv"],
                mark_as_read=mark, permit=cfg.permit, schedules=cfg.schedules, prepare=cfg.prepare,
            )
            Path(new_cfg.data_dir).mkdir(parents=True, exist_ok=True)
            save_config(new_cfg, get_config_path(cfg.company))
            st.success("Saved.")
        except Exception as e:
            st.error(str(e))

    if mkdirs_btn:
        try:
            Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
            Path(cfg.gmail.credentials_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.gmail.token_path).parent.mkdir(parents=True, exist_ok=True)
            st.success("Directories ensured.")
        except Exception as e:
            st.error(str(e))

    st.markdown("#### Create New Company")
    with st.form("create_company_form"):
        new_name = st.text_input("New company name", value="")
        create_btn = st.form_submit_button("Create Company")
    if create_btn and new_name.strip():
        try:
            name = new_name.strip()
            cfg_dict = new_company_config_template(name)
            cfg_path = get_config_path(name)
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            Path(f"secrets/{name}/google").mkdir(parents=True, exist_ok=True)
            Path(f"data/{name}/downloads").mkdir(parents=True, exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_dict, f, indent=2)
            st.success(f"Created {name}. Reload to see it in the selector.")
        except Exception as e:
            st.error(str(e))

    with st.expander("Danger Zone - Delete Company"):
        confirm = st.text_input("Type company name to confirm deletion", value="", key="admin_delete_confirm")
        if st.button("Delete Company", type="primary", disabled=(confirm.strip() != cfg.company), key="admin_delete_btn"):
            for p in [
                Path("companies") / cfg.company,
                Path(cfg.data_dir),
                Path("secrets") / cfg.company,
            ]:
                try:
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
            st.success("Deletion attempted. Refresh to update.")


# ---------------------------------------------------------------------------
# Settings tab
# ---------------------------------------------------------------------------

def _render_settings() -> None:
    gcfg = load_global_config()
    config_path = Path("config/global.json")
    current_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            current_config = json.load(f)

    with st.form("global_cfg_form"):
        permits_dir_str = st.text_input("Permits data dir", value=str(gcfg.permits_data_dir))
        austin_url = st.text_input("Austin permits base URL", value=gcfg.permit.austin_base_url)
        lookback = st.number_input("Default lookback hours", min_value=1, max_value=24 * 60,
                                   value=int(gcfg.permit.default_lookback_hours), step=1)
        st.markdown("**LLM Configuration**")
        deepseek_key = st.text_input(
            "DeepSeek API Key", value=current_config.get("deepseek_api_key", ""),
            type="password",
        )
        save_btn = st.form_submit_button("Save", type="primary")

    if save_btn:
        try:
            gcfg.permits_data_dir = Path(permits_dir_str)
            gcfg.permit.austin_base_url = austin_url
            gcfg.permit.default_lookback_hours = int(lookback)
            Path(gcfg.permits_data_dir).mkdir(parents=True, exist_ok=True)

            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_dict = current_config.copy()
            config_dict.update({
                "permits_data_dir": str(gcfg.permits_data_dir),
                "permit": {
                    "austin_base_url": gcfg.permit.austin_base_url,
                    "default_lookback_hours": gcfg.permit.default_lookback_hours,
                },
                "schedules": [
                    {"id": s.id, "enabled": s.enabled, "job": s.job,
                     "interval_minutes": s.interval_minutes, "params": s.params}
                    for s in gcfg.schedules
                ],
            })
            if deepseek_key.strip():
                config_dict["deepseek_api_key"] = deepseek_key.strip()
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)
            st.success("Saved global config.")
        except Exception as e:
            st.error(str(e))


# ---------------------------------------------------------------------------
# Master Data Import tab
# ---------------------------------------------------------------------------

def _to_parquet(df: pd.DataFrame, out_path: Path) -> None:
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = df2[c].astype("string").fillna("")
    table = pa.Table.from_pandas(df2, preserve_index=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


def _render_import(company: str) -> None:
    st.info("Upload historical files per type. We'll convert to Parquet for fast processing.")

    uploads = st.file_uploader(
        "Drop CSV/XLSX files here", type=["csv", "xlsx"],
        accept_multiple_files=True, key="admin_import_uploader",
    )
    if uploads:
        out_root = ROOT / "companies" / company / "parquet"
        out_root.mkdir(parents=True, exist_ok=True)
        results = []
        types = ["jobs", "customers", "memberships", "calls", "estimates", "invoices", "locations"]
        for up in uploads:
            try:
                name = up.name
                df = (pd.read_csv(up, dtype=str, keep_default_na=False)
                      if name.lower().endswith(".csv")
                      else pd.read_excel(up, dtype=str))
                t = next((t for t in types if t in name.lower()), "jobs")
                out = out_root / f"{t.capitalize()}.parquet"
                _to_parquet(df, out)
                results.append({"file": name, "type": t, "rows": len(df), "out": str(out)})
            except Exception as e:
                results.append({"file": up.name, "status": f"error: {e}"})
        if results:
            st.success("Uploaded and converted files")
            st.dataframe(results, width="stretch")


# ---------------------------------------------------------------------------
# Logs tab
# ---------------------------------------------------------------------------

def _render_logs(company: str, cfg) -> None:
    logs_dir = Path(cfg.data_dir) / "logs"
    logs_dir_alt = Path(cfg.data_dir).parent / "logs"
    limit = st.number_input("Max lines", min_value=50, max_value=5000, value=500, step=50, key="admin_log_limit")

    st.markdown("#### Change Dashboard")
    changes_df = load_all_changes([logs_dir, logs_dir_alt], int(limit) * 100)
    if not changes_df.empty:
        total = len(changes_df)
        updates = int((changes_df["change_type"] == "update_cell").sum())
        inserts = int((changes_df["change_type"] == "insert_row").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Changes", total)
        c2.metric("Updates", updates)
        c3.metric("Inserts", inserts)
        st.dataframe(changes_df.tail(500), width="stretch")
    else:
        st.info("No change logs found.")

    st.markdown("#### Download Log")
    log_path = logs_dir / "download_log.jsonl"
    entries = read_jsonl(log_path, int(limit)) or read_jsonl(logs_dir_alt / "download_log.jsonl", int(limit))
    if entries:
        st.write(f"{len(entries)} entries")
        st.dataframe(entries[-100:], width="stretch")
    else:
        st.info("No download log entries.")

    st.markdown("#### Maintenance")
    col1, col2 = st.columns(2)
    with col1:
        delete_btn = st.button("Delete All Company Logs", type="primary", key="admin_delete_logs")
    with col2:
        confirm = st.text_input("Type DELETE to confirm", value="", key="admin_delete_logs_confirm")
    if delete_btn and confirm.strip().upper() == "DELETE":
        count = 0
        for d in [logs_dir, logs_dir_alt]:
            if d.exists():
                for p in d.glob("*.jsonl"):
                    try:
                        p.unlink()
                        count += 1
                    except Exception:
                        pass
        st.success(f"Deleted {count} log files.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="DataHound Pro - Admin",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if UI:
        inject_custom_css()
        dh_page_header("Admin", "Company configuration, settings, data import, and logs")
    else:
        st.title("Admin")
        st.caption("Company configuration, settings, data import, and logs")

    company, cfg = select_company_config()
    if not company or not cfg:
        st.warning("Select a company to continue.")
        return

    tabs = st.tabs(["Companies", "Settings", "Data Import", "Logs"])

    with tabs[0]:
        _render_companies()
    with tabs[1]:
        _render_settings()
    with tabs[2]:
        _render_import(company)
    with tabs[3]:
        _render_logs(company, cfg)


if __name__ == "__main__":
    main()
