import hashlib
import json
import time
import sys
from pathlib import Path
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._shared import ensure_root_on_path, select_company_config, get_config_path

ensure_root_on_path()

from datahound.download.types import load_config, PrepareTypeRule
from datahound.download.gmail import GmailDownloader
from datahound.prepare.engine import prepare_latest_files
from datahound.prepare.validate import validate_against_master, write_validation_log, ValidationIssue
from datahound.upsert.engine import find_latest_prepared
from apps.components.scheduler_ui import (
    render_schedule_config, render_task_manager,
    create_scheduled_task, render_scheduler_status,
)
from datahound.scheduler import TaskType, ScheduleType

try:
    from apps.components.ui_components import (
        inject_custom_css, dh_page_header, dh_alert, dh_breadcrumbs, dh_path_with_copy,
    )
    UI = True
except ImportError:
    UI = False

try:
    from apps.components.upsert_ui import render as render_upsert
except ImportError:
    render_upsert = None

try:
    from datahound.profiles.enhanced_core_data import EnhancedCustomerCoreDataBuilder
    from datahound.profiles.types import ProfileBuildConfig, ProfileBuildMode
    PROFILES_AVAILABLE = True
except ImportError:
    PROFILES_AVAILABLE = False

from datetime import time as datetime_time


# ---------------------------------------------------------------------------
# Download tab
# ---------------------------------------------------------------------------

def _render_download(company: str, cfg) -> None:
    creds_ok = Path(cfg.gmail.credentials_path).exists()
    token_ok = Path(cfg.gmail.token_path).exists()
    st.write(f"Data dir: `{cfg.data_dir}`")
    st.write(f"Credentials: {'OK' if creds_ok else 'Missing'} | Token: {'OK' if token_ok else 'Missing'}")

    all_types = sorted(list(cfg.gmail.query_by_type.keys()))
    if not all_types:
        st.info("No file types defined in this company's config.")

    selected_types = st.multiselect("File types", all_types, default=all_types, key="dl_file_types")
    archive = st.checkbox("Archive existing files before downloading", value=False, key="dl_archive")
    dedup_after = st.checkbox("Delete duplicate files after download", value=False, key="dl_dedup")
    mark_as_read = st.checkbox("Mark matched emails as read", value=cfg.mark_as_read, key="dl_mark_read")

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("Run Download", type="primary", key="dl_run")
    with col2:
        mark_btn = st.button("Mark All Read (no download)", key="dl_mark_btn")

    if run_btn:
        try:
            cfg.mark_as_read = mark_as_read
            downloader = GmailDownloader(cfg)
            if archive:
                downloader.archive_existing_files()
            with st.spinner("Downloading..."):
                results = downloader.run(selected_types)
            st.success("Download complete")
            st.json(results)
            if dedup_after:
                base = Path(cfg.data_dir)
                removed = []
                for t, files in (results or {}).items():
                    if not files:
                        continue
                    seen: dict[str, Path] = {}
                    stamped = sorted(
                        [base / f for f in files if (base / f).exists()],
                        key=lambda p: p.stat().st_mtime, reverse=True,
                    )
                    for p in stamped:
                        try:
                            h = hashlib.sha256(p.read_bytes()).hexdigest()
                        except Exception:
                            continue
                        if h in seen:
                            try:
                                p.unlink()
                                removed.append(str(p.name))
                            except Exception:
                                pass
                        else:
                            seen[h] = p
                if removed:
                    st.warning(f"Removed {len(removed)} duplicate files")
        except Exception as e:
            st.error(str(e))

    if mark_btn:
        try:
            downloader = GmailDownloader(cfg)
            with st.spinner("Marking as read..."):
                summary = downloader.mark_all_as_read(selected_types)
            st.success("Done")
            st.json(summary)
        except Exception as e:
            st.error(str(e))


# ---------------------------------------------------------------------------
# Prepare tab
# ---------------------------------------------------------------------------

def _render_prepare(company: str, cfg) -> None:
    if not cfg.prepare:
        st.info("Prepare configuration not found.")
        return
    prep = cfg.prepare
    schema_dir = Path(prep.tables_dir).parent / "parquet"
    all_types = list(prep.file_type_to_master.keys())

    sel = st.multiselect("File types to prepare", all_types, default=all_types, key="prep_types")
    write_parquet = st.checkbox("Write Parquet", value=True, key="prep_parquet")
    write_csv = st.checkbox("Write CSV", value=False, key="prep_csv")

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("Run Prepare", key="prep_run")
    with col2:
        validate_btn = st.button("Validate", key="prep_validate")

    if run_btn:
        try:
            results = prepare_latest_files(cfg, selected_types=sel, write_csv=bool(write_csv), write_parquet=bool(write_parquet))
            st.success("Prepared")
            st.json({k: str(v) for k, v in results.items()})
        except Exception as e:
            st.error(str(e))

    if validate_btn:
        try:
            from datahound.prepare.engine import read_master_columns_any
            rows = []
            for ftype, master in prep.file_type_to_master.items():
                prepared_path = find_latest_prepared(Path(cfg.data_dir), ftype, prefer_parquet=True)
                if not prepared_path:
                    rows.append({"type": ftype, "prepared": "(not found)", "status": "no_file", "issues": ""})
                    continue
                master_cols = read_master_columns_any(prep.tables_dir, master, prefer_parquet=True)
                issues = validate_against_master(prepared_path, master_cols)
                write_validation_log(Path(cfg.data_dir), cfg.company, ftype, prepared_path.name, issues)
                status = "ok" if not issues else "issues"
                rows.append({
                    "type": ftype,
                    "prepared": prepared_path.name,
                    "status": status,
                    "issues": "; ".join([f"{i.kind}: {i.detail}" for i in issues]),
                })
            st.dataframe(rows, width="stretch")
            st.success("Validation complete")
        except Exception as e:
            st.error(str(e))

    with st.expander("Preview Prepared Output"):
        prev_type = st.selectbox("Type", all_types, key="prep_preview_type")
        if st.button("Preview", key="prep_preview_btn"):
            try:
                newest = find_latest_prepared(Path(cfg.data_dir), prev_type, prefer_parquet=True)
                if newest is None:
                    st.warning("No source file found for preview.")
                else:
                    if newest.suffix.lower() == ".parquet":
                        df = pd.read_parquet(newest)
                    elif newest.suffix.lower() == ".csv":
                        df = pd.read_csv(newest)
                    else:
                        df = pd.read_excel(newest)
                    st.dataframe(df.head(20), width="stretch")
            except Exception as e:
                st.error(str(e))


# ---------------------------------------------------------------------------
# Core Data tab
# ---------------------------------------------------------------------------

def _render_core_data(company: str, config) -> None:
    if not PROFILES_AVAILABLE:
        st.error("Profile builder not available. Check imports.")
        return

    parquet_dir = Path("companies") / company / "parquet"
    data_dir = Path("data") / company
    core_data_file = parquet_dir / "customer_core_data.parquet"
    core_data_exists = core_data_file.exists()

    col1, col2, col3 = st.columns(3)
    with col1:
        if core_data_exists:
            try:
                existing_df = pd.read_parquet(core_data_file)
                st.metric("Core Data Records", f"{len(existing_df):,}")
                last_modified = datetime.fromtimestamp(core_data_file.stat().st_mtime)
                st.metric("Last Updated", last_modified.strftime("%Y-%m-%d %H:%M"))
            except Exception:
                st.metric("Core Data Records", "Error")
        else:
            st.metric("Core Data Records", "Not Created")
    with col2:
        master_files = ["Customers.parquet", "Jobs.parquet", "Estimates.parquet", "Invoices.parquet"]
        available = sum(1 for f in master_files if (parquet_dir / f).exists())
        st.metric("Master Files", f"{available}/{len(master_files)}")
    with col3:
        permit_exists = Path("global_data/permits").exists()
        demo_exists = Path("global_data/demographics/demographics.parquet").exists()
        enhancements = []
        if permit_exists:
            enhancements.append("Permits")
        if demo_exists:
            enhancements.append("Demographics")
        st.metric("Enhancements", f"{len(enhancements)}")

    include_rfm = st.checkbox("Include RFM Analysis", value=True, key="cd_rfm")
    include_permits = st.checkbox("Include Permit Matching", value=permit_exists, key="cd_permits")
    include_demographics = st.checkbox("Include Demographics", value=demo_exists, key="cd_demo")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Core Data", type="primary", key="cd_create"):
            _build_core_data(company, parquet_dir, data_dir, ProfileBuildMode.ALL_CUSTOMERS,
                             include_rfm=include_rfm, include_permits=include_permits,
                             include_demographics=include_demographics)
    with col2:
        if st.button("Refresh Core Data", disabled=not core_data_exists, key="cd_refresh"):
            _build_core_data(company, parquet_dir, data_dir, ProfileBuildMode.NEW_CUSTOMERS_ONLY,
                             include_rfm=include_rfm, include_permits=include_permits,
                             include_demographics=include_demographics)


def _build_core_data(company: str, parquet_dir: Path, data_dir: Path,
                     mode: ProfileBuildMode, **kwargs) -> None:
    progress = st.progress(0)
    status = st.empty()
    try:
        status.text("Initializing...")
        builder = EnhancedCustomerCoreDataBuilder(company, parquet_dir, data_dir)
        build_config = ProfileBuildConfig(
            mode=mode,
            include_rfm=kwargs.get("include_rfm", True),
            include_demographics=kwargs.get("include_demographics", True),
            include_permits=kwargs.get("include_permits", True),
            include_marketable=True,
            include_segments=True,
        )
        progress.progress(0.1)
        status.text("Building customer profiles...")
        start = time.time()
        result = builder.build_enhanced_customer_profiles(build_config)
        duration = time.time() - start
        progress.progress(1.0)
        status.text("Complete!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Processed", f"{result.total_customers_processed:,}")
        c2.metric("Created", f"{result.new_profiles_created:,}")
        c3.metric("Time", f"{duration:.1f}s")
        st.success(f"Built {result.new_profiles_created:,} profiles in {duration:.1f}s")
    except Exception as e:
        progress.progress(0)
        status.text("Error")
        st.error(str(e))


# ---------------------------------------------------------------------------
# Automation tab
# ---------------------------------------------------------------------------

def _render_automation(company: str) -> None:
    st.markdown("Manage scheduled tasks for all pipeline steps.")

    for task_type, label in [
        (TaskType.DOWNLOAD, "Download Files"),
        (TaskType.PREPARE, "Prepare Files"),
        (TaskType.INTEGRATED_UPSERT, "Update Masters"),
        (TaskType.CREATE_CORE_DATA, "Build Profiles"),
    ]:
        with st.expander(f"{label} Schedule"):
            render_task_manager(
                task_type=task_type,
                company=company,
                task_name=label,
                task_description=f"Automated {label.lower()}",
                key_context="pipeline_auto",
            )

    st.markdown("---")
    render_scheduler_status(key_context="pipeline_auto")


# ---------------------------------------------------------------------------
# Pipeline Overview tab
# ---------------------------------------------------------------------------

def _render_overview(company: str, cfg) -> None:
    st.markdown(
        "The data pipeline processes your company's data through four sequential stages: "
        "**Download** raw files from Gmail, **Prepare** them into a standard format, "
        "**Update Masters** to merge changes into your master data files, and "
        "**Build Profiles** to create enriched customer profiles with RFM analysis."
    )

    data_dir = Path("data") / company
    parquet_dir = Path("companies") / company / "parquet"

    steps = [
        ("Download", data_dir / "logs" / "pipeline" / "download_log.jsonl"),
        ("Prepare", data_dir / "logs" / "pipeline" / "prepare_log.jsonl"),
        ("Update Masters", data_dir / "logs" / "integrated_upsert_log.jsonl"),
        ("Build Profiles", data_dir / "logs" / "customer_profile_build_log.jsonl"),
    ]

    cols = st.columns(len(steps))
    for col, (name, log_path) in zip(cols, steps):
        with col:
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    last = json.loads(lines[-1]) if lines else {}
                    ts = last.get("timestamp", last.get("ts", ""))[:16].replace("T", " ")
                    st.metric(name, f"{len(lines)} runs", ts)
                except Exception:
                    st.metric(name, "Log error")
            else:
                st.metric(name, "No runs yet")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="DataHound Pro - Data Pipeline",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if UI:
        inject_custom_css()
        dh_page_header("Data Pipeline", "Download, prepare, update, and build customer profiles")
    else:
        st.title("Data Pipeline")
        st.caption("Download, prepare, update, and build customer profiles")

    company, cfg = select_company_config()
    if not company or not cfg:
        st.warning("Select a company to continue.")
        return

    if UI:
        dh_alert(f"Active Company: {company}", "success")

    tabs = st.tabs([
        "Overview",
        "Download",
        "Prepare",
        "Update Masters",
        "Build Profiles",
        "Automation",
    ])

    with tabs[0]:
        _render_overview(company, cfg)
    with tabs[1]:
        _render_download(company, cfg)
    with tabs[2]:
        _render_prepare(company, cfg)
    with tabs[3]:
        if render_upsert:
            render_upsert(company, cfg)
        else:
            st.error("Upsert module not available.")
    with tabs[4]:
        _render_core_data(company, cfg)
    with tabs[5]:
        _render_automation(company)


if __name__ == "__main__":
    main()
