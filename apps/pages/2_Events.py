import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._shared import ensure_root_on_path, select_company_config

ensure_root_on_path()

try:
    from apps.components.ui_components import inject_custom_css, dh_page_header, dh_alert
    UI = True
except ImportError:
    UI = False

try:
    from apps.components.event_detection_ui import render as render_detection
except ImportError:
    render_detection = None

try:
    from apps.components.extraction_ui import render as render_extraction
except ImportError:
    render_extraction = None

from apps.components.event_dashboards import render_all_dashboards


def main() -> None:
    st.set_page_config(
        page_title="DataHound Pro - Events",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if UI:
        inject_custom_css()
        dh_page_header("Events", "Detect business events and track their performance")
    else:
        st.title("Events")
        st.caption("Detect business events and track their performance")

    company, cfg = select_company_config()
    if not company or not cfg:
        st.warning("Select a company to continue.")
        return

    if UI:
        dh_alert(f"Active Company: {company}", "success")

    tabs = st.tabs(["Event Detection", "Custom Extraction", "Performance Dashboards"])

    with tabs[0]:
        st.markdown(
            "Configure and run scans against your master data to detect business events "
            "such as cancellations, unsold estimates, overdue maintenance, lost customers, "
            "and aging systems."
        )
        if render_detection:
            render_detection(company, cfg)
        else:
            st.error("Event detection module not available. Check imports.")

    with tabs[1]:
        st.markdown(
            "Extract and enrich recent event data with configurable time and numeric filters. "
            "Results can be enriched with customer core data, demographics, and segmentation."
        )
        if render_extraction:
            render_extraction(company, cfg)
        else:
            st.error("Extraction module not available. Check imports.")

    with tabs[2]:
        st.markdown(
            "Track how detected events convert into revenue. Each dashboard shows "
            "the event definition, conversion formula, and performance metrics."
        )
        render_all_dashboards(company)


if __name__ == "__main__":
    main()
