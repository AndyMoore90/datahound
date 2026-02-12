# DataHound Pro

DataHound Pro is a business intelligence platform built for HVAC companies. It automatically processes customer data, detects marketing opportunities, and tracks how those opportunities convert into revenue.

## What It Does

The system solves a simple problem: HVAC companies generate a lot of customer data (jobs, estimates, calls, invoices, memberships) but don't have a way to automatically spot the customers who are most likely to need service. DataHound Pro finds those customers and tracks whether follow-up efforts actually work.

### The Five Business Events

DataHound detects five types of marketing opportunities:

| Event | What It Finds | Why It Matters |
|-------|--------------|----------------|
| **Cancellations** | Jobs that were scheduled then canceled | Chance to win back the customer |
| **Unsold Estimates** | Estimates that never converted to sales | Potential revenue that didn't close |
| **Overdue Maintenance** | Customers who haven't had service in 12+ months | At risk of switching to competitors |
| **Lost Customers** | Customers now using competitors (detected via public permit records) | Revenue going to competitors |
| **Second Chance Leads** | Customers who called wanting service but didn't book (detected via AI call transcript analysis) | High-intent leads that slipped through |

Each event has an automatic removal system - for example, a cancellation is removed from the active list if the customer books a new job, calls back, or texts. This keeps the marketing lists focused on actionable opportunities.

### Performance Tracking

For each event type, dashboards show:
- How many events were detected
- How many converted into actual business
- The conversion rate and trends over time

The Second Chance Leads dashboard connects to Google Sheets for live data and calculates period-by-period conversion rates with charts.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│                                                                 │
│   Gmail ──► Download ──► Prepare ──► Update Masters ──► Build   │
│   Inbox      raw files    normalize    merge changes    customer │
│              (XLSX/CSV)   to schema    + audit trail    profiles │
│                                                                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EVENT DETECTION                            │
│                                                                 │
│   Master Data ──► Scan for Events ──► Marketing Lists           │
│   (Customers,      (5 event types)     (active opportunities    │
│    Jobs, etc.)                          with auto-removal)      │
│                                                                 │
│   Permits ──► Lost Customer Detection                           │
│   (Austin)     (competitor activity at customer addresses)      │
│                                                                 │
│   Call Transcripts ──► AI Analysis ──► Second Chance Leads      │
│   (from Google)         (DeepSeek)     (unbooked service calls) │
│                                                                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE DASHBOARDS                        │
│                                                                 │
│   Each event type:  Total Detected ──► Converted ──► Rate       │
│                                                                 │
│   Home Dashboard:   KPIs, customer tiers, revenue opportunities │
│   Pipeline Monitor: Real-time status of all processing stages   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## App Structure

```
datahound_pro/
│
├── apps/                          ← Streamlit web interface
│   ├── Home.py                       Main dashboard (run this)
│   ├── _shared.py                    Shared utilities
│   ├── pages/
│   │   ├── 1_Data_Pipeline.py        Download → Prepare → Update → Build Profiles
│   │   ├── 2_Events.py               Event detection + extraction + performance
│   │   ├── 3_Pipeline_Monitor.py     Real-time pipeline health monitoring
│   │   ├── 4_Permits.py              Austin building permit management
│   │   └── 5_Admin.py                Companies, settings, data import, logs
│   └── components/
│       ├── ui_components.py          Reusable UI widgets
│       ├── scheduler_ui.py           Task scheduling interface
│       ├── event_detection_ui.py     Historical event configuration
│       ├── upsert_ui.py              Master data update interface
│       ├── extraction_ui.py          Custom data extraction interface
│       └── event_dashboards.py       Performance dashboards for all 5 events
│
├── datahound/                     ← Core business logic (Python package)
│   ├── download/                     Gmail attachment downloader
│   ├── prepare/                      Data normalization engine
│   ├── upsert/                       Master data merge with change tracking
│   ├── profiles/                     Customer profile builder (RFM, segments)
│   ├── events/                       Event detection and processing
│   ├── extract/                      Configurable data extraction
│   ├── scheduler/                    Background task scheduling
│   └── dashboard/                    Log aggregation and metrics
│
├── config/                        ← Configuration files
│   ├── global.json                   Global settings (API keys, paths)
│   ├── events/                       Per-company event detection rules
│   └── extraction/                   Per-company extraction configs
│
├── companies/                     ← Per-company master data
│   └── {Company Name}/
│       ├── config.json               Company configuration
│       └── parquet/                  Master parquet files (Customers, Jobs, etc.)
│
├── data/                          ← Runtime data and logs
│   └── {Company Name}/
│       ├── downloads/                Raw downloaded files
│       ├── events/master_files/      Detected event master files
│       ├── second_chance/            Second chance leads data
│       ├── sms_exports/              SMS activity export files
│       ├── recent_events/            Recent event tracking
│       └── logs/                     Pipeline and audit logs
│
├── services/                      ← Standalone background services
│   ├── transcript_pipeline.py        AI second chance lead detection
│   ├── event_upload.py               Google Sheets event sync
│   └── sms_sheet_sync.py             SMS activity data download
│
├── secrets/                       ← Google OAuth credentials (not committed)
├── start_automation.py            ← CLI automation service
└── requirements.txt               ← Python dependencies
```

## Running the App

**Start the web interface:**
```bash
streamlit run apps/Home.py
```

Everything else is managed from the UI. Go to **Admin > Services** and click
**"Start All Services"** to launch the scheduler with all background services
(transcript pipeline, event upload, SMS sync) at their default intervals.

You can also configure each service individually, change intervals, and run
any service once on demand — all from the same page.

### Advanced: CLI Mode

If you prefer terminal control, the services can still be run standalone:

```bash
python start_automation.py                                     # pipeline scheduler
python services/transcript_pipeline.py --interval-minutes 180  # AI lead detection
python services/event_upload.py --interval 20                  # Google Sheets sync
python services/sms_sheet_sync.py --interval-minutes 30        # SMS data download
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_CREDS` | Path to Google service account credentials JSON |
| `SECOND_CHANCE_SHEET_ID` | Google Sheets ID for second chance leads data |

These can be set in a `.env` file in `data/{company}/recent_events/`.

## Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt`

Install with:
```bash
pip install -r requirements.txt
```
