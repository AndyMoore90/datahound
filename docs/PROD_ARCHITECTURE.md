# Production Architecture Spec (Phase 0)

Date: 2026-02-26
Owner: DataHound Pro
Scope: Phase 0 — documentation + planning only (no functional changes)

## Goals

- Define a production target architecture that can evolve from the current monorepo without breaking existing workflows.
- Clarify bounded contexts and service contracts for incremental extraction.
- Provide a migration map, acceptance criteria, and rollback plan for safe delivery.

## Non-Goals (Phase 0)

- No runtime behavior changes.
- No data migrations executed.
- No deployment tooling changes.

## Current State (Summary)

- Monorepo with Streamlit UI, scheduler/automation, and data pipeline logic.
- Filesystem-based storage in `companies/` (master parquet) and `data/` (runtime data).
- Background services executed via the scheduler and standalone scripts.

## Production Target Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                Web UI (Streamlit)            │
                 │  Dashboards • Admin • Pipeline Monitor       │
                 └──────────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                             API / Orchestration                            │
│  Scheduler • Job Queue • Service Registry • Auth • Config                  │
└───────────────────────────────────────────────────────────────────────────┘
      │                 │                    │                    │
      ▼                 ▼                    ▼                    ▼
┌────────────┐   ┌────────────┐     ┌────────────────┐    ┌───────────────┐
│ Ingestion  │   │ Preparation│     │ Master Data    │    │ Event Engine  │
│ (Gmail,    │   │ & Normalize│     │ (Upsert/Profiles│    │ (5 event types│
│ Sheets)    │   │            │     │ /RFM segments) │    │ + AI leads)   │
└────────────┘   └────────────┘     └────────────────┘    └───────────────┘
      │                 │                    │                    │
      ▼                 ▼                    ▼                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                               Data Storage                                 │
│  Object Store (raw/prepared) • Warehouse (parquet tables) • Logs           │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                             Observability                                  │
│  Structured logs • Task metrics • Audit trails • Alerts                    │
└───────────────────────────────────────────────────────────────────────────┘
```

### Key Characteristics

- **Incremental extraction**: pipeline components can be isolated into services without changing data contracts.
- **Idempotent jobs**: every scheduled task should be rerunnable with deterministic outputs.
- **Central logging**: consistent log records for task execution and audit trails.
- **Separation of concerns**: UI reads from persisted data artifacts; compute is handled by jobs/services.

## Bounded Contexts

1. **Ingestion**: Fetch external inputs (Gmail attachments, Sheets, permits).
2. **Preparation**: Normalize raw inputs to canonical schemas.
3. **Master Data**: Upsert engine + profile generation + segmentation.
4. **Event Detection**: Generate marketing events and remove invalidated items.
5. **Lead Intelligence**: AI call transcript analysis and second-chance leads.
6. **Automation/Scheduler**: Orchestrate tasks, track execution, enforce retries.
7. **Admin & Configuration**: Company configs, extraction rules, service schedules.
8. **Observability**: Central logs, audit records, metrics, alerts.

## Service Contracts (Phase 0 target)

### 1) Scheduler → Task Services

- **Input**: `ScheduledTask` payload (task_id, task_type, company, settings, timeout).
- **Output**: `ExecutionRecord` with success flag, timestamps, log_file, error tail.
- **Behavior**: Must be idempotent and write log output to `central_logging`.

### 2) Ingestion → Preparation

- **Input**: Raw download files in `data/{company}/downloads`.
- **Output**: Prepared data files (CSV/Parquet) in `data/{company}/prepared`.
- **Contract**: Schema normalization per config; errors recorded with file-level context.

### 3) Preparation → Master Data Upsert

- **Input**: Prepared files for supported types (customers, jobs, estimates, etc.).
- **Output**: Parquet tables in `companies/{company}/parquet` with change tracking.
- **Contract**: Record `total_changes`, `new_records`, `updated_records`.

### 4) Master Data → Event Detection

- **Input**: Parquet master tables + event configs.
- **Output**: Event master files in `data/{company}/events/master_files`.
- **Contract**: Each event file contains `event_type`, `entity_id`, `detected_at`.

### 5) Event Detection → Dashboards

- **Input**: Event master files + KPI aggregates.
- **Output**: Read-only dashboard views (Streamlit).
- **Contract**: Dashboard reads from materialized artifacts; no hidden compute.

## Migration Map (Phase 0 → Phase 2)

### Phase 0 (now)

- Document target architecture and contracts.
- Standardize task execution logging and audit output (already present).
- Avoid any runtime changes.

### Phase 1 (incremental extraction)

- Externalize scheduler into a service boundary (start with job registry + status API).
- Introduce a stable storage abstraction (object store or mounted volume).
- Define explicit config schema versions.

### Phase 2 (production hardening)

- Deploy isolated services for ingestion, prep, and event detection.
- Add queue-backed execution with retries and dead-letter tracking.
- Implement observability dashboards + alerts for task failures.

## Acceptance Criteria

- A single doc exists that defines target architecture, bounded contexts, and service contracts.
- Migration map includes Phase 0–2 with incremental steps and sequencing.
- Rollback plan defined for any future migration execution.
- No functional changes in this phase (docs-only).

## Rollback Plan (for future phases)

- Maintain versioned configs and keep old job paths available.
- Preserve existing filesystem layout as the source of truth until new services prove parity.
- Use feature flags to gate any new execution paths.
- Roll back by:
  1. Disabling new scheduler/service routes.
  2. Re-enabling existing scripts and UI paths.
  3. Restoring previous config version.
  4. Verifying output parity with prior artifacts.

## Open Questions

- Which storage system is preferred for raw/prepared artifacts (local disk vs object storage)?
- Do we need a formal API for event uploads or can we continue with file-based contracts?
- What are the SLA expectations for each pipeline stage?
