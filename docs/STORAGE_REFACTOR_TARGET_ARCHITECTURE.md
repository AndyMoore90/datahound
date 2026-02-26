# DataHound Storage Refactor — Target Architecture (v1)

## Goals
- Keep current delivery velocity.
- Improve consistency, reliability, and auditability.
- Avoid big-bang rewrite.
- Preserve Parquet strengths for analytics.

---

## 1) Target storage model (hybrid)

### A. Control Plane (Postgres)
Use Postgres for stateful/transactional data:
- scheduler tasks + runs
- pipeline runs/manifests
- event index/status lifecycle
- idempotency keys
- notification delivery state
- review gate state

### B. Data Plane (Parquet)
Keep Parquet for bulk datasets:
- customers/jobs/calls/estimates/invoices snapshots
- event fact outputs
- extracted profile artifacts

### C. Config Plane (JSON, versioned)
Keep human-editable config JSONs, but with strict schema/version:
- `config/global.json` (no secrets)
- `config/events/*.json`
- `config/extraction/*.json`

Secrets move to env/secret manager.

---

## 2) Proposed module layout

```text
datahound/
  storage/
    __init__.py
    dal.py                 # high-level façade used by services
    manifest.py            # run manifests + checksums
    schemas.py             # dataset schemas + versions
    io/
      parquet_io.py        # atomic parquet read/write helpers
      csv_io.py            # strict csv read/write helpers
      json_io.py           # strict json/jsonl helpers
    db/
      engine.py            # SQLAlchemy engine/session
      models.py            # ORM tables
      repos/
        scheduler_repo.py
        run_repo.py
        event_repo.py
        notification_repo.py
        review_repo.py
```

Rule: business services (`services/*`, `datahound/*`) should call `storage.dal` only, not raw pandas/file ops.

---

## 3) Database schema (initial)

## scheduler_tasks
- id (uuid, pk)
- task_key (text, unique)
- task_type (text)
- company (text)
- config_json (jsonb)
- schedule_type (text)
- schedule_expr (text)
- timezone (text)
- status (text)  -- active|paused|deleted
- created_at (timestamptz)
- updated_at (timestamptz)

## scheduler_runs
- id (uuid, pk)
- task_id (uuid, fk scheduler_tasks.id)
- run_id (text, unique)
- started_at (timestamptz)
- finished_at (timestamptz)
- success (bool)
- message (text)
- duration_ms (bigint)
- metadata_json (jsonb)

## pipeline_runs
- id (uuid, pk)
- run_id (text, unique)
- company (text)
- pipeline_name (text)      -- transcript_pipeline, event_upload, etc
- stage (text)
- status (text)             -- running|success|failed
- input_manifest_json (jsonb)
- output_manifest_json (jsonb)
- error_json (jsonb)
- started_at (timestamptz)
- finished_at (timestamptz)

## dataset_versions
- id (uuid, pk)
- dataset_name (text)
- company (text)
- version_tag (text)
- schema_version (text)
- file_path (text)
- row_count (bigint)
- checksum (text)
- produced_by_run_id (text)
- created_at (timestamptz)

## events_index
- id (uuid, pk)
- event_id (text, unique)
- company (text)
- event_type (text)
- entity_type (text)
- entity_id (text)
- severity (text)
- status (text)            -- active|resolved|archived
- first_seen_at (timestamptz)
- last_seen_at (timestamptz)
- source_file (text)
- details_json (jsonb)

## review_gates
- id (uuid, pk)
- task_id (text)
- pr_number (int)
- repo (text)
- branch (text)
- ci_passed (bool)
- codex_passed (bool)
- claude_passed (bool)
- gemini_passed (bool)
- branch_up_to_date (bool)
- screenshots_included (bool)
- mode (text)              -- strict|soft
- ready (bool)
- updated_at (timestamptz)

## notifications
- id (uuid, pk)
- task_id (text)
- channel (text)
- target (text)
- message_type (text)      -- ready|failed|info
- payload_json (jsonb)
- sent_at (timestamptz)
- status (text)            -- sent|failed
- provider_message_id (text)

## idempotency_keys
- key (text, pk)
- scope (text)
- run_id (text)
- created_at (timestamptz)
- expires_at (timestamptz)

---

## 4) Parquet zone conventions

```text
data/
  <company>/
    bronze/       # raw immutable ingests (append only)
    silver/       # normalized curated tables
    gold/         # feature/business outputs
    manifests/    # run + dataset manifests
```

Naming:
- `dataset=<name>/dt=YYYY-MM-DD/part-*.parquet` (partition-ready)

---

## 5) DAL contract (example)

## Scheduler
- `create_task(task)`
- `update_task(task)`
- `list_active_tasks()`
- `record_task_run(run)`

## Pipeline/run metadata
- `start_pipeline_run(...) -> run_id`
- `finish_pipeline_run(run_id, status, manifests, error)`
- `register_dataset_version(...)`

## Events
- `upsert_events_index(events, run_id)`
- `resolve_missing_events(current_event_ids, event_type, company)`

## Review/notify
- `upsert_review_gate(task_id, pr, checks...)`
- `mark_ready(task_id)`
- `record_notification(...)`

---

## 6) Refactor phases

## Phase 0 (safe prep)
- Add `datahound/storage/` package with no behavior change.
- Add JSON schema validation for config files.
- Remove secrets from `config/global.json`.

## Phase 1 (control-plane migration)
- Move scheduler persistence from JSON files to Postgres tables.
- Keep existing JSON logs temporarily for compatibility.

## Phase 2 (run manifesting)
- Add pipeline run_id and dataset manifests.
- All services write run metadata via DAL.

## Phase 3 (event lifecycle centralization)
- Keep parquet master files for facts.
- Move canonical active/resolved index to `events_index` table.

## Phase 4 (gate + notify centralization)
- Store review gate states + notification receipts in DB.
- Keep swarm JSON as cache/read model if desired.

## Phase 5 (cleanup)
- Deprecate duplicate JSON state files.
- Keep JSON exports as optional audit artifacts only.

---

## 7) Immediate engineering tasks for swarm

1. `feat/storage-dal-scaffold`
   - create `datahound/storage/*` package + interfaces
2. `feat/postgres-engine-and-models`
   - add SQLAlchemy engine + initial tables
3. `feat/scheduler-persistence-db`
   - migrate `datahound/scheduler/persistence.py` to DB-backed repo
4. `feat/pipeline-run-manifests`
   - add run_id + manifest writing utilities
5. `feat/review-gate-db-sync`
   - sync swarm gate outputs into DB `review_gates`

---

## 8) Non-negotiable guardrails
- No plaintext API keys in repo files.
- Atomic writes for all file outputs (tmp + rename).
- Schema version tags on every curated dataset.
- Idempotency key checks on external side effects.
- Keep migration reversible for first 2 phases.
