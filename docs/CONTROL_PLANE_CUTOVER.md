# Control Plane Cutover (Phase 5)

## Goal
Make DB the primary control-plane source when available, while keeping JSONL as compatibility/audit exports.

## Toggle
`DATAHOUND_CONTROL_PLANE_SOURCE=auto|db|json`

- `auto` (default): use DB when DAL is available, otherwise JSON
- `db`: force DB and fail fast when DAL is unavailable
- `json`: force JSON/read-model mode

## Current rollout
- Source selector helper added: `datahound/storage/control_plane.py`
- Reconciliation tool now reports selected source:
  - `datahound/devops/reconcile_review_notify.py`

## Next rollout steps
1. Update runtime readers to use selector helper (DB-first, JSON fallback)
2. Mark JSON logs as exports only in runbooks
3. Add scheduled reconciliation checks for drift during transition

## Drift check command
Use this in cron/CI after DB is enabled:

```bash
DATAHOUND_STORAGE_URL=postgresql+psycopg://... \
python -m datahound.devops.control_plane_drift_check --max-missing 0
```

Cron wrapper module:

```bash
DATAHOUND_STORAGE_URL=postgresql+psycopg://... \
python -m datahound.devops.run_control_plane_drift_check
```

Exit codes:
- `0`: drift within threshold
- `1`: drift above threshold
- `2`: DAL/repo configuration issue
