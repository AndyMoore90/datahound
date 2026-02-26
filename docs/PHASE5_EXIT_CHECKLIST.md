# Phase 5 Exit Checklist (Control-Plane Cutover)

Use this checklist to declare Phase 5 complete.

## Required settings
- [ ] `DATAHOUND_CONTROL_PLANE_SOURCE=db`
- [ ] `DATAHOUND_STORAGE_URL` (or `DATABASE_URL`) set in runtime env

## Required checks
- [ ] `python -m datahound.preflight` passes
- [ ] `python -m datahound.devops.control_plane_readiness` returns exit code 0
- [ ] `python -m datahound.devops.control_plane_drift_check --max-missing 0` returns exit code 0
- [ ] `python -m datahound.devops.phase5_exit_check` returns exit code 0 (aggregated gate)

## Dashboard data-source checks
- [ ] Pipeline Monitor loads without errors
- [ ] `pipeline_runs` stage present from DB-backed source
- [ ] `review_notify` stage present from DB-backed source
- [ ] `control_plane_overview` stage present and non-empty

## Operational checks
- [ ] swarm auto-merge run records review gate state in DB
- [ ] swarm notification sync writes idempotent notification rows
- [ ] drift check scheduled (cron/CI) using `python -m datahound.devops.run_control_plane_drift_check`

## Rollback plan verified
- [ ] fallback mode documented: `DATAHOUND_CONTROL_PLANE_SOURCE=json`
- [ ] JSON export/read-model paths retained for compatibility during rollback window

## Sign-off
- [ ] Engineering sign-off
- [ ] Operations sign-off
- [ ] Date recorded
