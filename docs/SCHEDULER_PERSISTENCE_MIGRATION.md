# Scheduler Persistence Migration

Phase 2 moves the DataHound scheduler state out of JSON flat files and into the
Postgres-backed storage DAL found in `datahound.storage`. This document
summarizes how to enable the database path, what happens to existing data, and
how to roll back if needed.

## Enabling the Postgres backend

1. **Provide a connection URL** via `DATAHOUND_STORAGE_URL` (or `DATABASE_URL`).
   The connection must be reachable from the scheduler process and should point
   at the `datahound.storage` Postgres schema.
2. Start (or restart) the scheduler. `SchedulerPersistence` now calls
   `datahound.storage.bootstrap.get_storage_dal_from_env()` on construction and
   wires the scheduler repository plus tables automatically.  Metadata tables
   are created on demand using SQLAlchemy models.
3. Optional: set `DATAHOUND_SCHEDULER_TZ` if you need the stored timezone label
   to differ from the default `UTC`.

No other code changes or configuration flags are required; the scheduler keeps
its existing public APIs.

## Automatic migration + backups

* On the first launch with `DATAHOUND_STORAGE_URL` configured, the DAL is
  bootstrapped. If the `scheduler_tasks` table is empty but the JSON snapshot
  at `data/scheduler/scheduled_tasks.json` exists, every task is serialized into
  a `SchedulerTaskRecord` and inserted. This is idempotent and safe to re-run.
* After each DB write (create/update/delete) and on every task listing, the JSON
  snapshot is refreshed. This keeps a ready-to-use rollback artifact at
  `data/scheduler/scheduled_tasks.json`.
* Task execution history records are duplicated: they are inserted into
  `scheduler_runs` *and* appended to the existing
  `logging/scheduler/task_history.jsonl` file so dashboards that tail the file
  continue to work.

## Rollback / failure handling

* If the DAL raises an exception (connection errors, migrations missing, etc.)
  the persistence layer prints the failure, disables the DB path for that
  process, and falls back to the JSON files automatically.
* To intentionally roll back, simply unset `DATAHOUND_STORAGE_URL` (or set it to
  an empty value) and restart the scheduler. The JSON snapshot becomes
  authoritative again thanks to the continuous backups.
* `clear_old_history()` trims both the DB (`scheduler_runs`) and file history so
  clean-up behaves the same regardless of backend.

## Testing + verification

* Unit tests live in `tests/test_scheduler_persistence.py` and exercise the file
  fallback, DB-backed flow, and bootstrap migration behavior.
* Run them with:

  ```bash
  .venv/bin/python -m unittest tests.test_scheduler_persistence
  ```

* To verify the migration in an environment, run the scheduler once with the DB
  URL configured, confirm records exist in `scheduler_tasks`, and validate the
  JSON backup still contains the same number of tasks for safety.
