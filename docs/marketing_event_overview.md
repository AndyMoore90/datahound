# Marketing Guide: Event Signals and Workflow

## Cancellations

- **Historical scan**
  - Source: `Jobs.parquet`.
  - Rule: status equals `Canceled` or `Cancelled` (case-insensitive).

- **Recent file update**
  - When a job is cancelled and no matching record exists, the change processor appends a new entry to `recent_cancellations.parquet` (enriched with job metadata).
  - If the job is later completed or rescheduled, the related cancellation is archived with reason `job_completed` or `job_rescheduled` and removed from the active list.
  - Age rules drop cancellations older than their configured window (for example 30 days).

- **Google Sheet sync**
  - The export script reads `recent_cancellations.parquet`, joins customer context, writes a CSV, and updates the “Recent Cancellations” sheet.
  - New events are added, changed rows are updated, and missing rows are deleted (unless `--keep-in-sheet` is set).
  - Deleted rows with a customer name are copied into the `Invalid Events` worksheet; rows missing a customer name are skipped.

## Unsold Estimates

- **Historical scan**
  - Source: `Estimates.parquet`.
  - Rule: status in (`Dismissed`, `Open`) and summary does **not** contain `"This is an empty"`.
  - Deduplicates to the most recent estimate per customer/location.
  - Stored fields: estimate ID, customer ID, location ID, summary, creation date, total amount, status.

- **Recent file update**
  - When an estimate changes to a qualifying status, a new entry lands in `recent_unsold_estimates.parquet`.
  - If the estimate is later marked sold, the entry is archived with reason `estimate_sold`.
  - Age or activity rules can remove stale records if configured.

- **Google Sheet sync**
  - Pulls the parquet file, enriches with customer profile data, writes a CSV, and updates the “Recent Unsold Estimates” sheet.
  - Adds new rows, updates changed ones, and deletes missing ones (optionally keeping them if desired).
  - Deleted rows with customer names are copied to `Invalid Events`; nameless rows are skipped and logged.

## Lost Customers

- **Historical scan**
  - Sources: customers, calls, permits.
  - Rule: customer has McCullough permit history, then a competitor permit dated after McCullough’s last work, and no later McCullough return.
  - Requires both first and last contact dates in call history.
  - Stored fields: competitor used, permit timeline, first/last contact dates, permit counts, “lost” classification.

- **Recent file update**
  - Lost customers are tracked historically; the change processor primarily handles cleanup.
  - When a “lost” customer contacts us again, their recent entry is archived with reason `customer_contact`.

- **Google Sheet sync**
  - Enriches the lost-customer parquet with core metrics (RFM, tier, jobs, calls) and pushes rows to the “Recent Lost Customers” sheet.
  - Adds/updates active lost customers, removes ones no longer present, and files removed rows under `Invalid Events` if they contain a customer name.

## Overdue Maintenance

- **Historical scan**
  - Sources: jobs, locations, customers.
  - Rule: maintenance job type/class with last maintenance at least 12 months ago (within configured max range).
  - Evaluated for both customer and location entities.
  - Stored fields: months overdue, last maintenance date, severity band (medium/high/critical), plus customer/location details.

- **Recent file update**
  - Regularly recomputed; any customer/location exceeding the threshold is added to `recent_overdue_maintenance.parquet`.
  - If maintenance occurs within the recent window (e.g., last 30 days), the entry archives with reason `maintenance_completed`.
  - Age rules remove very old overdue records automatically.

- **Google Sheet sync**
  - Reads the overdue parquet, joins customer enrichment, writes a CSV, and updates the “Recent Overdue Maintenance” sheet.
  - Adds new overdue cases, updates existing ones, removes resolved entries, and logs deletions with customer names to `Invalid Events`.

