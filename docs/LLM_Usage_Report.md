# LLM Usage Report: Datahound Pro

This report documents how Large Language Models (LLMs) are used across the Datahound Pro codebase.

## Summary

| Process | Model | API Provider | Purpose |
|---------|-------|--------------|---------|
| Second Chance Lead Analysis | deepseek-reasoner | DeepSeek (OpenAI-compatible) | Analyze call transcripts to identify customers who requested service but did not book |
| Aging Systems (Job History) | deepseek-chat | DeepSeek (OpenAI-compatible) | Determine HVAC system age from job histories |
| Aging Systems (YAML Profiles) | deepseek-chat | DeepSeek (OpenAI-compatible) | Analyze permit + job history for locations with structured YAML profiles |
| Permit Replacement | deepseek-chat | DeepSeek (OpenAI-compatible) | Detect system replacements from permit records *(implemented but not wired)* |

**Total: 3 active LLM processes**, plus 1 implemented analyzer not yet integrated.

---

## 1. Transcript Pipeline – Second Chance Lead Analysis

**File:** `services/transcript_pipeline.py`

**Classes:** `SecondChanceLeadAnalyzer`

**Model:** `deepseek-reasoner` (configurable via `MODEL_NAME`)

**Flow:**
1. Download call transcripts from Google Sheets
2. Create customer profiles grouped by phone number
3. For each customer, send call transcripts and summary to the LLM
4. LLM returns structured JSON with:
   - `is_second_chance_lead` (bool)
   - `was_customer_call`, `was_service_request`, `was_booked`
   - Invalidation reasons (e.g., rescheduling, invoice request, outside service area)
   - `reasoning`, `referenced_transcripts`, `recommendations`

**LLM calls:**
- Main: `analyze_customer_profile()` – full prompt with step-by-step logic
- Fallback: Simpler prompt if main analysis fails

**Concurrency:** ThreadPoolExecutor with configurable `CONCURRENT_API_CALLS` (default 50)

**API key:** `DEEPSEEK_API_KEY` (env or in code); uses `base_url="https://api.deepseek.com"`

---

## 2. Event Detection – Aging Systems (Job History)

**Files:**
- `datahound/events/llm_utils.py` – `SystemAgeAnalyzer`
- `datahound/events/scan_methods.py` – `scan_aging_systems()`
- `datahound/events/engine.py` – `EventScanner` wires in the analyzer

**Model:** `deepseek-chat` (via `LLMConfig`)

**Flow:**
1. Load locations and jobs from parquet
2. Build job histories per location
3. For each location with jobs, call `SystemAgeAnalyzer.analyze_location_jobs(location_id, jobs)`
4. LLM receives job summaries and returns JSON: `age`, `job_date`, `text_snippet`, `reasoning`, `confidence`, `source_type`, `replacement_detected`
5. Age rules: "Age of Equipment: X years", "installed in YEAR", replacement dates

**Concurrency:** `ThreadPoolExecutor` with `concurrent_llm_calls` (default 20)

**Output:** `EventResult` objects for locations with aging systems (configurable min age, default 15+ years)

---

## 3. Event Detection – Aging Systems (YAML Profiles)

**Files:**
- `datahound/events/llm_utils.py` – `LLMAnalyzer`, `build_aging_systems_system_prompt()`
- `apps/components/event_detection_ui.py` – Event Detection UI

**Model:** `deepseek-chat`

**Flow:**
1. User selects YAML files (one per location) containing `aggregates`, `permit_history`, `job_history`
2. `build_aging_systems_system_prompt()` builds a prompt with 8 questions:
   - `last_system_install_date`, `go_to_contractor`, `used_other_contractor_after_mccullough`
   - `estimated_current_system_age_years`, `most_recent_replacement`, `mccullough_has_replaced_system_here`
   - `last_permit_issue_date`, `last_job_date`
3. `LLMAnalyzer.analyze_texts_concurrent()` processes requests in parallel
4. Results written to `aging_systems.parquet` (or configurable filename)

**UI entry points:**
- Historical run: Triggered after permit history processing when YAML profiles exist
- LLM Settings tab: "Run on Selected" or "Run on All in Directory"

---

## 4. Permit Replacement Analyzer (Implemented, Not Wired)

**Files:**
- `datahound/events/llm_utils.py` – `PermitReplacementAnalyzer`
- `datahound/events/scan_methods.py` – `scan_permit_replacements()` returns `[]` with TODO

**Model:** `deepseek-chat`

**Status:** `PermitReplacementAnalyzer.analyze_location_permits()` exists and is implemented. `scan_permit_replacements()` is a stub that returns an empty list. It would analyze permit records to detect HVAC system replacements and determine if McCullough performed the replacement.

---

## 5. Components That Consume LLM Output (No Direct LLM Calls)

| Component | Role |
|-----------|------|
| `datahound/extract/engine.py` | Reads `second_chance_leads.parquet` produced by transcript pipeline; enriches with customer data; renames `reasoning` → `LLM Reasoning` |
| `datahound/events/dashboard.py` | Loads `llm_analysis.jsonl` for usage analytics (success rate, duration, cost estimation) |
| `datahound/events/logging.py` | Writes `llm_analysis.jsonl` for event-detection LLM calls |

---

## API Configuration

| Source | Keys | Provider |
|--------|------|----------|
| Transcript pipeline | `DEEPSEEK_API_KEY` (env or hardcoded) | DeepSeek |
| Event detection | `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `config/global.json` → `deepseek_api_key`, `.env` | DeepSeek |

All LLM usage goes through the OpenAI client with `base_url` set to `https://api.deepseek.com`.

---

## Placeholder / Future LLM Usage

In `event_detection_ui.py`, these event types have UI placeholders but no LLM logic yet:

- Overdue Maintenance
- Canceled Jobs
- Unsold Estimates
- Lost Customers

---

## Data Flow Overview

```
[Google Sheets] → transcript_pipeline.py
    → SecondChanceLeadAnalyzer (DeepSeek)
        → second_chance_leads.parquet
            → CustomExtractionEngine.extract_second_chance_leads()
                → recent_second_chance_leads.parquet

[Jobs.parquet, Locations.parquet] → EventScanner
    → scan_aging_systems() → SystemAgeAnalyzer (DeepSeek)
        → EventResult[] (aging_systems events)

[YAML profiles] → event_detection_ui
    → LLMAnalyzer.analyze_texts_concurrent() (DeepSeek)
        → aging_systems.parquet
```
