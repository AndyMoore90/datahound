# AI Usage Observability v2

This document defines the AI usage observability v2 pipeline for DataHound Pro. It aggregates usage across three execution paths:

1. Swarm task registry/logs
2. OpenClaw agent/subagent session usage metadata
3. CI review checks/workflow runs

The system provides provider/model/agent accounting, exact vs estimated labeling, untracked-delta indicators, and daily reconciliation reporting.

## Quick Start

- **Dashboard (Streamlit):** `AI Ops Usage v2` page
- **CLI report:**

```bash
python -m datahound.observability.ai_usage_v2 --days-back 30
```

- **Write daily reconciliation report:**

```bash
python -m datahound.observability.ai_usage_v2 --days-back 30 --write-reconciliation
```

## Log Sources

The aggregator looks for JSONL logs in these default locations (override with env vars).

| Source | Default paths | Env override |
| --- | --- | --- |
| Swarm | `logging/ai_ops/swarm_usage.jsonl`<br>`logging/cron_monitor/swarm_tasks.jsonl`<br>`logging/swarm/swarm_tasks.jsonl` | `AI_OPS_SWARM_LOGS` |
| OpenClaw sessions | `logging/ai_ops/openclaw_sessions.jsonl`<br>`logging/openclaw/session_usage.jsonl`<br>`logging/openclaw/usage.jsonl` | `AI_OPS_OPENCLAW_LOGS` |
| CI workflows | `logging/ai_ops/ci_usage.jsonl`<br>`logging/ci/ai_usage.jsonl`<br>`logging/ci/review_usage.jsonl` | `AI_OPS_CI_LOGS` |
| Provider-attributed usage | `logging/ai_ops/provider_usage.jsonl` | `AI_OPS_PROVIDER_USAGE` |

## Expected JSONL Schema (Flexible)

The collector is tolerant to schema differences. The following fields are recognized:

```json
{
  "ts": "2026-02-26T04:00:00Z",
  "provider": "deepseek",
  "model": "deepseek-chat",
  "agent": "swarm-fallback",
  "execution_path": "swarm",
  "usage_type": "exact",
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 800,
    "total_tokens": 2000,
    "cost_usd": 0.28
  },
  "task_id": "swarm-task-123",
  "run_id": "run-456"
}
```

Supported alternate fields include `prompt_tokens`, `completion_tokens`, `total_tokens`, `token_usage`, `estimated`, and `model_name`.

## Provider-Attributed Usage (for Reconciliation)

To compute untracked deltas, add provider totals to `logging/ai_ops/provider_usage.jsonl`:

```json
{
  "date": "2026-02-26",
  "provider": "deepseek",
  "tokens_total": 250000,
  "cost_usd": 35.0,
  "usage_type": "exact",
  "source": "billing_export"
}
```

The daily reconciliation report compares tracked totals vs provider totals and highlights untracked usage.

## Output Metrics

- **Breakdowns:** provider, model, agent, execution path, and source
- **Labeling:** exact vs estimated usage
- **Untracked delta:** provider totals minus tracked totals

## Notes

- No secrets are stored or displayed in logs or dashboards.
- The collector only reads JSONL files on disk.
