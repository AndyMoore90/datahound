# Runbook: AI Usage Observability v2

## Purpose

This runbook helps operators validate AI usage reporting across swarm tasks, OpenClaw sessions, and CI workflows.

## When to Use

- Daily reconciliation review
- Investigating missing or unexpected AI usage
- Verifying new pipeline sources are captured

## Preconditions

- JSONL usage logs are available under `logging/ai_ops/` (or custom paths via env vars)
- Optional provider usage exports for reconciliation

## Commands

### 1) Generate current summary

```bash
python -m datahound.observability.ai_usage_v2 --days-back 30
```

### 2) Write reconciliation report

```bash
python -m datahound.observability.ai_usage_v2 --days-back 30 --write-reconciliation
```

Output defaults to: `logging/ai_ops/reconciliation_daily.jsonl`

### 3) Review the Streamlit dashboard

Open **AI Ops Usage v2** page in the app to view breakdowns and raw records.

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| No records shown | Logs missing or incorrect paths | Verify log locations or set `AI_OPS_SWARM_LOGS`, `AI_OPS_OPENCLAW_LOGS`, `AI_OPS_CI_LOGS` |
| Provider reconciliation empty | No provider usage export | Add `logging/ai_ops/provider_usage.jsonl` entries |
| Large untracked delta | Provider totals exceed tracked | Confirm all execution paths emit usage logs |

## Safety

- Do **not** store API keys in logs.
- Strip payloads that might include sensitive prompts before exporting usage logs.
