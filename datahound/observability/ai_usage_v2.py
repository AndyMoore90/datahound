"""AI usage observability v2: aggregate usage across swarm, OpenClaw, and CI."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = PROJECT_ROOT / "logging" / "ai_ops"

SWARM_DEFAULT_LOGS = [
    PROJECT_ROOT / "logging" / "ai_ops" / "swarm_usage.jsonl",
    PROJECT_ROOT / "logging" / "cron_monitor" / "swarm_tasks.jsonl",
    PROJECT_ROOT / "logging" / "swarm" / "swarm_tasks.jsonl",
]
OPENCLAW_DEFAULT_LOGS = [
    PROJECT_ROOT / "logging" / "ai_ops" / "openclaw_sessions.jsonl",
    PROJECT_ROOT / "logging" / "openclaw" / "session_usage.jsonl",
    PROJECT_ROOT / "logging" / "openclaw" / "usage.jsonl",
]
CI_DEFAULT_LOGS = [
    PROJECT_ROOT / "logging" / "ai_ops" / "ci_usage.jsonl",
    PROJECT_ROOT / "logging" / "ci" / "ai_usage.jsonl",
    PROJECT_ROOT / "logging" / "ci" / "review_usage.jsonl",
]
PROVIDER_ATTRIBUTED_DEFAULT = PROJECT_ROOT / "logging" / "ai_ops" / "provider_usage.jsonl"


@dataclass
class UsageRecord:
    timestamp: datetime
    provider: str
    model: str
    agent: str
    execution_path: str
    source: str
    usage_type: str
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    cost_usd: float
    task_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "date": self.timestamp.date().isoformat(),
            "provider": self.provider,
            "model": self.model,
            "agent": self.agent,
            "execution_path": self.execution_path,
            "source": self.source,
            "usage_type": self.usage_type,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "tokens_total": self.tokens_total,
            "cost_usd": self.cost_usd,
            "task_id": self.task_id,
            "run_id": self.run_id,
        }


@dataclass
class ProviderUsage:
    date: str
    provider: str
    tokens_total: int
    cost_usd: float
    usage_type: str
    source: str


def _read_env_paths(var_name: str) -> List[Path]:
    env_value = os.environ.get(var_name, "")
    if not env_value:
        return []
    paths = []
    for item in env_value.split(","):
        item = item.strip()
        if not item:
            continue
        p = Path(item).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        paths.append(p)
    return paths


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return entries


def _parse_timestamp(entry: Dict[str, Any]) -> Optional[datetime]:
    candidates = [
        entry.get("ts"),
        entry.get("timestamp"),
        entry.get("created_at"),
        entry.get("started_at"),
        entry.get("ended_at"),
    ]
    for value in candidates:
        if not value:
            continue
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            try:
                cleaned = value.replace("Z", "+00:00")
                return datetime.fromisoformat(cleaned)
            except ValueError:
                continue
    return None


def _infer_provider(model: str) -> str:
    model_lower = model.lower()
    if "deepseek" in model_lower:
        return "deepseek"
    if "claude" in model_lower:
        return "anthropic"
    if "gpt" in model_lower or "openai" in model_lower:
        return "openai"
    if "kimi" in model_lower:
        return "moonshot"
    return "unknown"


def _extract_tokens(entry: Dict[str, Any]) -> Tuple[int, int, int, str]:
    usage = entry.get("usage") or entry.get("token_usage") or entry.get("tokens") or {}
    prompt = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or entry.get("prompt_tokens")
        or entry.get("input_tokens")
        or entry.get("tokens_prompt")
        or entry.get("prompt")
        or 0
    )
    completion = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or entry.get("completion_tokens")
        or entry.get("output_tokens")
        or entry.get("tokens_completion")
        or entry.get("completion")
        or 0
    )
    total = (
        usage.get("total_tokens")
        or entry.get("total_tokens")
        or entry.get("tokens_total")
        or entry.get("total")
        or 0
    )

    prompt = int(prompt or 0)
    completion = int(completion or 0)
    total = int(total or 0)

    if total == 0 and (prompt or completion):
        total = prompt + completion

    usage_type = entry.get("usage_type") or ""
    if usage_type:
        usage_type = str(usage_type).lower()
    elif entry.get("estimated") or entry.get("is_estimated"):
        usage_type = "estimated"
    elif entry.get("estimated_tokens") or entry.get("token_estimate"):
        usage_type = "estimated"
    else:
        usage_type = "exact" if total > 0 else "unknown"

    return prompt, completion, total, usage_type


def _extract_cost(entry: Dict[str, Any]) -> float:
    cost = entry.get("cost_usd")
    if cost is None:
        usage = entry.get("usage") or entry.get("token_usage") or {}
        cost = usage.get("cost_usd") or usage.get("cost") or entry.get("cost")
    try:
        return float(cost)
    except (TypeError, ValueError):
        return 0.0


def _normalize_record(entry: Dict[str, Any], source: str, default_execution_path: str) -> Optional[UsageRecord]:
    ts = _parse_timestamp(entry)
    if not ts:
        return None

    model = entry.get("model") or entry.get("model_name") or entry.get("llm_model") or "unknown"
    provider = entry.get("provider") or entry.get("vendor") or _infer_provider(model)
    agent = entry.get("agent") or entry.get("agent_id") or entry.get("session_agent") or "unknown"
    execution_path = entry.get("execution_path") or entry.get("path") or default_execution_path

    prompt, completion, total, usage_type = _extract_tokens(entry)
    cost_usd = _extract_cost(entry)

    task_id = entry.get("task_id") or entry.get("job_id") or entry.get("swarm_task_id")
    run_id = entry.get("run_id") or entry.get("workflow_run_id") or entry.get("session_id")

    return UsageRecord(
        timestamp=ts,
        provider=str(provider),
        model=str(model),
        agent=str(agent),
        execution_path=str(execution_path),
        source=source,
        usage_type=str(usage_type),
        tokens_prompt=prompt,
        tokens_completion=completion,
        tokens_total=total,
        cost_usd=cost_usd,
        task_id=str(task_id) if task_id else None,
        run_id=str(run_id) if run_id else None,
    )


def _collect_from_paths(paths: Iterable[Path], source: str, execution_path: str, cutoff: datetime) -> List[UsageRecord]:
    records: List[UsageRecord] = []
    for path in paths:
        for entry in _load_jsonl(path):
            record = _normalize_record(entry, source, execution_path)
            if record and record.timestamp >= cutoff:
                records.append(record)
    return records


def load_usage_records(days_back: int = 30) -> List[UsageRecord]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days_back)

    swarm_paths = _read_env_paths("AI_OPS_SWARM_LOGS") or SWARM_DEFAULT_LOGS
    openclaw_paths = _read_env_paths("AI_OPS_OPENCLAW_LOGS") or OPENCLAW_DEFAULT_LOGS
    ci_paths = _read_env_paths("AI_OPS_CI_LOGS") or CI_DEFAULT_LOGS

    records: List[UsageRecord] = []
    records.extend(_collect_from_paths(swarm_paths, "swarm", "swarm", cutoff))
    records.extend(_collect_from_paths(openclaw_paths, "openclaw", "direct-agent", cutoff))
    records.extend(_collect_from_paths(ci_paths, "ci", "ci", cutoff))
    return records


def load_provider_attributed_usage(days_back: int = 30) -> List[ProviderUsage]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
    provider_path = Path(os.environ.get("AI_OPS_PROVIDER_USAGE", str(PROVIDER_ATTRIBUTED_DEFAULT)))
    if not provider_path.is_absolute():
        provider_path = PROJECT_ROOT / provider_path
    entries = _load_jsonl(provider_path)
    results: List[ProviderUsage] = []
    for entry in entries:
        ts = _parse_timestamp(entry) or datetime.fromisoformat(entry.get("date", "1970-01-01"))
        if ts < cutoff:
            continue
        results.append(
            ProviderUsage(
                date=entry.get("date") or ts.date().isoformat(),
                provider=entry.get("provider", "unknown"),
                tokens_total=int(entry.get("tokens_total", 0) or 0),
                cost_usd=float(entry.get("cost_usd", 0.0) or 0.0),
                usage_type=entry.get("usage_type", "exact"),
                source=entry.get("source", "provider_export"),
            )
        )
    return results


def summarize_usage(records: List[UsageRecord]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "counts": {"exact": 0, "estimated": 0, "unknown": 0},
        "tokens_by_usage_type": {"exact": 0, "estimated": 0, "unknown": 0},
        "by_provider": {},
        "by_model": {},
        "by_agent": {},
        "by_execution_path": {},
        "by_source": {},
    }
    for record in records:
        summary["total_tokens"] += record.tokens_total
        summary["total_cost_usd"] += record.cost_usd
        summary["counts"].setdefault(record.usage_type, 0)
        summary["tokens_by_usage_type"].setdefault(record.usage_type, 0)
        summary["counts"][record.usage_type] += 1
        summary["tokens_by_usage_type"][record.usage_type] += record.tokens_total

        for key, bucket in [
            (record.provider, "by_provider"),
            (record.model, "by_model"),
            (record.agent, "by_agent"),
            (record.execution_path, "by_execution_path"),
            (record.source, "by_source"),
        ]:
            entry = summary[bucket].setdefault(key, {"tokens": 0, "cost_usd": 0.0, "count": 0})
            entry["tokens"] += record.tokens_total
            entry["cost_usd"] += record.cost_usd
            entry["count"] += 1

    summary["total_cost_usd"] = round(summary["total_cost_usd"], 4)
    return summary


def build_daily_reconciliation(records: List[UsageRecord], provider_usage: List[ProviderUsage]) -> List[Dict[str, Any]]:
    tracked: Dict[Tuple[str, str], Dict[str, float]] = {}
    for record in records:
        key = (record.timestamp.date().isoformat(), record.provider)
        bucket = tracked.setdefault(key, {"tokens": 0, "cost": 0.0})
        bucket["tokens"] += record.tokens_total
        bucket["cost"] += record.cost_usd

    report: List[Dict[str, Any]] = []
    for entry in provider_usage:
        key = (entry.date, entry.provider)
        tracked_bucket = tracked.get(key, {"tokens": 0, "cost": 0.0})
        delta_tokens = entry.tokens_total - tracked_bucket["tokens"]
        delta_cost = entry.cost_usd - tracked_bucket["cost"]
        report.append(
            {
                "date": entry.date,
                "provider": entry.provider,
                "provider_tokens": entry.tokens_total,
                "tracked_tokens": tracked_bucket["tokens"],
                "untracked_delta_tokens": delta_tokens,
                "provider_cost_usd": round(entry.cost_usd, 4),
                "tracked_cost_usd": round(tracked_bucket["cost"], 4),
                "untracked_delta_cost_usd": round(delta_cost, 4),
                "usage_type": entry.usage_type,
                "source": entry.source,
            }
        )

    return sorted(report, key=lambda row: (row["date"], row["provider"]))


def write_reconciliation_report(report: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for row in report:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI usage observability v2 report")
    parser.add_argument("--days-back", type=int, default=30)
    parser.add_argument("--write-reconciliation", action="store_true")
    parser.add_argument("--reconciliation-path", type=str, default=str(LOG_ROOT / "reconciliation_daily.jsonl"))
    args = parser.parse_args()

    records = load_usage_records(days_back=args.days_back)
    summary = summarize_usage(records)

    print("AI usage summary")
    print(json.dumps(summary, indent=2))

    provider_usage = load_provider_attributed_usage(days_back=args.days_back)
    if provider_usage:
        report = build_daily_reconciliation(records, provider_usage)
        print("\nDaily reconciliation")
        print(json.dumps(report, indent=2))
        if args.write_reconciliation:
            write_reconciliation_report(report, Path(args.reconciliation_path))
            print(f"\nWrote reconciliation to {args.reconciliation_path}")


if __name__ == "__main__":
    main()
