#!/usr/bin/env python3
"""Swarm auto-merge workflow for low-risk pull requests."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from central_logging import writer as log_writer

LOG = logging.getLogger("swarm-automerge")
API_BASE = "https://api.github.com"
DEFAULT_CONFIG_PATH = Path("config/swarm_automerge_policy.json")
LOG_FILE_NAME = "swarm_auto_merge.jsonl"
JOB_NAME = "swarm_auto_merge"
PASS_CONCLUSIONS = {"success", "neutral", "skipped"}


@dataclass
class AutoMergePolicy:
    """Policy definition loaded from config."""

    allow_branch_patterns: Sequence[str] = field(default_factory=list)
    allow_labels: Sequence[str] = field(default_factory=list)
    critical_keywords: Sequence[str] = field(default_factory=list)
    critical_labels: Sequence[str] = field(default_factory=list)
    required_checks: Sequence[str] = field(default_factory=list)
    base_branches: Sequence[str] = field(default_factory=lambda: ["main"])
    merge_method: str = "squash"
    max_pr_age_hours: Optional[int] = 96
    notify: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.allow_branch_patterns = [pattern.lower() for pattern in self.allow_branch_patterns]
        self.allow_labels = [label.lower() for label in self.allow_labels]
        self.critical_keywords = [kw.lower() for kw in self.critical_keywords]
        self.critical_labels = [label.lower() for label in self.critical_labels]
        self.required_checks = [chk.lower() for chk in self.required_checks]
        self.base_branches = [branch.lower() for branch in self.base_branches]
        self.merge_method = (self.merge_method or "squash").lower()
        self._allow_label_set = set(self.allow_labels)
        self._critical_label_set = set(self.critical_labels)
        self._base_branch_set = set(self.base_branches)

    def branch_allowlisted(self, branch: str) -> bool:
        branch_lower = branch.lower()
        return any(fnmatch(branch_lower, pattern) for pattern in self.allow_branch_patterns)

    def has_allow_label(self, labels: Iterable[str]) -> bool:
        return any(label in self._allow_label_set for label in labels)

    def has_critical_label(self, labels: Iterable[str]) -> bool:
        return any(label in self._critical_label_set for label in labels)

    def contains_critical_keyword(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.critical_keywords)

    def base_branch_allowed(self, branch: str) -> bool:
        if not self._base_branch_set:
            return True
        return branch.lower() in self._base_branch_set

    @classmethod
    def from_file(cls, path: Path) -> "AutoMergePolicy":
        """Load policy from JSON config file."""
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return cls(
            allow_branch_patterns=config.get("allow_branch_patterns", []),
            allow_labels=config.get("allow_labels", []),
            critical_keywords=config.get("critical_keywords", []),
            critical_labels=config.get("critical_labels", []),
            required_checks=config.get("required_checks", []),
            base_branches=config.get("base_branches", ["main"]),
            merge_method=config.get("merge_method", "squash"),
            max_pr_age_hours=config.get("max_pr_age_hours", 96),
            notify=config.get("notify", {}),
        )

    def is_candidate(self, pr: Dict[str, Any]) -> Tuple[bool, str, bool]:
        """Return (allowed, reason, is_critical)."""

        branch = pr.get("head", {}).get("ref", "")
        labels = {label.get("name", "").lower() for label in pr.get("labels", [])}
        title = pr.get("title", "")
        body = pr.get("body") or ""
        combined_text = " ".join(filter(None, [branch, title, body]))

        if self.has_critical_label(labels):
            return False, "critical-label", True
        if self.contains_critical_keyword(combined_text):
            return False, "critical-keyword", True
        allow_branch = self.branch_allowlisted(branch)
        allow_label = self.has_allow_label(labels)
        if not (allow_branch or allow_label):
            return False, "branch-not-allowlisted", False
        return True, "allowlisted", False

    def is_too_old(self, updated_at: datetime) -> bool:
        if not self.max_pr_age_hours:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_pr_age_hours)
        return updated_at < cutoff

class GitHubClient:
    """Minimal GitHub REST client."""

    def __init__(self, token: str, owner: str, repo: str) -> None:
        if not token:
            raise ValueError("GitHub token is required")
        self.owner = owner
        self.repo = repo
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "datahound-swarm-automerge",
            }
        )
        self.base_url = f"{API_BASE}/repos/{owner}/{repo}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.request(method, url, timeout=15, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"GitHub API error {resp.status_code}: {resp.text}")
        if resp.text:
            return resp.json()
        return None

    def list_pull_requests(self, per_page: int = 50) -> List[Dict[str, Any]]:
        params = {"state": "open", "per_page": per_page, "sort": "updated", "direction": "desc"}
        return self._request("GET", "/pulls", params=params)

    def get_pull_request(self, number: int) -> Dict[str, Any]:
        return self._request("GET", f"/pulls/{number}")

    def get_combined_status(self, sha: str) -> Dict[str, Any]:
        return self._request("GET", f"/commits/{sha}/status")

    def get_check_runs(self, sha: str) -> Dict[str, Any]:
        return self._request("GET", f"/commits/{sha}/check-runs")

    def merge_pull_request(self, number: int, sha: str, method: str) -> Dict[str, Any]:
        payload = {"merge_method": method, "sha": sha}
        return self._request("PUT", f"/pulls/{number}/merge", json=payload)


class TelegramNotifier:
    def __init__(self, token: Optional[str], chat_id: Optional[str], events: Iterable[str]):
        self.token = token
        self.chat_id = chat_id
        self.events = {event for event in events if event}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TelegramNotifier":
        telegram_cfg = config.get("notify", {}).get("telegram", {}) if config else {}
        events = telegram_cfg.get("events", []) if telegram_cfg.get("enabled", False) else []
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        return cls(token, chat_id, events)

    def notify(self, event: str, message: str) -> None:
        if not self.token or not self.chat_id:
            return
        if self.events and event not in self.events:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message}
        try:
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code >= 400:
                LOG.warning("Telegram send failed (%s): %s", resp.status_code, resp.text)
        except requests.RequestException as exc:
            LOG.warning("Telegram send exception: %s", exc)


class CronMonitorLogger:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run

    def write(self, status: str, pr: Dict[str, Any], message: str, extra: Dict[str, Any] | None = None) -> None:
        record = {
            "job": JOB_NAME,
            "status": status,
            "dry_run": self.dry_run,
            "message": message,
            "pr_number": pr.get("number"),
            "title": pr.get("title"),
            "head": pr.get("head", {}).get("ref"),
            "author": pr.get("user", {}).get("login"),
        }
        if extra:
            record.update(extra)
        try:
            log_writer.write("cron_monitor", LOG_FILE_NAME, record)
        except Exception as exc:  # pragma: no cover - logging should not break job
            LOG.debug("Unable to persist cron monitor log: %s", exc)


def detect_repo() -> Tuple[str, str]:
    try:
        remote = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("Unable to detect git remote origin") from exc
    if not remote:
        raise RuntimeError("Empty origin remote")
    repo_part: Optional[str] = None
    if remote.startswith("git@"):
        _, repo_part = remote.split(":", 1)
    elif remote.startswith("https://") or remote.startswith("http://"):
        idx = remote.find("github.com/")
        if idx != -1:
            repo_part = remote[idx + len("github.com/") :]
    if not repo_part:
        raise RuntimeError(f"Unsupported remote format: {remote}")
    repo_part = repo_part.rstrip(".git")
    owner, repo = repo_part.split("/", 1)
    return owner, repo


def parse_github_repo(repo_str: Optional[str]) -> Tuple[str, str]:
    if repo_str:
        if "/" not in repo_str:
            raise ValueError("--repo must be in owner/repo format")
        owner, repo = repo_str.split("/", 1)
        return owner, repo
    return detect_repo()


def parse_dt(value: str) -> datetime:
    dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return dt.replace(tzinfo=timezone.utc)


class AutoMergeWorker:
    def __init__(
        self,
        client: GitHubClient,
        policy: AutoMergePolicy,
        logger: CronMonitorLogger,
        notifier: TelegramNotifier,
        dry_run: bool = False,
    ) -> None:
        self.client = client
        self.policy = policy
        self.logger = logger
        self.notifier = notifier
        self.dry_run = dry_run

    def run(self, max_candidates: Optional[int] = None) -> Dict[str, Any]:
        merged: List[int] = []
        skipped: List[Tuple[int, str]] = []
        critical_skips: List[Tuple[int, str]] = []
        failures: List[str] = []

        prs = self.client.list_pull_requests()
        for pr_stub in prs:
            number = pr_stub.get("number")
            if not number:
                continue
            if pr_stub.get("draft"):
                skipped.append((number, "draft"))
                continue
            if not self.policy.base_branch_allowed(pr_stub.get("base", {}).get("ref", "")):
                skipped.append((number, "base-branch"))
                continue
            try:
                pr = self._get_mergeable_pr(number)
            except RuntimeError as exc:
                failures.append(f"pr #{number}: {exc}")
                continue
            allowed, reason, is_critical = self.policy.is_candidate(pr)
            if not allowed:
                skipped.append((number, reason))
                if is_critical:
                    critical_skips.append((number, reason))
                    self._notify("skipped-critical", pr, reason)
                continue
            updated_at_str = pr.get("updated_at") or pr.get("created_at")
            updated_at = parse_dt(updated_at_str) if updated_at_str else datetime.now(timezone.utc)
            if self.policy.is_too_old(updated_at):
                skipped.append((number, "stale"))
                continue
            ready, check_reason = self._checks_pass(pr)
            if not ready:
                skipped.append((number, check_reason or "checks"))
                continue
            if max_candidates and len(merged) >= max_candidates:
                break
            if self.dry_run:
                LOG.info("DRY-RUN would merge #%s %s", number, pr.get("title"))
                self.logger.write("dry-run", pr, "Auto-merge dry-run")
                merged.append(number)
                continue
            try:
                self.client.merge_pull_request(number, pr["head"]["sha"], self.policy.merge_method)
                LOG.info("Auto-merged #%s %s", number, pr.get("title"))
                merged.append(number)
                self.logger.write("auto-merged", pr, "Auto-merged pull request")
                self._notify("auto-merged", pr, "✅ Auto-merged PR #{number}: {title}".format(
                    number=number, title=pr.get("title", "")
                ))
            except RuntimeError as exc:
                failures.append(f"merge #{number}: {exc}")
                self.logger.write("error", pr, "Merge failed", {"error": str(exc)})

        return {
            "merged": merged,
            "skipped": skipped,
            "critical_skips": critical_skips,
            "failures": failures,
        }

    def _get_mergeable_pr(self, number: int) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(4):
            try:
                pr = self.client.get_pull_request(number)
            except RuntimeError as exc:
                last_exc = exc
                time.sleep(1)
                continue
            mergeable = pr.get("mergeable")
            state = (pr.get("mergeable_state") or "").lower()
            if mergeable is None and attempt < 3:
                time.sleep(1)
                continue
            if mergeable is False or state in {"dirty", "blocked", "draft"}:
                raise RuntimeError(f"mergeable_state={state or mergeable}")
            return pr
        raise RuntimeError(f"unable to fetch mergeable PR: {last_exc}")

    def _checks_pass(self, pr: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        head = pr.get("head", {})
        sha = head.get("sha")
        if not sha:
            return False, "missing-sha"
        status = self.client.get_combined_status(sha)
        if status.get("state") != "success":
            return False, "combined-status"
        check_runs_payload = self.client.get_check_runs(sha)
        check_runs = check_runs_payload.get("check_runs", []) if isinstance(check_runs_payload, dict) else []
        passed: Dict[str, bool] = {}
        for run in check_runs:
            name = (run.get("name") or "").lower()
            conclusion = (run.get("conclusion") or "").lower()
            if not name:
                continue
            for required in self.policy.required_checks:
                if required in name and conclusion in PASS_CONCLUSIONS:
                    passed[required] = True
        for required in self.policy.required_checks:
            if not passed.get(required):
                return False, f"missing-check:{required}"
        return True, None

    def _notify(self, event: str, pr: Dict[str, Any], reason: str) -> None:
        if not event:
            return
        title = pr.get("title", "")
        number = pr.get("number")
        branch = pr.get("head", {}).get("ref")
        message = f"[{JOB_NAME}] {event} PR #{number} ({branch}) - {title}\nReason: {reason}"
        self.notifier.notify(event, message)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swarm auto-merge for low-risk PRs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to auto-merge policy JSON")
    parser.add_argument("--repo", type=str, help="GitHub repo (owner/repo). Defaults to origin remote." )
    parser.add_argument("--dry-run", action="store_true", help="Enable dry-run mode")
    parser.add_argument("--github-token", type=str, help="GitHub token (overrides GITHUB_TOKEN env)")
    parser.add_argument("--max-prs", type=int, default=None, help="Max pull requests to merge per run")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")
    try:
        policy = AutoMergePolicy.from_file(args.config)
    except Exception as exc:
        LOG.error("Unable to load auto-merge policy: %s", exc)
        return 1
    try:
        owner, repo = parse_github_repo(args.repo)
    except Exception as exc:
        LOG.error("Unable to determine repository: %s", exc)
        return 1
    token = args.github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        LOG.error("GITHUB_TOKEN is required")
        return 1
    client = GitHubClient(token, owner, repo)
    notifier = TelegramNotifier.from_config(policy.__dict__ | {"notify": policy.__dict__.get("notify", {})})
    # policy.__dict__ does not contain notify; reload raw JSON for notifier
    try:
        with args.config.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
    except Exception:
        raw_config = {}
    notifier = TelegramNotifier.from_config(raw_config)
    logger = CronMonitorLogger(args.dry_run)
    worker = AutoMergeWorker(client, policy, logger, notifier, dry_run=args.dry_run)
    result = worker.run(max_candidates=args.max_prs)
    LOG.info("Merged: %s | Skipped: %s | Critical skips: %s | Failures: %s", len(result["merged"]), len(result["skipped"]), len(result["critical_skips"]), len(result["failures"]))
    if result["failures"]:
        for failure in result["failures"]:
            LOG.error("Failure: %s", failure)
    return 0 if not result["failures"] else 2


if __name__ == "__main__":
    sys.exit(main())
