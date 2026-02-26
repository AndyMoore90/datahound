#!/usr/bin/env python3
"""Tests for swarm auto-merge workflow."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from datahound.devops.swarm_automerge import (
    AutoMergePolicy,
    AutoMergeWorker,
    CronMonitorLogger,
    GitHubClient,
    TelegramNotifier,
    detect_repo,
    parse_dt,
    parse_github_repo,
)


@pytest.fixture
def sample_policy_dict() -> Dict[str, Any]:
    return {
        "allow_branch_patterns": ["ui/*", "docs/*", "chore/*"],
        "allow_labels": ["automerge"],
        "critical_keywords": ["security", "db", "migration"],
        "critical_labels": ["do-not-automerge"],
        "required_checks": ["ci", "review"],
        "base_branches": ["main"],
        "merge_method": "squash",
        "max_pr_age_hours": 96,
        "notify": {
            "telegram": {
                "enabled": True,
                "events": ["auto-merged", "skipped-critical"],
            }
        },
    }


@pytest.fixture
def sample_policy(sample_policy_dict) -> AutoMergePolicy:
    return AutoMergePolicy(
        allow_branch_patterns=sample_policy_dict["allow_branch_patterns"],
        allow_labels=sample_policy_dict["allow_labels"],
        critical_keywords=sample_policy_dict["critical_keywords"],
        critical_labels=sample_policy_dict["critical_labels"],
        required_checks=sample_policy_dict["required_checks"],
        base_branches=sample_policy_dict["base_branches"],
        merge_method=sample_policy_dict["merge_method"],
        max_pr_age_hours=sample_policy_dict["max_pr_age_hours"],
        notify=sample_policy_dict["notify"],
    )


@pytest.fixture
def temp_config_file(sample_policy_dict, tmp_path: Path) -> Path:
    config_file = tmp_path / "policy.json"
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(sample_policy_dict, f)
    return config_file


class TestAutoMergePolicy:
    def test_branch_allowlisted_match(self, sample_policy):
        assert sample_policy.branch_allowlisted("ui/fix-button")
        assert sample_policy.branch_allowlisted("docs/readme")
        assert sample_policy.branch_allowlisted("chore/cleanup")

    def test_branch_allowlisted_no_match(self, sample_policy):
        assert not sample_policy.branch_allowlisted("feat/new-feature")
        assert not sample_policy.branch_allowlisted("fix/bug")

    def test_branch_allowlisted_case_insensitive(self, sample_policy):
        assert sample_policy.branch_allowlisted("UI/Fix-Button")
        assert sample_policy.branch_allowlisted("DOCS/README")

    def test_has_allow_label(self, sample_policy):
        assert sample_policy.has_allow_label(["automerge"])
        assert sample_policy.has_allow_label(["other", "automerge"])
        assert not sample_policy.has_allow_label(["other"])

    def test_has_critical_label(self, sample_policy):
        assert sample_policy.has_critical_label(["do-not-automerge"])
        assert not sample_policy.has_critical_label(["other"])

    def test_contains_critical_keyword(self, sample_policy):
        assert sample_policy.contains_critical_keyword("This is a security fix")
        assert sample_policy.contains_critical_keyword("db migration script")
        assert sample_policy.contains_critical_keyword("SECURITY update")
        assert not sample_policy.contains_critical_keyword("ui fix")

    def test_base_branch_allowed(self, sample_policy):
        assert sample_policy.base_branch_allowed("main")
        assert sample_policy.base_branch_allowed("MAIN")
        assert not sample_policy.base_branch_allowed("develop")

    def test_is_candidate_allowlisted_branch(self, sample_policy):
        pr = {
            "head": {"ref": "ui/fix-button"},
            "labels": [],
            "title": "Fix button color",
            "body": "Updates button styling",
        }
        allowed, reason, is_critical = sample_policy.is_candidate(pr)
        assert allowed
        assert reason == "allowlisted"
        assert not is_critical

    def test_is_candidate_critical_keyword(self, sample_policy):
        pr = {
            "head": {"ref": "ui/fix-button"},
            "labels": [],
            "title": "Security fix for XSS",
            "body": "",
        }
        allowed, reason, is_critical = sample_policy.is_candidate(pr)
        assert not allowed
        assert reason == "critical-keyword"
        assert is_critical

    def test_is_candidate_critical_label(self, sample_policy):
        pr = {
            "head": {"ref": "ui/fix-button"},
            "labels": [{"name": "do-not-automerge"}],
            "title": "Fix button",
            "body": "",
        }
        allowed, reason, is_critical = sample_policy.is_candidate(pr)
        assert not allowed
        assert reason == "critical-label"
        assert is_critical

    def test_is_candidate_not_allowlisted(self, sample_policy):
        pr = {
            "head": {"ref": "feat/new-feature"},
            "labels": [],
            "title": "Add new feature",
            "body": "",
        }
        allowed, reason, is_critical = sample_policy.is_candidate(pr)
        assert not allowed
        assert reason == "branch-not-allowlisted"
        assert not is_critical

    def test_is_candidate_allow_label_overrides(self, sample_policy):
        pr = {
            "head": {"ref": "feat/risky-feature"},
            "labels": [{"name": "automerge"}],
            "title": "Risky feature",
            "body": "",
        }
        allowed, reason, is_critical = sample_policy.is_candidate(pr)
        assert allowed
        assert reason == "allowlisted"

    def test_is_too_old(self, sample_policy):
        old_dt = datetime.now(timezone.utc) - timedelta(hours=100)
        recent_dt = datetime.now(timezone.utc) - timedelta(hours=24)
        assert sample_policy.is_too_old(old_dt)
        assert not sample_policy.is_too_old(recent_dt)

    def test_is_too_old_disabled(self):
        policy = AutoMergePolicy(max_pr_age_hours=None)
        old_dt = datetime.now(timezone.utc) - timedelta(days=365)
        assert not policy.is_too_old(old_dt)

    def test_from_file(self, temp_config_file):
        policy = AutoMergePolicy.from_file(temp_config_file)
        assert policy.allow_branch_patterns == ["ui/*", "docs/*", "chore/*"]
        assert policy.merge_method == "squash"
        assert policy.max_pr_age_hours == 96


class TestGitHubClient:
    def test_init_requires_token(self):
        with pytest.raises(ValueError, match="GitHub token is required"):
            GitHubClient("", "owner", "repo")

    @patch("requests.Session.request")
    def test_list_pull_requests(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '[{"number": 1}]'
        mock_response.json.return_value = [{"number": 1}]
        mock_request.return_value = mock_response

        client = GitHubClient("token", "owner", "repo")
        prs = client.list_pull_requests()
        assert len(prs) == 1
        assert prs[0]["number"] == 1

    @patch("requests.Session.request")
    def test_get_pull_request(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"number": 42}'
        mock_response.json.return_value = {"number": 42}
        mock_request.return_value = mock_response

        client = GitHubClient("token", "owner", "repo")
        pr = client.get_pull_request(42)
        assert pr["number"] == 42

    @patch("requests.Session.request")
    def test_merge_pull_request(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"merged": true}'
        mock_response.json.return_value = {"merged": True}
        mock_request.return_value = mock_response

        client = GitHubClient("token", "owner", "repo")
        result = client.merge_pull_request(42, "abc123", "squash")
        assert result["merged"] is True

    @patch("requests.Session.request")
    def test_api_error_handling(self, mock_request):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_request.return_value = mock_response

        client = GitHubClient("token", "owner", "repo")
        with pytest.raises(RuntimeError, match="GitHub API error 404"):
            client.list_pull_requests()


class TestTelegramNotifier:
    def test_from_config_enabled(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")
        config = {
            "notify": {
                "telegram": {
                    "enabled": True,
                    "events": ["auto-merged"],
                }
            }
        }
        notifier = TelegramNotifier.from_config(config)
        assert notifier.token == "bot-token"
        assert notifier.chat_id == "chat-id"
        assert "auto-merged" in notifier.events

    def test_from_config_disabled(self):
        config = {
            "notify": {
                "telegram": {
                    "enabled": False,
                    "events": ["auto-merged"],
                }
            }
        }
        notifier = TelegramNotifier.from_config(config)
        assert not notifier.events

    @patch("requests.post")
    def test_notify_sends_message(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        notifier = TelegramNotifier("token", "chat-id", ["auto-merged"])
        notifier.notify("auto-merged", "Test message")
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_notify_skips_filtered_events(self, mock_post):
        notifier = TelegramNotifier("token", "chat-id", ["auto-merged"])
        notifier.notify("other-event", "Test message")
        mock_post.assert_not_called()

    def test_notify_no_token(self):
        notifier = TelegramNotifier(None, "chat-id", ["auto-merged"])
        notifier.notify("auto-merged", "Test message")
        # Should not raise


class TestCronMonitorLogger:
    @patch("datahound.devops.swarm_automerge.log_writer.write")
    def test_write(self, mock_write):
        logger = CronMonitorLogger(dry_run=False)
        pr = {
            "number": 42,
            "title": "Test PR",
            "head": {"ref": "test-branch"},
            "user": {"login": "testuser"},
        }
        logger.write("auto-merged", pr, "Test message")
        mock_write.assert_called_once()
        args = mock_write.call_args[0]
        assert args[0] == "cron_monitor"
        assert args[1] == "swarm_auto_merge.jsonl"
        record = args[2]
        assert record["status"] == "auto-merged"
        assert record["pr_number"] == 42


class TestAutoMergeWorker:
    @pytest.fixture
    def mock_client(self):
        client = Mock(spec=GitHubClient)
        return client

    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=CronMonitorLogger)

    @pytest.fixture
    def mock_notifier(self):
        return Mock(spec=TelegramNotifier)

    def test_run_skips_draft(self, sample_policy, mock_client, mock_logger, mock_notifier):
        mock_client.list_pull_requests.return_value = [{"number": 1, "draft": True}]
        worker = AutoMergeWorker(mock_client, sample_policy, mock_logger, mock_notifier)
        result = worker.run()
        assert len(result["merged"]) == 0
        assert (1, "draft") in result["skipped"]

    def test_run_skips_wrong_base_branch(self, sample_policy, mock_client, mock_logger, mock_notifier):
        mock_client.list_pull_requests.return_value = [
            {"number": 1, "draft": False, "base": {"ref": "develop"}}
        ]
        worker = AutoMergeWorker(mock_client, sample_policy, mock_logger, mock_notifier)
        result = worker.run()
        assert (1, "base-branch") in result["skipped"]

    def test_run_dry_run_mode(self, sample_policy, mock_client, mock_logger, mock_notifier):
        pr = {
            "number": 1,
            "draft": False,
            "base": {"ref": "main"},
            "head": {"ref": "ui/fix", "sha": "abc123"},
            "title": "UI fix",
            "body": "",
            "labels": [],
            "mergeable": True,
            "mergeable_state": "clean",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        mock_client.list_pull_requests.return_value = [pr]
        mock_client.get_pull_request.return_value = pr
        mock_client.get_combined_status.return_value = {"state": "success"}
        mock_client.get_check_runs.return_value = {
            "check_runs": [
                {"name": "ci-test", "conclusion": "success"},
                {"name": "review-check", "conclusion": "success"},
            ]
        }
        worker = AutoMergeWorker(mock_client, sample_policy, mock_logger, mock_notifier, dry_run=True)
        result = worker.run()
        assert 1 in result["merged"]
        mock_client.merge_pull_request.assert_not_called()

    def test_run_merges_eligible_pr(self, sample_policy, mock_client, mock_logger, mock_notifier):
        pr = {
            "number": 1,
            "draft": False,
            "base": {"ref": "main"},
            "head": {"ref": "ui/fix", "sha": "abc123"},
            "title": "UI fix",
            "body": "",
            "labels": [],
            "mergeable": True,
            "mergeable_state": "clean",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        mock_client.list_pull_requests.return_value = [pr]
        mock_client.get_pull_request.return_value = pr
        mock_client.get_combined_status.return_value = {"state": "success"}
        mock_client.get_check_runs.return_value = {
            "check_runs": [
                {"name": "ci-test", "conclusion": "success"},
                {"name": "review-check", "conclusion": "success"},
            ]
        }
        mock_client.merge_pull_request.return_value = {"merged": True}
        worker = AutoMergeWorker(mock_client, sample_policy, mock_logger, mock_notifier, dry_run=False)
        result = worker.run()
        assert 1 in result["merged"]
        mock_client.merge_pull_request.assert_called_once_with(1, "abc123", "squash")

    def test_run_skips_critical_and_notifies(self, sample_policy, mock_client, mock_logger, mock_notifier):
        pr = {
            "number": 1,
            "draft": False,
            "base": {"ref": "main"},
            "head": {"ref": "ui/fix", "sha": "abc123"},
            "title": "Security fix",
            "body": "",
            "labels": [],
            "mergeable": True,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        mock_client.list_pull_requests.return_value = [pr]
        mock_client.get_pull_request.return_value = pr
        worker = AutoMergeWorker(mock_client, sample_policy, mock_logger, mock_notifier)
        result = worker.run()
        assert (1, "critical-keyword") in result["critical_skips"]
        mock_notifier.notify.assert_called()


class TestHelperFunctions:
    def test_parse_github_repo(self):
        owner, repo = parse_github_repo("testowner/testrepo")
        assert owner == "testowner"
        assert repo == "testrepo"

    def test_parse_github_repo_invalid(self):
        with pytest.raises(ValueError, match="must be in owner/repo format"):
            parse_github_repo("invalid")

    def test_parse_dt(self):
        dt = parse_dt("2026-02-25T18:00:00Z")
        assert dt.year == 2026
        assert dt.month == 2
        assert dt.day == 25
        assert dt.tzinfo == timezone.utc

    @patch("subprocess.check_output")
    def test_detect_repo_https(self, mock_check_output):
        mock_check_output.return_value = "https://github.com/testowner/testrepo.git\n"
        owner, repo = detect_repo()
        assert owner == "testowner"
        assert repo == "testrepo"

    @patch("subprocess.check_output")
    def test_detect_repo_ssh(self, mock_check_output):
        mock_check_output.return_value = "git@github.com:testowner/testrepo.git\n"
        owner, repo = detect_repo()
        assert owner == "testowner"
        assert repo == "testrepo"
