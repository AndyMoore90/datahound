# Swarm Auto-Merge Workflow

## Overview

The swarm auto-merge workflow automatically merges low-risk pull requests after CI and review gates pass, while requiring manual merge for critical changes.

## Features

- **Config-driven policy**: Allowlist branch patterns, labels, and critical keyword denylist
- **Safety gates**: Ensures CI passes and required checks (codex-review, claude-review) complete
- **Dry-run mode**: Test policy without actually merging
- **Logging**: Integrates with central_logging for audit trail
- **Telegram notifications**: Alerts when PRs are auto-merged or skipped as critical
- **Age limits**: Skip stale PRs that haven't been updated recently

## Policy Configuration

Configuration is stored in `config/swarm_automerge_policy.json`:

```json
{
  "allow_branch_patterns": [
    "streamlit/*",
    "ui/*",
    "docs/*",
    "chore/*",
    "fix/ui-*"
  ],
  "allow_labels": [
    "automerge",
    "safe-to-merge"
  ],
  "critical_keywords": [
    "critical",
    "security",
    "db",
    "migration",
    "scheduler",
    "breaking"
  ],
  "critical_labels": [
    "do-not-automerge",
    "manual-merge",
    "security"
  ],
  "base_branches": ["main"],
  "required_checks": [
    "ci",
    "codex-review",
    "claude-review"
  ],
  "merge_method": "squash",
  "max_pr_age_hours": 96,
  "notify": {
    "telegram": {
      "enabled": true,
      "events": ["auto-merged", "skipped-critical"]
    }
  }
}
```

### Policy Fields

- **allow_branch_patterns**: Shell-style wildcard patterns for low-risk branches
- **allow_labels**: PR labels that explicitly allow auto-merge
- **critical_keywords**: Keywords in PR title/body/branch that block auto-merge
- **critical_labels**: PR labels that block auto-merge
- **base_branches**: Target branches that can receive auto-merges (usually `["main"]`)
- **required_checks**: CI check names that must pass (uses substring match)
- **merge_method**: `squash`, `merge`, or `rebase`
- **max_pr_age_hours**: Skip PRs not updated within this window (or `null` to disable)
- **notify.telegram**: Telegram notification configuration

## Usage

### Manual Invocation

```bash
# Dry-run mode (no actual merges)
python -m datahound.devops.swarm_automerge --dry-run

# Merge up to 5 eligible PRs
python -m datahound.devops.swarm_automerge --max-prs 5

# Override config path
python -m datahound.devops.swarm_automerge --config /path/to/policy.json

# Specify repo explicitly
python -m datahound.devops.swarm_automerge --repo owner/reponame
```

### Environment Variables

- `GITHUB_TOKEN`: GitHub API token with repo permissions (required)
- `TELEGRAM_BOT_TOKEN`: Telegram bot token (optional, for notifications)
- `TELEGRAM_CHAT_ID`: Telegram chat ID for notifications (optional)

### Cron Integration

To run auto-merge periodically, use OpenClaw cron jobs:

```bash
# Schedule to run every 30 minutes (isolated session, auto-announces results)
openclaw cron add \
  --name "swarm-automerge" \
  --schedule-every 30m \
  --session-target isolated \
  --payload-kind agentTurn \
  --payload-message "Run swarm auto-merge check: cd /path/to/datahound && python -m datahound.devops.swarm_automerge" \
  --delivery-mode announce
```

## Merge Criteria

A PR is eligible for auto-merge if **ALL** of the following are true:

1. ✅ PR is open and not a draft
2. ✅ Base branch is in `base_branches` (e.g., `main`)
3. ✅ Branch name matches an `allow_branch_patterns` pattern **OR** has an `allow_labels` label
4. ❌ Does **NOT** have any `critical_labels`
5. ❌ Does **NOT** contain any `critical_keywords` in title, body, or branch name
6. ✅ Updated within `max_pr_age_hours` (if set)
7. ✅ Combined status is `success`
8. ✅ All `required_checks` have passed (conclusion: success, neutral, or skipped)
9. ✅ PR is mergeable (no conflicts, not blocked)

## Default Behavior

**Auto-merge allowed:**
- `streamlit/*`, `ui/*`, `docs/*`, `chore/*` branches
- PRs with `automerge` or `safe-to-merge` labels
- Non-critical fixes like `fix/ui-*`, `fix/docs-*`

**Manual merge required:**
- Critical keywords: `security`, `db`, `migration`, `scheduler`, `breaking`
- Critical labels: `do-not-automerge`, `manual-merge`, `security`
- Any other branch pattern not explicitly allowlisted

## Testing

### Unit Tests

```bash
# Run auto-merge tests
.venv/bin/pytest tests/test_swarm_automerge.py -v

# Test with coverage
.venv/bin/pytest tests/test_swarm_automerge.py --cov=datahound.devops.swarm_automerge
```

### Integration Test (Dry-Run)

```bash
# Test against real repo in dry-run mode
export GITHUB_TOKEN="your_token"
python -m datahound.devops.swarm_automerge --dry-run --log-level DEBUG
```

## Rollback Instructions

If auto-merge causes issues:

### 1. Immediate Disable

**Option A: Delete cron job**
```bash
openclaw cron list
openclaw cron remove --job-id <swarm-automerge-job-id>
```

**Option B: Disable in config**
```bash
# Set all patterns to empty arrays to block everything
echo '{"allow_branch_patterns": [], "allow_labels": [], ...}' > config/swarm_automerge_policy.json
```

### 2. Revert Auto-Merged PR

If a specific auto-merged PR causes issues:

```bash
# Find the merge commit
git log --oneline --grep="Merge pull request #<number>"

# Revert it
git revert <merge-commit-sha> -m 1
git push origin main
```

### 3. Audit What Was Merged

Check central logging:

```bash
# View auto-merge log
cat logging/cron_monitor/swarm_auto_merge.jsonl | jq '.[] | select(.status == "auto-merged")'
```

### 4. Emergency: Block Future Auto-Merges

Add the `do-not-automerge` label to all open PRs:

```bash
gh pr list --json number --jq '.[].number' | xargs -I{} gh pr edit {} --add-label do-not-automerge
```

## Monitoring

### Logs

- **JSONL logs**: `logging/cron_monitor/swarm_auto_merge.jsonl`
- **Stdout logs**: Captured by OpenClaw if run via cron

### Telegram Notifications

When `notify.telegram.enabled` is true:
- **auto-merged**: Notification when PR is successfully merged
- **skipped-critical**: Notification when PR is skipped due to critical keywords/labels

### Sample Log Entry

```json
{
  "job": "swarm_auto_merge",
  "status": "auto-merged",
  "dry_run": false,
  "message": "Auto-merged pull request",
  "pr_number": 123,
  "title": "chore: update streamlit dashboard colors",
  "head": "chore/streamlit-colors",
  "author": "bot-user",
  "timestamp": "2026-02-25T18:30:00Z"
}
```

## Troubleshooting

### PRs Not Auto-Merging

1. **Check required checks**: Ensure all `required_checks` pass
   ```bash
   gh pr checks <pr-number>
   ```

2. **Verify branch pattern**: Does branch match an `allow_branch_patterns` pattern?
   ```bash
   python -c "from fnmatch import fnmatch; print(fnmatch('your-branch', 'ui/*'))"
   ```

3. **Check for critical keywords**: Search PR title/body for `critical_keywords`

4. **Run in dry-run with debug**:
   ```bash
   python -m datahound.devops.swarm_automerge --dry-run --log-level DEBUG
   ```

### False Positives (Wrong PRs Merged)

- Review and tighten `allow_branch_patterns`
- Add missing keywords to `critical_keywords`
- Use `critical_labels` for exceptions

### Notification Not Sending

- Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` env vars
- Check `notify.telegram.enabled` is `true` in config
- Ensure event is in `notify.telegram.events` list

## Security Considerations

- **Token permissions**: GitHub token needs `repo` scope for merging
- **Review bypass**: Auto-merge does NOT bypass branch protection rules; required reviews must still be satisfied
- **Audit trail**: All merges are logged to `swarm_auto_merge.jsonl`
- **Rate limits**: API calls are throttled; script respects GitHub rate limits
- **Dry-run first**: Always test with `--dry-run` before enabling cron

## Future Enhancements

- [ ] Support for additional notification channels (Slack, Discord)
- [ ] Configurable retry logic for transient API failures
- [ ] Per-branch policy overrides
- [ ] Metrics dashboard integration
- [ ] Auto-comment on PR before merging
