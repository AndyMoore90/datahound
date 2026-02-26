# DataHound Automation

This document lists all automation features and workflows available in the DataHound project.

## GitHub Automation

### Swarm Auto-Merge

**Location:** `datahound/devops/swarm_automerge.py`  
**Config:** `config/swarm_automerge_policy.json`  
**Docs:** [docs/swarm_automerge.md](swarm_automerge.md)

Automatically merges low-risk pull requests after CI and review gates pass.

**Quick start:**
```bash
# Dry-run to test policy
GITHUB_TOKEN=xxx python -m datahound.devops.swarm_automerge --dry-run

# Run for real
GITHUB_TOKEN=xxx python -m datahound.devops.swarm_automerge
```

**Cron schedule:**
```bash
openclaw cron add \
  --name "swarm-automerge" \
  --schedule-every 30m \
  --session-target isolated \
  --payload-kind agentTurn \
  --payload-message "Run swarm auto-merge: cd /path/to/datahound && python -m datahound.devops.swarm_automerge"
```

## Future Automation Ideas

- Automated dependency updates
- Stale PR cleanup
- Release automation
- Backup verification
- Performance monitoring alerts
