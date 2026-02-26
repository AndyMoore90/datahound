# SWARM_DONE.md — Definition of Done for Swarm Branches

This file defines the non-negotiable completion contract for swarm/agent branches.

## Scope boundary (important)

This swarm policy applies **only** to work inside the Datahound repository scope.

In-scope examples:
- `projects/datahound/**`
- Datahound PRs, branches, CI, docs, and devops scripts

Out-of-scope examples:
- workspace-level assistant/admin tasks
- non-Datahound projects
- personal ops unrelated to Datahound delivery

Out-of-scope agents do **not** need to run the swarm pipeline by default.
They should operate independently and consult the orchestrator only when:
- they need shared decisions/dependencies
- they are blocked
- their change could impact Datahound scope

## Task sizing + bundling (required before spawning workers)

Do not send tiny tasks through the full swarm pipeline one-by-one.

- **Small task**: ~1 file / <30 LOC / no behavioral change / docs-only or tiny config tweak
- **Medium task**: multiple files or behavioral/runtime/CI impact
- **Large task**: cross-module/system-level change

Rule:
- Pipeline tasks must be **medium+**.
- Small tasks must be grouped into a **single themed bundle branch** until they reach medium scope (e.g., `bundle/docs-ci-hygiene`, `bundle/ui-polish-round-3`).

Bundling acceptance criteria:
- Same risk profile and ownership area
- Same validation profile
- Clear single PR narrative

## Modes (enforcement profiles)

- **strict** (default): requires explicit checks + PR URL
- **safe**: fewer checks allowed for low-risk bundles (docs/CI/meta), but still requires clean tree + pushed branch

Use strict for code/runtime/dependency/schema changes.
Use safe only for low-risk scoped work.

## Required outputs (every task)

A task is **DONE** only if all of these are present:

1. **Local branch is clean** (`git status --porcelain` is empty)
2. **At least one commit exists for the task**
3. **Branch is pushed to origin**
4. **PR exists** OR task is explicitly marked **NO-OP** with reason
5. **Validation evidence** is provided (tests/checks run, or explicit reason for skip)

If any item is missing, the task is **INCOMPLETE**.

---

## Standard completion payload (required)

Every swarm worker/orchestrator handoff must include:

- `branch:`
- `head_commit:`
- `origin_tracking:`
- `working_tree_clean: true|false`
- `checks_run:` (list)
- `pr_url:` (or `no-op` + reason)

Example:

```text
branch: feat/swarm-ci-review-gates
head_commit: 9ec1f81
origin_tracking: origin/feat/swarm-ci-review-gates
working_tree_clean: true
checks_run: ["python -m compileall datahound services apps"]
pr_url: https://github.com/AndyMoore90/datahound/pull/123
```

---

## Enforcement

Use `scripts/swarm_finalize.sh` as the final step of every swarm run.

Default behavior (`--mode strict`):
- Fails on dirty tree
- Fails if branch has no upstream
- Fails if upstream differs from local HEAD
- Requires at least one `--checks` command
- Requires `--pr-url` (unless NO-OP)
- Prints a machine- and human-readable summary payload

Safe behavior (`--mode safe`):
- Same git hygiene checks
- Checks are optional (recommended)
- PR URL optional (recommended)

Optional behavior:
- `--allow-noop --noop-reason "..."` for intentionally no-change tasks

---

## Suggested orchestrator sequence

1. Worker makes changes
2. Worker runs checks
3. Worker commits
4. Worker pushes
5. Worker opens PR
6. Worker runs `scripts/swarm_finalize.sh`
7. Orchestrator accepts only if finalize passes

---

## Hard rule

No “done” message without passing finalize.
