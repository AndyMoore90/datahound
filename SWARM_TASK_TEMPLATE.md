# SWARM_TASK_TEMPLATE.md

Use this template for every swarm worker task so completion is consistent and merge-ready.

## Scope boundary

Use this template only for Datahound-scoped tasks (`projects/datahound/**`).
For non-Datahound work, agents may execute outside swarm and only consult the orchestrator when needed.

---

## Copy/Paste Task Prompt (for swarm workers)

```md
You are working in the Datahound repo.

## Objective
<describe the specific task clearly>

## Constraints
- Keep scope tight to this task only.
- Do not touch unrelated files.
- Do not leave a dirty worktree.
- If this is a small task, bundle it with related small tasks into one medium-sized branch before finalizing.

## Mode
- `strict` for code/runtime/dependency/schema changes
- `safe` for low-risk docs/CI/meta changes

## Required workflow (must follow in order)
1) Implement the change.
2) Run task-relevant validation checks.
3) Commit with a clear Conventional Commit message.
4) Push branch to origin.
5) Open/update PR.
6) Run finalize:
   scripts/swarm_finalize.sh \
     --mode <strict|safe> \
     --checks "<check 1>" \
     --checks "<check 2>" \
     --pr-url "<PR URL>"

## Definition of done (strict)
Return this payload exactly:
- branch:
- head_commit:
- origin_tracking:
- working_tree_clean:
- checks_run:
- pr_url:

If no code changes are required, run:
  scripts/swarm_finalize.sh --allow-noop --noop-reason "<reason>"

## Output format
- Brief summary of changes (3-6 bullets)
- Final completion payload
```

---

## Small-task bundling protocol

When a task is small, do not open a standalone PR.

Bundle branch naming examples:
- `bundle/docs-ci-hygiene-YYYYMMDD`
- `bundle/ui-polish-YYYYMMDD`

Bundle requirements:
- max 3-6 small items
- same area/risk class
- one coherent PR summary

## Recommended check sets by task type

### Python code changes
```bash
python -m compileall datahound services apps
```

### Dependency/runtime changes
```bash
python -m compileall datahound services apps
python - <<'PY'
import datahound
print('imports-ok')
PY
```

### Workflow/CI-only changes
```bash
python -m py_compile datahound/devops/ai_review_gate.py
```

(Use checks that are valid for the files you changed.)

---

## Commit message examples

- `fix: resolve streamlit runtime error in data pipeline page`
- `ci: make deepseek review advisory`
- `docs: add storage refactor target architecture v1`
- `feat: add swarm automerge policy config and runner`

---

## Orchestrator acceptance rule

A task is accepted only when:
- `scripts/swarm_finalize.sh` passes
- completion payload is present
- PR URL is present (or explicit approved no-op)
