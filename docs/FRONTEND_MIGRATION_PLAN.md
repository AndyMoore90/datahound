# Frontend Migration Plan (Streamlit -> Web App)

## Decision
Datahound should migrate away from Streamlit for production-facing frontend use.

## Target stack
- Frontend: Next.js + React + Tailwind
- API: FastAPI-backed endpoints for control-plane and analytics views
- Auth: add session/auth guard before broad rollout

## Agent delegation policy (required)
All frontend/UI implementation tasks must be delegated to a dedicated frontend agent using:
- **Model:** `anthropic/claude-opus-4-5`

Backend/data/refactor tasks remain on the core engineering lane.
Mixed tickets must be split into separate backend + frontend subtasks.

## Phased rollout
1. **API-first parity**
   - expose read-only endpoints currently consumed by Streamlit pages
2. **New web dashboard v1**
   - operator home, pipeline runs, scheduler health, review gate status
3. **Parallel run**
   - keep Streamlit as fallback while validating parity and reliability
4. **Cutover**
   - make web UI primary
   - Streamlit becomes internal fallback only
5. **Retirement**
   - remove Streamlit pages once parity + reliability checks pass

## Immediate next frontend ticket
- Build read-only page for pipeline run history and scheduler status from DB-backed endpoints.
