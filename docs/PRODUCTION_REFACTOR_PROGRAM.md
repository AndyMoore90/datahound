# Datahound Production Refactor Program

## Objective
Transform Datahound from prototype-oriented architecture into a production-ready platform while keeping current operations usable.

## Principles
- Reliability over feature count
- Clear service boundaries
- Deterministic history/auditability
- Idempotent ingestion and replay support
- Cost-aware AI usage (right model for right task)

## Workstreams
1. Platform Architecture (frontend/backend/data boundaries)
2. Data Reliability (ingestion, retries, dead-letter, history)
3. Event Detection Quality (rule validation + regression tests)
4. Orchestration Refactor (orchestrator-first control with app trigger capability)
5. Security & Readiness (secrets/tokens preflight and startup gates)
6. UX Overhaul (non-technical operator clarity)

## Phase Plan
### Phase 0: Baseline + Spec
- Capture current flows and failure modes
- Define target architecture and acceptance criteria
- Produce migration strategy with rollback points

### Phase 1: Foundations
- Introduce canonical service interfaces
- Add structured logging and run correlation IDs
- Add startup preflight validator for required creds/tokens

### Phase 2: Reliability Hardening
- Idempotent ingestion and checkpointing
- Event history precision and immutable run/event records
- Replay/backfill pipeline

### Phase 3: UX + Frontend Productionization
- Production web dashboard stack and navigation model
- Explainability and operator-centric views

### Phase 4: Orchestration Unification
- App-triggered automation hooks
- Orchestrator-managed execution lifecycle as primary path

## AI Cost Governance
- Critical core code: gpt-5.3-codex
- Frontend/dashboard: claude-opus-4-5
- Important non-critical core code: gpt-5.2-codex
- Use smaller scope prompts, strict task boundaries, and minimal diff strategy
- Require evidence-based checks only for changed surfaces
