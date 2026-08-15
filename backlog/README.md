# DaNa backlog

Pending and deferred work, one file per task. This is the durable to-do list
that outlives any single session — the roadmap lives in
[../docs/PLAN.md](../docs/PLAN.md); this tracks what's actually left.

## Conventions

- **Status:** `TODO` · `IN PROGRESS` · `BLOCKED` · `DONE`
- **Filename:** `TASK-NNN-short-slug.md`
- Each task states: status, priority, what unblocks it, acceptance criteria, and
  a pointer back to the plan/ADR it comes from.
- When a task is finished, set Status: DONE and note the commit/PR; don't delete
  the file (it's the record).

## Index

| Task | Status | Priority | Summary |
|---|---|---|---|
| — | DONE | — | Phase 1: scaffold + provider framework (see docs/PLAN.md §3) |
| [TASK-001](TASK-001-phase0-audio-spike.md) | BLOCKED | P0 | Phase 0 audio spike — confirm/overturn ADR-0001 (needs live mic) |
| [TASK-002](TASK-002-phase2-wake-gate.md) | DONE* | P1 | Phase 2: wake gate + IDLE/ACTIVE state machine (*real KWS/VAD in TASK-003) |
| [TASK-003](TASK-003-real-providers.md) | TODO | P1 | Implement real providers behind the ABCs (cloud + local sets) |
| [TASK-004](TASK-004-phase3-conversation-core.md) | TODO | P1 | Phase 3: Pipecat conversation core, barge-in, AEC, latency taps |
| [TASK-005](TASK-005-phase4-brain.md) | TODO | P2 | Phase 4: LangGraph brain — graph, checkpointing, interrupts |
| [TASK-006](TASK-006-phase5-actions.md) | TODO | P2 | Phase 5: action layer — Playwright + OS, gate, audit log |
| [TASK-007](TASK-007-phase6-control-ui.md) | TODO | P2 | Phase 6: FastAPI control API + React dashboard |
| [TASK-008](TASK-008-phase7-hardening.md) | TODO | P3 | Phase 7: autostart, secret vault, local-only mode, S2S, CI |
| [TASK-009](TASK-009-tooling-ci.md) | TODO | P3 | Dev tooling: pre-commit hooks + CI pipeline |
| [TASK-010](TASK-010-vietnamese-and-oww.md) | TODO | P3 | Vietnamese fixtures/quality + openWakeWord "Hey DaNa" training |
