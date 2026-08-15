# TASK-007 — Phase 6: control API + management UI

- **Status:** TODO
- **Priority:** P2
- **Blocked by:** TASK-002/004/005/006 (there must be state/config/events to serve).
- **Source:** docs/PLAN.md §5 (API contract), §6, ADR-0002.

## Goal

FastAPI control plane inside the agent (REST config CRUD + WebSocket live
events), bound to loopback with a bearer token; a React + Vite dashboard that
renders config forms from the JSON Schema and shows live transcript/latency/FSM.

## Acceptance criteria

- [ ] REST endpoints per docs/PLAN.md §5 (config, schema, providers, session,
      status, mic, confirmations, audit, health); secrets redacted.
- [ ] WebSocket `/events` emits typed envelopes (state_transition,
      transcript_*, latency_span, confirmation_request, audit_entry, ...).
- [ ] Config edit → validate → hot-apply (safe fields) round-trips; unsafe
      fields flagged `restart_required`.
- [ ] Dashboard: schema-driven config editor, start/stop, transcript + FSM view,
      latency waterfall, audit view, mic mute, pending-confirmation prompts.
- [ ] Agent runs headless (UI closed) with zero degradation.
- [ ] Loopback-only + token auth; no remote exposure in v1.
