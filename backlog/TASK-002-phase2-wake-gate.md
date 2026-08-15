# TASK-002 — Phase 2: wake gate + IDLE/ACTIVE state machine

- **Status:** DONE (offline architecture + test harness). Real KWS/VAD models
  and the 1 h desk-audio false-wake check are tracked in TASK-003 / TASK-010.
- **Priority:** P1
- **Source:** docs/PLAN.md §3 (Phase 2), voice-agent-architecture skill.

## Goal

The custom wake gate that sits in front of the conversation core: VAD gate →
KWS (wake engine) → explicit IDLE/ACTIVE state machine, ending on "Bye DaNa"
**and** an inactivity timeout. Built on the mock providers so it is fully
testable offline.

## Acceptance criteria

- [x] `VADProvider` ABC + mock (energy gate), registered and swappable.
- [x] Explicit state machine: `State` enum + transition table, emits a
      `StateTransition` event on every edge (IDLE→ACTIVE on wake, ACTIVE→IDLE on
      end phrase / timeout / cancel).
- [x] `WakeGate` composes VAD + wake engine + state machine; forwards audio to
      the session only while ACTIVE; enforces inactivity timeout via an injected
      clock (deterministic in tests).
- [x] FSM integration test drives sentinel/silence frame sequences (fixtures)
      and asserts on transitions + events, not audio bytes.
- [ ] Real wake engines (Porcupine default, openWakeWord) — deferred to
      TASK-003; the mock proves the gate offline.
- [ ] Silero VAD real provider — deferred to TASK-003.

## Notes

Real KWS/VAD models and recorded audio clips land with TASK-003 and TASK-010;
this task delivers the gate architecture and its offline test harness.
