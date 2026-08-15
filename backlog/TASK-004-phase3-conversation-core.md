# TASK-004 — Phase 3: conversation core (Pipecat)

- **Status:** TODO
- **Priority:** P1
- **Blocked by:** TASK-001 (spike confirms Pipecat) + TASK-003 (real providers).
- **Source:** docs/PLAN.md §3 (Phase 3), §7 (latency budget).

## Goal

Assemble the Pipecat cascade session behind a `ConversationProvider` that accepts
either a cascade (STT+LLM+TTS) or a single S2S model. Wire the wake gate as the
upstream frame processor. Production barge-in + AEC config; per-stage latency taps.

## Acceptance criteria

- [ ] "Hey DaNa → converse → Bye DaNa" loop works end to end with the cloud set.
- [ ] Median perceived latency ≤ 500 ms cloud set (stretch); ≤ 700 ms hard gate.
- [ ] No self-triggering over a 30-minute session (AEC working).
- [ ] Barge-in cancels TTS + pending LLM within ~300 ms.
- [ ] Both cloud and local sets pass en + vi smoke tests.
- [ ] `latency_span` events emitted per stage (vad, stt, llm_ttft, tts_ttfb, playback).
- [ ] Language switch en/vi from config.
