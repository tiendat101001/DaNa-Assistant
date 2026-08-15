# TASK-001 — Phase 0 audio spike (confirm/overturn ADR-0001)

- **Status:** BLOCKED
- **Priority:** P0
- **Blocked by:** needs an interactive session on the target Windows 11 machine
  with a real microphone and speakers. Cannot be done autonomously/offline.
- **Source:** [docs/adr/ADR-0001-realtime-engine.md](../docs/adr/ADR-0001-realtime-engine.md),
  docs/PLAN.md §2–3.

## Goal

Throwaway Pipecat pipeline (mic → Deepgram → LLM → ElevenLabs → speakers) on
this machine to validate the Pipecat decision, especially acoustic echo
cancellation on the bare local mic.

## Acceptance criteria (ADR-0001 exit gate — all must pass)

- [ ] Local mic → STT → LLM → TTS → speakers runs stably for a 10-minute session.
- [ ] Median perceived turn latency ≤ 800 ms un-tuned, with per-stage timings captured.
- [ ] Barge-in: user speech during TTS cancels playback + pending work within ~300 ms.
- [ ] No self-triggering: assistant speech does not re-enter STT over ≥ 20 replies
      (validate AEC mitigation ladder: Windows voice-capture DSP → WebRTC AEC3 filter
      → SmallWebRTCTransport → half-duplex).
- [ ] A custom frame processor (stub wake gate) can drop/pass audio upstream of STT.

## Outcome

Record result as a decision note; if AEC/barge-in fail after mitigations 1–3,
switch to LiveKit Agents (documented fallback). Spike code is throwaway and kept
out of the production package.
