# ADR-0001 — Real-time voice engine: adopt Pipecat

- **Status:** Proposed (to be confirmed by the Phase 0 spike; see exit criteria below)
- **Date:** 2026-08-15
- **Decision drivers:** single-machine always-on desktop assistant on Windows; sub-500 ms latency target; provider swappability incl. local CPU-only providers; en + vi; custom wake gate must sit *in front of* the conversation loop; solo-maintainable.

## Context

The brief (§2) mandates adopting a real-time engine rather than rebuilding the STT↔LLM↔TTS loop, and requires a Phase 0 evaluation of **Pipecat vs LiveKit Agents vs from-scratch**. Both frameworks provide streaming cascade pipelines, turn detection, barge-in, VAD, and 20+ swappable providers per layer, plus speech-to-speech providers.

The decisive difference is **deployment shape**:

- **Pipecat** is a composable pipeline of frame processors running **in one Python process**. Its local audio transport does mic → pipeline → speakers directly on the machine — no media server, no rooms, no network hop for audio.
- **LiveKit Agents** is architected around agents joining **rooms on a LiveKit server** (self-hosted or cloud); the client publishes mic audio over WebRTC into the room. Its server-less "console mode" is explicitly a dev/testing convenience, not the production shape. For DaNa — always-on, single machine, mic and speakers on the same box as the agent — the server+worker+client topology is pure overhead: an extra process to install, supervise, and update on Windows, and an extra hop for audio.

## Decision

**Adopt Pipecat** as the real-time engine. Do not build the loop from scratch.

Rationale, in order of weight:

1. **Topology fits the product.** DaNa's production shape *is* Pipecat's local shape: one always-on Python process owning mic, pipeline, and speakers. LiveKit's production shape requires running a media server on the desktop for an audio path that never leaves the machine.
2. **The frame-processor model gives us our custom insertion points.** The wake gate (VAD + KWS + IDLE/ACTIVE state machine) becomes a frame processor placed before STT that swallows audio frames in `IDLE` and opens on wake. Per-stage latency taps are likewise small processors between stages. In LiveKit, `AgentSession` is more integrated and the pre-session audio path is less ours to own.
3. **Broadest provider catalog on both required sets.** Cloud (Deepgram, AssemblyAI, Gemini, OpenAI, Anthropic, ElevenLabs, Cartesia…), local (faster-whisper, Piper/Kokoro, Ollama/OpenAI-compatible), and S2S (OpenAI Realtime, Gemini Live, Nova Sonic, Grok) — so the brief's provider matrix maps to existing Pipecat services plus a thin config layer, per the provider-plugin skill.
4. **Windows-friendly, minimal footprint.** Pure Python + PyAudio; no Go server binary or Docker requirement on the user's desktop. BSD-2 licensed.

**From-scratch is rejected**: it means owning endpointing, interruption cancellation, echo cancellation, and jitter handling forever — precisely the failure mode the voice-agent-architecture skill warns against, with zero product differentiation.

## Known risk: echo cancellation on the local transport

This is the honest weak spot and the reason the decision stays "Proposed" until the spike passes. Browser/WebRTC clients (LiveKit's strength) get AEC3 for free; a bare PyAudio capture does not, and AEC on Pipecat's local transport is a known open concern (pipecat-ai/pipecat#670). Mitigation ladder, to be validated in Phase 0 on this Windows 11 machine, in order of preference:

1. **Windows voice-capture DSP** — open the mic in Communications/speech processing mode so the OS APM does AEC before Python ever sees audio.
2. **Software AEC filter** — a WebRTC AEC3 binding (e.g. `webrtc-audio-processing`) as a Pipecat audio filter, fed with the TTS playback as the reference signal.
3. **Pipecat SmallWebRTCTransport** — serverless P2P WebRTC to a local page; browser-grade AEC at the cost of coupling capture to a running browser context (acceptable as fallback, wrong as default for "always-on").
4. **Half-duplex gating** — duck/ignore mic while TTS plays. Kills barge-in; last resort only, and would trigger reconsideration of LiveKit.

## Phase 0 spike exit criteria (confirm or overturn this ADR)

- [ ] Local mic → Deepgram → LLM → ElevenLabs → speakers runs stably on Windows 11 for a 10-minute session.
- [ ] Median perceived turn latency ≤ 800 ms in the spike (un-tuned), with per-stage timings captured.
- [ ] Barge-in works: user speech during TTS cancels playback and LLM/TTS work within ~300 ms.
- [ ] No self-triggering: assistant speech does not re-enter STT (AEC option 1 or 2 above works), measured over ≥ 20 spoken replies.
- [ ] A custom frame processor (stub wake gate) can drop/pass audio frames upstream of STT.

## Consequences

- The provider abstraction (§5 of the brief) is implemented as thin wrappers/factories over Pipecat service classes, not re-implemented streaming clients. Wake and Action providers remain fully custom ABCs.
- We take a dependency on Pipecat's release cadence; pin versions, wrap all Pipecat imports behind our provider layer so churn is contained.
- **Revisit triggers → LiveKit Agents:** (a) spike fails AEC/barge-in criteria after mitigations 1–3; (b) DaNa later needs multi-device/phone/remote access, where LiveKit's WebRTC transport and telephony story wins.
