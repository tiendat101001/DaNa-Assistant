---
name: voice-agent-architecture
description: Architecture and best practices for real-time, always-on voice assistants and speech-to-speech agents. Use this skill whenever the user is building a voice assistant, voice agent, wake-word ("Hey X") system, always-listening app, speech-to-speech pipeline, or anything combining a microphone with STT, an LLM, and TTS — even if they don't name a framework. It covers the wake-word gate, VAD, turn-taking, barge-in, echo cancellation, latency budgets, the cascade-vs-speech-to-speech decision, the idle/active state machine, and the critical build-vs-adopt call (Pipecat / LiveKit Agents vs rolling your own loop). Consult it before designing any audio-in/audio-out conversational system so the design avoids the common naive mistakes.
---

# Voice Agent Architecture

Real-time voice agents fail for predictable reasons: the wake mechanism is built wrong, latency is treated as an afterthought, and the hard real-time plumbing (interruption, echo cancellation, turn detection) is hand-rolled. This skill encodes the architecture that avoids those failures.

## First principle: adopt the real-time engine, do not rebuild it

The pluggable STT to LLM to TTS loop — with turn detection, barge-in (letting the user interrupt the bot mid-sentence), and acoustic echo cancellation — is a **solved problem**. Two mature open-source Python frameworks provide it with dozens of swappable providers per layer:

- **Pipecat** (by Daily): a composable pipeline of frame processors. You assemble STT + LLM + TTS (or a speech-to-speech model) and it handles the real-time concerns. Broadest provider catalog; strong for novel/custom pipelines.
- **LiveKit Agents**: an `AgentSession` that runs STT, LLM, TTS as pluggable components with the SDK wiring up streaming, turn detection, and interruption. Strong WebRTC transport; good foundation for a production platform.

**Default recommendation: build on one of these, not from scratch.** Rolling your own orchestrator means owning the hard real-time loop forever — NAT traversal, jitter buffers, endpointing, interruption cancellation, echo cancellation. Only hand-roll when requirements genuinely don't fit either framework, and when they don't, record *why* in an ADR.

A user asking for "multiple swappable providers" almost always thinks they must build the abstraction themselves. They should not: the framework's plugin system **is** that abstraction. Their custom layer is a thin config wrapper on top (see the `provider-plugin-architecture` skill). Reserve bespoke engineering for the parts the frameworks don't cover: the **wake-word gate**, the **agent brain**, **system/OS actions**, and the **UI**.

When you propose an architecture, always present build-vs-adopt explicitly with a recommendation and rationale — never silently hand-roll the loop.

## The wake-word mechanism (the most common mistake)

Do **NOT** run STT continuously and string-match the wake phrase. It is high-latency, expensive (you pay to transcribe silence), and inaccurate. Wake detection is a distinct, cheap, always-on component that sits *in front of* the rest of the system.

Correct signal chain:

```
mic → VAD (cheap gate) → Wake-Word Engine (KWS) → [on hit] → Conversation Session → [end phrase / timeout] → back to VAD
```

- **VAD (Voice Activity Detection)** — Silero VAD or webrtcvad. A dB/energy threshold alone is crude; use a real VAD model. VAD's job is only to decide "is someone speaking" so you don't run heavier models on silence.
- **Wake-Word Engine (KWS = keyword spotting)** — a dedicated on-device model that listens *only* for the wake phrase. Options:
  - **openWakeWord** — Apache-2.0, free, runs many models in real time on a Raspberry Pi core; custom phrases require training a small model.
  - **Porcupine / Picovoice** — commercial, very accurate, custom wake word trained in seconds with no ML work; free tier exists.
  - Make the wake engine itself a swappable provider so the choice can change later.
- **End of session** — detect an end phrase (e.g. "Bye X") via the KWS engine or via intent detection in the conversation loop, **and** an inactivity timeout. Never rely on the end phrase alone; users forget to say it.

## Model the lifecycle as an explicit state machine

```
IDLE  --wake detected-->  ACTIVE(listening/conversing)
ACTIVE  --end phrase | inactivity timeout-->  IDLE
ACTIVE  --error / cancel-->  IDLE
```

Make states and transitions first-class in code (an enum + a transition table or a small library), not implicit boolean flags scattered across callbacks. Emit an event on every transition so the UI and logs can observe it. This is the single most important structural decision for debuggability.

## Always-on hazards you must design for

An always-listening assistant that also speaks aloud has two failure modes that a push-to-talk demo never hits:

1. **Self-triggering / echo** — when the assistant speaks, the microphone hears its own TTS and may re-trigger the wake word or feed its own voice back into STT. Mitigate with **acoustic echo cancellation (AEC)**, or half-duplex gating (mute the mic while speaking) as a fallback. Pipecat/LiveKit include AEC; this is a major reason to adopt one.
2. **Barge-in** — the user interrupts while the assistant is talking. The pipeline must detect user speech, **cancel** pending LLM/TTS work immediately, and switch to listening. Frameworks handle this; a hand-rolled loop must implement cancellation carefully or the bot talks over the user.

## Latency is the product

Perceived turn latency (user stops talking to assistant starts talking) is the metric that decides whether the assistant feels usable.

- Under ~300 ms feels human; over ~600 ms feels sluggish; over ~1.5 s and users give up.
- **Stream at every stage**: streaming STT (partial transcripts), streaming LLM tokens, streaming TTS (start speaking on the first sentence, not the whole reply). Never wait for a full transcript, then a full completion, then a full audio file.
- Use endpointing / turn detection to start the LLM as soon as the user's turn ends, not after a fixed silence timeout.
- Budget latency per stage and measure it per stage (see Observability).

## Cascade vs speech-to-speech

Two ways to build the conversation core:

- **Cascade (STT to LLM to TTS)** — the default. Maximum control and provider flexibility; each layer is swappable; easiest to add tools, guardrails, and text logging. Choose this unless you have a specific reason not to.
- **Speech-to-speech (single multimodal model)** — e.g. OpenAI Realtime, Gemini Live, Grok Voice, Ultravox, Nova Sonic. Lowest latency and most natural prosody, but the whole turn is one vendor's black box: nothing to swap, harder to inject tools/guardrails, weaker text observability.

Design the abstraction so a "conversation provider" can be **either** a cascade **or** a single speech-to-speech model — then the choice is config, not a rewrite. Reach for speech-to-speech only where naturalness is the point.

## Reference module layout

A complete voice assistant is these concerns, kept separate:

1. **Wake gate** — VAD + KWS + state machine (custom, in front).
2. **Conversation core** — the adopted framework session (cascade or S2S).
3. **Brain** — the agent/LLM logic and tools (see `langgraph-agent-patterns`).
4. **System actions** — browser/OS control invoked as structured tool calls (see `safe-computer-use`).
5. **Control surface** — a local API + UI to configure and monitor everything.

## Observability (non-negotiable for voice)

Voice bugs are timing bugs and you cannot see them without traces.

- Log **per-stage latency** for every turn (VAD -> STT -> LLM first token -> TTS first audio).
- Log the **state machine transitions** and the **transcript** of each turn.
- Expose a live debug view (framework debuggers like Pipecat's Whisker/Tail, or your own) so a human can watch turns in real time.
- Keep an audit trail of any actions the assistant took (ties into `safe-computer-use`).

## Testing voice systems

- Use **mock providers** implementing the same interfaces so the suite runs with no API keys and no microphone.
- Drive the state machine with **recorded audio fixtures** (a wake clip, a command clip, silence) instead of a live mic.
- Assert on transitions and emitted events, not on audio bytes.

## Design-review checklist

Before approving any voice-agent design, confirm:

- [ ] Build-vs-adopt decided explicitly (Pipecat / LiveKit / from-scratch) with rationale.
- [ ] Wake detection uses a dedicated KWS engine, NOT continuous STT + string match.
- [ ] VAD is a real model, not a bare dB threshold.
- [ ] Explicit IDLE/ACTIVE state machine with events on transitions.
- [ ] Echo cancellation (or half-duplex gating) and barge-in are handled.
- [ ] End-of-session is phrase **and** timeout, not phrase alone.
- [ ] Everything streams; per-stage latency is budgeted and measured.
- [ ] Conversation provider abstraction allows cascade OR speech-to-speech.
- [ ] Mock providers + recorded-audio fixtures make the suite runnable offline.
