# TASK-003 — Implement real providers behind the ABCs

- **Status:** TODO
- **Priority:** P1
- **Source:** docs/PLAN.md §1 (v1 provider selection), provider-plugin skill.

## Goal

Fill the registry slots with real backends. Each is a thin adapter over a vendor
SDK (or a Pipecat service in Phase 3), imports its SDK lazily, declares accurate
`Capabilities`, and passes the shared ABC contract suite. Keys come from `.env`;
live smoke tests are opt-in (`-m live`), never in CI.

## Providers to implement

**Fast cloud set (latency-first default):**
- [ ] STT: Deepgram Nova-3 (streaming, en + vi)
- [ ] STT: AssemblyAI (streaming)
- [ ] LLM: Gemini, OpenAI, Anthropic (native)
- [ ] TTS: ElevenLabs Flash v2.5 (en + vi)
- [ ] Wake: Porcupine / Picovoice ("Hey DaNa" custom keyword) — **v1 default**

**Local set (CPU-only; also local-only privacy mode):**
- [ ] STT: faster-whisper (int8)
- [ ] LLM: OpenAI-compatible via `base_url` (covers DeepSeek, Grok, Ollama, vLLM)
- [ ] TTS: Piper (`vi_VN` + `en_US`), Kokoro
- [ ] Wake: openWakeWord
- [ ] VAD: Silero VAD

**Speech-to-speech (one, to prove the abstraction — Phase 7):**
- [ ] OpenAI Realtime **or** Gemini Live

## Acceptance criteria

- [ ] Each provider passes the ABC contract suite (offline where mockable).
- [ ] Capability flags accurate (streaming, languages, tool_calling, self_hosted).
- [ ] Missing SDK/extra breaks only that provider, not app import.
- [ ] Live smoke test per provider behind the `live` marker.
