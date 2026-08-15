# DaNa — Always-On Voice Assistant · Build Brief for Claude Code (Fable 5)

## 0. How to use this brief (read first)
Before writing any code, do the following **in order**:
1. Ask me the clarifying questions in §8 — do not assume answers.
2. Produce a **phased plan + architecture decision records (ADRs)** in markdown for my review.
3. Scaffold repos and write code **only after I approve the plan**.
Never run destructive commands; show a dry-run / preview and wait for confirmation.

---

## 1. Goal
A desktop / personal voice assistant ("**DaNa**") that is always listening, wakes on **"Hey DaNa"**, holds a natural low-latency spoken conversation, can act on my **browser and OS**, and returns to idle on **"Bye DaNa"** or after inactivity. Every heavy component (wake word, STT, LLM, TTS, actions) must be **provider-swappable via config**.

---

## 2. Strategic constraint — adopt the real-time engine, don't rebuild it
The pluggable STT↔LLM↔TTS loop with turn detection, barge-in (interruption), and acoustic echo cancellation is a solved problem. Two mature open-source Python frameworks provide it with 20+ swappable providers per layer: **Pipecat** (Daily) and **LiveKit Agents**.

- **Phase 0 spike:** evaluate Pipecat vs LiveKit vs from-scratch; pick one and record the decision as an ADR.
- Satisfy my provider-abstraction requirement **through the framework's plugin system + a thin config layer**, not a from-scratch audio loop.
- Reserve custom engineering for what is actually mine: the **wake-word gate**, the **LangGraph brain**, the **OS/browser action layer**, and the **management UI**.
- Both frameworks support **speech-to-speech models** (OpenAI Realtime, Gemini Live, Grok Voice, Ultravox, Nova Sonic) as swappable providers — design for this (see §4).

---

## 3. Corrected wake-word mechanism
**Do NOT** run STT continuously and string-match "Hey DaNa" — high latency, high cost, low accuracy. Use a dedicated always-on, on-device keyword-spotting (KWS) engine.

- **Providers (swappable):** `openWakeWord` (Apache-2.0, free, custom phrase needs training) or `Porcupine`/Picovoice (commercial, custom word in seconds, very accurate).
- **Pipeline:** mic → lightweight VAD (Silero / webrtcvad) as a cheap gate → KWS engine listens for "Hey DaNa" → on hit, transition state and hand the audio stream to the conversation session.
- **End of session:** detect "Bye DaNa" (KWS or intent in the loop) **or** an inactivity timeout → return to idle.
- **State machine (explicit):** `IDLE → (wake) → ACTIVE → (bye / timeout) → IDLE`.
- Because DaNa is always-on and speaks aloud, **acoustic echo cancellation** and **barge-in** are mandatory — another reason to sit on Pipecat/LiveKit.

---

## 4. Modules (each behind a provider abstraction)
Each module exposes a common interface (ABC) and is selected at runtime from config. A **conversation provider** may be **either** a cascade (STT + LLM + TTS) **or** a single **speech-to-speech** model — the abstraction must allow both. Default to cascade; use speech-to-speech only where naturalness is the point.

| Module | Interface | Providers (v1 = swappable) |
|---|---|---|
| Wake-Word | `WakeWordProvider` | openWakeWord, Porcupine |
| Core Brain | LangGraph/LangChain agent (state graph, tools, memory/checkpointing, token streaming) | Gemini, OpenAI, OpenAI-compatible/self-hosted, DeepSeek, Grok, Anthropic |
| Communication — STT | `STTProvider` (streaming) | faster-whisper / WhisperLive (self-hosted), AssemblyAI, Deepgram |
| Communication — TTS | `TTSProvider` (streaming) | Piper / Kokoro (local), ElevenLabs, Cartesia, OpenAI |
| System-Interaction | `ActionProvider` | Browser via Playwright; bounded OS/app actions |

**System-Interaction safety (non-negotiable):** the LLM emits **structured tool calls**, never raw shell. Enforce an **action allowlist**, **confirmation for destructive/irreversible actions**, **dry-run/preview**, and an **audit log**. Playwright: locators + auto-wait + page-object structure.

---

## 5. Provider abstraction design
- One ABC per capability: `WakeWordProvider`, `STTProvider`, `LLMProvider`, `TTSProvider`, `S2SProvider`, `ActionProvider`.
- **Registry + factory:** providers self-register; selection is config-driven.
- **Config:** one typed schema (`pydantic-settings`); precedence **env > file > defaults**; secrets via `.env`/secret store; hot-reload where feasible. This schema is the **shared contract with the UI** (§6).
- **Capability flags** declared per provider (streaming?, languages?, self-hosted?, latency class) so a chosen combo can be validated before start.

---

## 6. Two codebases + their contract
- **`dana-agent` (Python):** the always-on assistant — wake, session, brain, actions, provider plugins. Packaged with `uv`, fully async, structured logging, testable with **mock providers + recorded-audio fixtures** (no live keys needed to run tests).
- **`dana-ui` (management dashboard):** view/edit all config (providers, URLs, ports, keys, wake phrase, dB/VAD thresholds), start/stop/monitor the agent, live status + transcript stream, per-stage latency metrics.
- **Contract:** the agent exposes a **local control API** — **WebSocket** for live status/transcripts/latency, **REST** for config CRUD — over the shared config schema as single source of truth. Specify this API in the plan.
- Propose and justify the UI stack (e.g. FastAPI in the agent + React/Next, or a Tauri/Electron desktop shell).

---

## 7. Cross-cutting requirements
- **Latency:** target **sub-500 ms** perceived turn latency; stream at every stage.
- **Security/privacy:** always-listening + OS/browser control is a large attack surface — mic **mute control**, **local-only audio** option, **credential vault**, action allowlist + confirmations, full **audit log**.
- **Observability:** per-stage latency traces, structured logs, a debug view of the state machine + transcripts.
- **Testing:** unit tests per provider against the ABC; integration test of the state machine with mock audio; suite runs offline.
- **Platform:** state target OS(es) first (Windows/macOS/Linux) — drives audio I/O and OS-automation choices.

---

## 8. Clarifying questions to answer before planning
1. Primary OS/platform first? Desktop app or background service?
2. Priority order: latency vs cost vs privacy? (drives local vs cloud, cascade vs speech-to-speech default)
3. Languages — English only, or Vietnamese too? (affects STT/TTS/wake choices)
4. Which providers are required at v1 vs later?
5. How much autonomy for OS/browser actions, and where must a human confirm?
6. Existing infra to target (self-hosted Whisper, GPUs)?

---

## 9. Deliverables for the planning turn (no production code yet)
1. Clarifying questions answered/asked.
2. Phase 0 build-vs-adopt evaluation (Pipecat vs LiveKit vs from-scratch) → recommendation as an ADR.
3. Phased roadmap: Phase 0 spike → wake gate → conversation core → brain → actions → UI → hardening.
4. Repo layout for both codebases + the control-API contract.
5. Provider ABC + config-schema sketch.
