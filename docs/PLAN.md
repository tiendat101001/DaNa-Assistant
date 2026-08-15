# DaNa — Phased Build Plan

**Status: awaiting your approval. No production code will be written until you approve this plan (brief §0).**

Companion ADRs: [ADR-0001 Real-time engine → Pipecat](adr/ADR-0001-realtime-engine.md) · [ADR-0002 UI stack → FastAPI + React](adr/ADR-0002-ui-stack.md)

---

## 1. Clarifying questions — answered (brief §8)

| # | Question | Your answer |
|---|---|---|
| 1 | Platform / form factor | **Windows 11 first**; always-on background/tray process; **web dashboard** on localhost |
| 2 | Priorities | **Latency > privacy > cost** — cloud fast-path defaults; privacy via mic-mute + opt-in local-only mode |
| 3 | Languages | **English + Vietnamese at v1** |
| 4 | Action autonomy | **Standard gates** — read-only free; reversible writes logged; destructive always confirmed |
| 5 | v1 providers | **Fast cloud set + Local set + extra brain vendors** (no speech-to-speech at v1; abstraction still supports it) |
| 6 | Infra | **Cloud API keys in hand**; no GPU, no self-hosted endpoints → local set is CPU-tuned |

**Implications baked into this plan:**
- Default stack (latency-first, vi-capable): **Deepgram Nova-3** streaming STT (`vi` + `en`) → **Gemini Flash / GPT-4o-mini-class** brain → **ElevenLabs Flash v2.5** TTS → **Porcupine** wake ("Hey DaNa" custom keyword, free tier).
- Local swap set (CPU-only, also the "local-only privacy mode"): **faster-whisper** (small/int8) → any **OpenAI-compatible `base_url`** brain → **Piper** (`vi_VN` + `en_US` voices) → **openWakeWord**.
- Extra brains at v1: **Anthropic** and **Gemini/OpenAI** as native providers; **DeepSeek, Grok, Ollama, vLLM** all via the single OpenAI-compatible provider with configurable `base_url` (one class, not four).
- Language handling v1: a configured active language (`en` or `vi`) switchable per session from UI/config; capability validation guarantees the chosen STT/TTS combo covers both. Automatic mid-conversation code-switching is explicitly deferred (hard problem; Deepgram multi + multilingual TTS make it reachable later).
- Wake/end phrases "Hey DaNa" / "Bye DaNa" are phonetically English — unaffected by language choice.

## 2. Phase 0 decision — Pipecat (summary; full reasoning in ADR-0001)

**Adopt Pipecat.** Its in-process local-audio pipeline is exactly DaNa's production shape (one always-on process owning mic → pipeline → speakers, no media server), its frame-processor architecture gives first-class insertion points for the custom wake gate and latency taps, and its provider catalog covers every row of the brief's matrix including local CPU providers. LiveKit Agents remains the documented fallback (its server+room topology is overhead for a single-machine assistant; it wins if the AEC spike fails or multi-device access becomes a requirement). From-scratch is rejected outright.

The one genuine risk — **acoustic echo cancellation on a bare local mic** — has a four-step mitigation ladder and hard spike exit criteria in ADR-0001. The ADR is confirmed or overturned by Phase 0, not by opinion.

## 3. Roadmap

Ordering per brief §9.3, with scaffolding pulled ahead of the wake gate because everything depends on the config/provider framework. Estimates are focused working days, not calendar promises.

| Phase | Scope | Key deliverables | Exit criteria |
|---|---|---|---|
| **0 — Spike & ADRs** (~2–3 d) | Throwaway Pipecat pipeline on this machine; brief LiveKit console-mode comparison | Latency + AEC + barge-in measurements; ADR-0001 confirmed | All 5 spike criteria in ADR-0001 pass |
| **1 — Scaffold & provider framework** (~2–3 d) | `dana-agent` repo per python-service-scaffold skill: uv, src layout, ruff/mypy strict/pytest, structlog; config schema; registry + factories; **mock providers for every ABC** | Running skeleton app; shared ABC test suite; `.env.example` | `uv run pytest` green **offline, no keys**; unknown-provider/bad-combo errors are actionable |
| **2 — Wake gate & state machine** (~3–4 d) | Silero VAD gate → KWS frame processor; `IDLE/ACTIVE` FSM as enum + transition table emitting events; Porcupine + openWakeWord providers; "Bye DaNa" **and** inactivity timeout | Wake gate processor; recorded-audio fixtures (wake clip, command, silence, vi + en) | FSM integration test passes on fixtures; false-wake sanity check on 1 h of desk audio |
| **3 — Conversation core** (~4–5 d) | Pipecat cascade session behind `ConversationProvider`; cloud + local sets wired; per-stage latency taps; barge-in + AEC production config; language switch en/vi | Working "Hey DaNa → converse → Bye DaNa" loop | Median perceived latency ≤ 500 ms cloud set (stretch; ≤ 700 ms hard); no self-trigger in 30-min session; both sets pass vi + en smoke tests |
| **4 — Brain (LangGraph)** (~3–4 d) | Typed state + agent/tools graph per langgraph-agent-patterns skill; SQLite checkpointer (thread = conversation session); token streaming into TTS; **interrupt-based confirmation gates**; message trim/summarize strategy | Brain module; fake-LLM routing tests; checkpoint/resume test | Tool loop + confirmations work with fake model; tokens stream to TTS with no full-completion waits |
| **5 — Action layer** (~4–6 d) | `ActionProvider` per safe-computer-use skill: Playwright browser tools (role/text locators, page objects, dedicated profile, storage-state auth, domain allowlist) + bounded OS actions (launch allowlisted apps, focus window, media keys, clipboard); reversibility classes; append-only audit log (JSONL); voice + UI confirmation flow | Action registry with allowlist config; audit log; adversarial prompt-injection tests | Gate layer unit tests pass (refusals, confirmations, arg validation); destructive action demonstrably impossible without confirmation |
| **6 — Control API & UI** (~5–7 d) | FastAPI control plane (contract in §5); `dana-ui` React/Vite dashboard: schema-driven config editor, start/stop, live transcript + FSM view, latency waterfall, audit view, mic mute, pending-confirmation prompts | Versioned API; dashboard served by agent on 127.0.0.1 | Edit config → validate → hot-apply (safe fields) round-trips; live events render in UI; agent runs headless unaffected |
| **7 — Hardening** (~4–6 d) | Windows autostart (Task Scheduler) + tray icon; secrets → Windows Credential Manager via `keyring`; **local-only mode** toggle; one S2S provider wired to prove the abstraction; CI (lint/type/test); docs | Installable always-on setup; S2S config option; README/runbooks | Survives reboot; 24 h soak run without leak/crash; suite green in CI |

Total: roughly **4–6 focused weeks**. Each phase ends with a working, demonstrable increment — you can stop or reprioritize at any boundary.

## 4. Repositories

Brief §6 mandates two codebases. Proposal (flag if you disagree): **one git repo (`dana`) containing both packages** — solo developer, one shared contract, atomic cross-cutting changes. Splitting into two repos later is mechanical.

```
dana/
├── docs/                        # this plan, ADRs, API contract
├── dana-agent/                  # Python — the always-on assistant
│   ├── pyproject.toml           # uv-managed; ruff + mypy(strict) + pytest config
│   ├── uv.lock
│   ├── .env.example             # every env var documented; secrets never committed
│   ├── src/dana_agent/
│   │   ├── app.py               # composition root: config → build providers → wire pipeline
│   │   ├── config/              # pydantic-settings schema (single source of truth)
│   │   ├── providers/           # registry + factories + one module per backend
│   │   │   ├── base.py          # ABCs + Capabilities (see §6)
│   │   │   ├── wake/  stt/  tts/  llm/  s2s/
│   │   ├── wake/                # VAD gate, KWS frame processor, IDLE/ACTIVE FSM
│   │   ├── conversation/        # Pipecat pipeline assembly (cascade | s2s), latency taps
│   │   ├── brain/               # LangGraph graph, tools, checkpointing, interrupts
│   │   ├── actions/             # gate/validator, allowlists, audit log, browser/, os_win/
│   │   ├── control/             # FastAPI REST + WS, event bus, schema endpoint
│   │   └── observability/       # structlog setup, latency spans, correlation ids
│   └── tests/
│       ├── fakes/               # mock provider per capability (default test backend)
│       ├── fixtures/audio/      # recorded wake/command/silence clips (en + vi)
│       └── ...                  # ABC contract suite, FSM, gates, routing — all offline
└── dana-ui/                     # TypeScript — management dashboard
    ├── package.json             # React + Vite + TS
    └── src/
        ├── api/                 # typed client generated from OpenAPI + WS event types
        ├── features/            # config-editor (JSON-Schema-driven), transcript, latency,
        │                        # state-machine view, audit, confirmations, mic control
        └── ...
```

## 5. Control-API contract (agent ⇄ UI)

Bound to `127.0.0.1` only; bearer token from a local file; versioned under `/api/v1`.

**REST (config & control):**

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/config` | Current config, secrets redacted |
| `PUT /api/v1/config` | Validated write; response labels each change `applied` (hot-reloaded) or `restart_required` |
| `GET /api/v1/config/schema` | JSON Schema of the pydantic settings — the UI renders forms from this |
| `GET /api/v1/providers` | Registry catalog: name, capability flags (streaming, languages, self_hosted, latency_class) per capability |
| `POST /api/v1/session/start` · `/stop` | Start/stop the voice pipeline (agent process stays up) |
| `GET /api/v1/status` | FSM state, active providers, uptime, session info |
| `POST /api/v1/mic/mute` · `/unmute` | Hard mic control (also on tray icon) |
| `POST /api/v1/confirmations/{id}` | Approve/deny a pending gated action from the UI |
| `GET /api/v1/audit?since=…` | Page through the append-only action audit log |
| `GET /api/v1/health` | Liveness |

**WebSocket `/api/v1/events`** — server-push, typed JSON envelope `{type, ts, session_id, seq, data}`:

| Event type | Payload (data) |
|---|---|
| `state_transition` | `{from, to, cause}` — every FSM edge |
| `transcript_partial` / `transcript_final` | `{role, text, lang}` |
| `assistant_turn` | `{text, interrupted: bool}` |
| `latency_span` | `{turn_id, stage: vad\|stt\|llm_ttft\|tts_ttfb\|playback, ms}` — feeds the waterfall |
| `confirmation_request` | `{id, tool, args_resolved, reversibility, preview}` — mirrors the voice confirmation |
| `audit_entry` | tool, resolved args, decision, result |
| `config_changed` / `error` | diff summary / structured error |

## 6. Provider abstraction & config sketch (illustrative, not final code)

One ABC per capability, same four pieces everywhere (interface / providers / registry+factory / typed config), per the provider-plugin skill. Thin rule from ADR-0001: **STT/TTS/LLM/S2S providers are factories that construct and configure Pipecat service objects** — we do not re-implement streaming clients. Wake and Action providers are fully custom.

```python
class Capabilities(BaseModel, frozen=True):
    streaming: bool
    languages: frozenset[str]          # must cover configured langs, e.g. {"en","vi"}
    self_hosted: bool = False
    tool_calling: bool = False
    latency_class: Literal["realtime", "fast", "batch"] = "fast"

class WakeWordProvider(ABC):           # fully custom (Porcupine / openWakeWord)
    @abstractmethod
    def process(self, frame: AudioFrame) -> WakeEvent | None: ...
    capabilities: Capabilities

class STTProvider(ABC):                # thin: builds a Pipecat STT service
    @abstractmethod
    def build_service(self) -> "pipecat STT service": ...
    capabilities: Capabilities
# LLMProvider / TTSProvider / S2SProvider follow the same shape.
# ConversationConfig selects mode: "cascade" (stt+llm+tts) | "s2s" (one provider).

@register("stt", "deepgram")
class DeepgramSTT(STTProvider): ...    # self-registration; factory names no vendor
```

```python
class AppConfig(BaseSettings):         # env > file (dana.toml) > defaults; DANA__ prefix
    language: Literal["en", "vi"] = "en"
    wake: WakeConfig                   # provider, phrase, sensitivity, timeout_s
    conversation: ConversationConfig   # mode + stt/llm/tts (or s2s) sub-configs
    actions: ActionsConfig             # allowlisted domains/apps, gate policy
    control: ControlConfig             # host=127.0.0.1, port, token_path
    privacy: PrivacyConfig             # local_only: bool, audio_retention
    # secrets: SecretStr fields, sourced from env / Windows Credential Manager
```

Startup validation fails fast with actionable errors: unknown provider name, missing key for a *selected* provider, or a combo whose capabilities don't cover the configured language (e.g. TTS without `vi`).

## 7. Latency budget (cloud set, perceived turn = user stops speaking → first audio out)

| Stage | Budget | Instrument |
|---|---|---|
| Endpointing / turn end detection | 100–200 ms | `latency_span: vad` |
| STT final (streaming, incremental) | 50–150 ms | `stt` |
| LLM first token (Flash-class model) | 200–350 ms | `llm_ttft` |
| TTS first byte (ElevenLabs Flash v2.5) | 75–150 ms | `tts_ttfb` |
| Output buffer / playback start | 20–50 ms | `playback` |
| **Total** | **~450–900 ms** | per-turn waterfall in UI |

Sub-500 ms is achievable but not guaranteed — it requires the fast cloud set, sentence-level TTS streaming, and tuned endpointing. Every stage streams; nothing ever waits for a full transcript, completion, or audio file. The local CPU-only set will land ~1.5–3 s and is positioned as the privacy mode, not the latency mode.

## 8. Security & privacy commitments (brief §7 + safe-computer-use skill)

- LLM emits **structured typed tool calls only**; a non-LLM gate layer validates and executes. No raw shell, no `eval`.
- **Allowlist, default-deny** for actions, browser domains, and OS apps; every argument validated.
- Reversibility classes: read-only → free; reversible → run + audit; destructive → **LangGraph interrupt → voice or UI confirmation** with a preview of exactly what will run.
- **Append-only audit log** of every action (JSONL), surfaced live in the dashboard.
- Secrets in **Windows Credential Manager** (via `keyring`), `SecretStr` in config, redacted in logs/API/UI; the model never sees raw credentials — the execution layer injects them.
- **Mic mute** is a hard control (tray + API) that stops capture, not a soft flag. **Local-only mode** flips the whole stack to the local provider set and blocks non-allowlisted egress.
- Untrusted page/text content the agent reads is **never treated as instructions** — adversarial prompt-injection fixtures are part of the Phase 5 test suite.
- Playwright runs a **dedicated browser profile** (isolated context, storage-state auth), never your daily profile.

## 9. Testing strategy (offline by default)

- Mock provider per capability in `tests/fakes/` is the default backend; **the whole suite runs with no network, no keys, no microphone**.
- One shared contract test suite against each ABC, parametrized over the registry.
- FSM driven by recorded audio fixtures (en + vi); assertions on transitions and events, never audio bytes.
- Brain routing tested with a fake model returning canned tool calls; checkpoint/resume tested explicitly.
- The action gate layer gets the hardest tests: refusals, confirmation bypass attempts, path/domain validation, injection fixtures.
- Live-provider smoke tests exist but are opt-in (`-m live`), never in CI.

## 10. Risks

| Risk | Mitigation |
|---|---|
| AEC on local mic (top risk) | ADR-0001 mitigation ladder; hard Phase 0 exit criteria; LiveKit fallback documented |
| Sub-500 ms miss | Budget + per-stage measurement from Phase 3 day one; 700 ms hard gate; endpointing tuning |
| Vietnamese quality variance (STT tones, TTS naturalness) | vi fixtures in the suite from Phase 2; Deepgram Nova-3 primary, faster-whisper fallback; ElevenLabs primary, Piper `vi_VN` fallback |
| openWakeWord "Hey DaNa" needs a trained model | Porcupine (custom keyword in minutes, free tier) is the v1 default; oWW training is a follow-up task, not a blocker |
| Pipecat API churn | Version pinning; all framework imports contained inside the provider layer |
| OS-automation blast radius | Off-by-default, app allowlist, APIs preferred over GUI automation, standard gates |

## 11. What I need from you

1. **Approve or amend this plan** (phases, repo shape in §4, API contract in §5).
2. Confirm **ADR-0001 (Pipecat)** as the working decision pending spike results, and **ADR-0002 (FastAPI + React)**.
3. On approval I start **Phase 0** — spike code is throwaway and clearly separated; production scaffolding begins only at Phase 1.
