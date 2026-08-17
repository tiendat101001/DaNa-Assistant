# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What DaNa is

An always-on Windows voice assistant: wakes on "Hey DaNa", holds a low-latency
spoken conversation, can act on the browser/OS, and returns to idle on "Bye DaNa"
or an inactivity timeout. Every heavy component (wake word, STT, LLM, TTS,
actions) is provider-swappable purely from config. Priorities, in order:
**latency > privacy > cost**. Languages: **English + Vietnamese**.

Monorepo with two packages: `dana-agent/` (Python, the assistant — the only one
that currently exists) and `dana-ui/` (a planned React/Vite dashboard). The
build brief is [DaNa-build-brief.md](DaNa-build-brief.md); the phased roadmap and
current status live in [docs/PLAN.md](docs/PLAN.md), [docs/adr/](docs/adr/), and
[backlog/](backlog/) (one file per task). **Note:** `docs/`, `backlog/`, and
`.claude/` are gitignored — read them for context, but they are not version-
tracked.

## Commands

All commands run from `dana-agent/`:

```bash
uv sync --dev              # create venv + install (core + dev; heavy deps are per-phase extras)
uv run pytest              # full offline suite (no network, no keys, no microphone)
uv run pytest tests/test_brain.py::test_name   # single test
uv run pytest -m live      # opt-in live-provider smoke tests — needs real API keys, never in CI
uv run pytest -m 'not live'  # explicitly skip live tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # strict type check
uv run python -m dana_agent --validate-only    # build + validate the provider stack, then exit
uv run python -m dana_agent --bypass-wake      # run pipeline, skip wake gate (no mic/wake engine needed)
```

Heavy/vendor dependencies are **optional extras grouped by phase** in
`pyproject.toml` (`cloud`, `local`, `wake`, `brain`, `actions`, `control`, …), so
the core framework and its offline suite install and run with none of them.
Install an extra with `uv sync --extra <name>`. `mypy` targets 3.13 but the code
must run on 3.11+.

## Architecture

The spine is a **config-driven provider plugin system**. Understand these four
pieces together before changing provider wiring:

1. **ABCs — one per capability** ([providers/base.py](dana-agent/src/dana_agent/providers/base.py)):
   `VADProvider`, `WakeWordProvider`, `STTProvider`, `LLMProvider`, `TTSProvider`,
   `S2SProvider`, `ActionProvider`. All async/streaming, all speak neutral domain
   types from [types.py](dana-agent/src/dana_agent/types.py) — **never a vendor
   SDK's types**. Every provider also declares a frozen `Capabilities` (streaming,
   languages, self_hosted, tool_calling, latency_class) as data.
2. **Registry + factory** ([providers/registry.py](dana-agent/src/dana_agent/providers/registry.py)):
   providers self-register with `@register("<capability>", "<name>")` at import
   time; typed `build_*` helpers construct one from its sub-config and assert the
   ABC. No if/elif ladders, and the factory names no vendor.
3. **Typed config = single source of truth** ([config.py](dana-agent/src/dana_agent/config.py)):
   one `pydantic-settings` schema, a sub-model per capability. Precedence is
   **env > `dana.toml` > defaults**. This same schema is the intended contract
   with the future UI. Secrets are `SecretStr`.
4. **Startup validation** ([capabilities.py](dana-agent/src/dana_agent/capabilities.py)):
   an unknown provider name, a missing key for a *selected* provider, or a
   provider combo whose capabilities don't cover the configured language **fails
   at startup, not mid-conversation**.

**Composition root:** [app.py](dana-agent/src/dana_agent/app.py) `build_stack()`
is the one place that reads config and constructs concrete providers; everything
else receives its dependencies. It returns an `AssembledStack`. Cascade mode
(`stt`+`tts`+`brain`, `s2s=None`) is the default; speech-to-speech mode is the
alternative (`s2s` set, brain/stt/tts None). The brain owns the LLM via
`init_chat_model`, so there is no separate cascade LLM object.

**Config selection:** a self-hosted or OpenAI-compatible backend is the *same*
provider with a `base_url` override — not a new provider class. DeepSeek, Grok,
Ollama, vLLM all go through one OpenAI-compatible brain provider.

### Runtime flow (cascade)

`mic → VAD → wake gate (KWS + IDLE/ACTIVE FSM) → STT → brain → TTS → speakers`.

- **Wake gate** ([wake/](dana-agent/src/dana_agent/wake/)): VAD is a cheap gate so
  the KWS engine never runs on silence; an explicit enum + transition-table FSM
  emits an event on every edge, forwards audio only while ACTIVE, and ends on the
  end phrase **and** an inactivity timeout (never the phrase alone — users forget
  it). The clock is injected for deterministic tests.
- **Conversation core** ([conversation/pipeline.py](dana-agent/src/dana_agent/conversation/pipeline.py)):
  STT/LLM/TTS/S2S providers are **thin factories over Pipecat service objects** —
  we do not re-implement streaming clients (see [ADR-0001](docs/adr/ADR-0001-realtime-engine.md),
  which adopts Pipecat over LiveKit/from-scratch). Provider-ABC backends without
  a native Pipecat service are bridged with a `FrameProcessor` wrapper.
- **Brain** ([brain/](dana-agent/src/dana_agent/brain/)): a LangGraph agent/tools
  graph. `Brain.run_turn` translates LangGraph's event stream into DaNa's neutral
  `BrainEvent`s (`TokenDelta`, `ToolInvoked`, `ToolCompleted`, `ConfirmationRequired`,
  `TurnComplete`, `StepLimitReached`) so the voice pipeline never sees LangChain
  types. Conversation memory is a configurable checkpointer (`memory` in-process,
  or `sqlite` durable/`:memory:`); `thread_id` = one conversation session. Tokens
  stream to TTS as they arrive.

### Action safety (non-negotiable — see [safe-computer-use skill] & PLAN.md §8)

The LLM emits **structured typed tool calls only**; a non-LLM gate layer
validates and executes. No raw shell, no `eval`.

- **Gate** ([actions/gate.py](dana-agent/src/dana_agent/actions/gate.py)):
  classify every action by `Reversibility`, then decide per policy — read-only →
  ALLOW, destructive → CONFIRM (all policies), out-of-scope → REFUSE. The
  `conservative` policy also confirms reversible writes.
- **Tools** ([brain/tools.py](dana-agent/src/dana_agent/brain/tools.py)): a gated
  tool that needs confirmation calls LangGraph's `interrupt` to pause the graph
  for a human decision (resumable via `Brain.resume_turn(approved=...)` because
  state is checkpointed). Nothing executes raw model output.
- Every decision and execution is written to an append-only audit log
  ([actions/audit.py](dana-agent/src/dana_agent/actions/audit.py)).

## Conventions

- **Fully async, structured logging** via structlog
  ([observability/](dana-agent/src/dana_agent/observability/)) — use
  `get_logger(__name__)` and event-style logging (`log.info("event_name", key=val)`).
- **Import heavy/vendor SDKs inside the function or module that needs them**
  (often inside the provider factory), so a missing optional extra breaks only
  that provider, not the whole app or the offline suite.
- **Tests are offline by default** ([tests/conftest.py](dana-agent/tests/conftest.py)):
  mock providers are the default backend, `AppConfig()` builds an all-mock cascade
  stack. Fakes live in `tests/fakes/`, recorded/scripted sequences in
  `tests/fixtures/`. The `registry_snapshot` fixture isolates tests that register
  throwaway providers. Live-provider tests are marked `@pytest.mark.live` and
  excluded from CI.
- A shared **contract test suite** ([tests/test_providers_contract.py](dana-agent/tests/test_providers_contract.py))
  runs against each ABC, parametrized over the registry.
- Windows-first: `__main__.py` guards POSIX-only signal handling behind
  `sys.platform != "win32"`.

## Adding a provider

1. Implement the capability's ABC in `providers/<capability>/<name>.py`.
2. Decorate it `@register("<capability>", "<name>")`; import heavy SDKs inside the
   module.
3. Declare accurate `Capabilities` — the startup validator relies on them.
4. Import it from the capability package's `__init__.py`, then run the contract
   suite.
