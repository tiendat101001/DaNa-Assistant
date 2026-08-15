# dana-agent

The always-on voice assistant process for **DaNa**. Wake gate, conversation
session, brain, actions, and provider plugins live here. See
[../docs/PLAN.md](../docs/PLAN.md) for the phased plan and
[../docs/adr/](../docs/adr/) for the architecture decisions.

## Status: Phase 1 (framework) + Phase 2 (wake gate) delivered

Delivered:

- **Provider framework** — one ABC per capability (`VADProvider`,
  `WakeWordProvider`, `STTProvider`, `LLMProvider`, `TTSProvider`,
  `S2SProvider`, `ActionProvider`), a self-registration registry + typed
  factories, and declared `Capabilities` validated at startup.
- **Typed config** (`pydantic-settings`) as the single source of truth;
  precedence env > `dana.toml` > defaults; secrets as `SecretStr`.
- **Mock providers** for every capability — deterministic, offline, the default
  backend for tests and local dev.
- **Wake gate** (`dana_agent.wake`) — VAD gate → KWS → an explicit IDLE/ACTIVE
  state machine (enum + transition table) that emits an event on every edge,
  forwards audio to the session only while ACTIVE, and ends on the end phrase
  **and** an inactivity timeout (deterministic via an injected clock).
- **Offline test suite** — runs with no network, no secrets, no microphone.

Not yet built (later phases, tracked in [../backlog/](../backlog/)): the Pipecat
conversation core, real provider backends, the LangGraph brain, the action
layer, and the FastAPI control API + dashboard.

## Setup

```bash
cd dana-agent
uv sync --dev          # create venv + install (core deps only; extras are per-phase)
uv run python -m dana_agent   # load config, validate + assemble the mock stack, report ready
```

## Develop

```bash
uv run pytest          # offline test suite (no keys needed)
uv run ruff check .    # lint
uv run ruff format .   # format
uv run mypy            # strict type check
```

## Configuration

Copy `.env.example` to `.env` (or `dana.example.toml` to `dana.toml`) and edit.
Every backend is selected by a `provider` key and swapped without touching code.
Self-hosted / OpenAI-compatible endpoints use `base_url`, not a new provider.

One shared settings object is the single source of truth for the whole project:

```python
from dana_agent.config import get_settings

settings = get_settings()  # cached AppConfig, loaded from env > .env / dana.toml > defaults
```

Brain conversation memory is a configurable checkpointer
(`DANA_BRAIN__MEMORY__BACKEND`): `memory` (in-process) or `sqlite` with
`SQLITE_PATH` as a file (durable across restarts) or `":memory:"` (SQLite
in-memory DB).

## Adding a provider

1. Implement the capability's ABC in `src/dana_agent/providers/<capability>/<name>.py`.
2. Decorate it: `@register("<capability>", "<name>")`; import heavy SDKs inside
   the module so a missing extra breaks only that provider.
3. Declare accurate `Capabilities`. The startup validator does the rest.
4. Import it from the capability package `__init__.py` and run the contract suite.
