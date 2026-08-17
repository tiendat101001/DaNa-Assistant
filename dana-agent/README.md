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

Also delivered in later phases: the Pipecat conversation core, real provider
backends, the LangGraph brain, the action layer, the FastAPI control API +
dashboard, and Phase 7 hardening (secrets vault, `local_only`, Windows autostart,
system tray, one S2S provider end to end).

## Hardening (Phase 7)

- **Secrets in the OS vault** — API keys resolve from the Windows Credential
  Manager (`keyring`) first, then the environment / `.env` (dev only).
  Precedence: an explicit `DANA_*__API_KEY` > vault > plain env. Manage keys
  without echoing them to shell history:

  ```bash
  uv run python -m dana_agent secrets set ASSEMBLYAI_API_KEY   # prompts / reads stdin
  uv run python -m dana_agent secrets get ASSEMBLYAI_API_KEY
  uv run python -m dana_agent secrets delete ASSEMBLYAI_API_KEY
  ```

- **`local_only` privacy mode** (`DANA_PRIVACY__LOCAL_ONLY=true`) — a
  **construction-time policy**, not an OS firewall. It refuses to *build* any
  conversation provider that could send audio/text off the machine: every
  selected provider must have a loopback/private `base_url` **and** advertise
  `Capabilities.self_hosted`; a cloud provider (or a public `base_url`) fails
  fast at startup with a `ConfigError`. It cannot stop a dependency that ignores
  the policy and does not block egress at the OS level — for hard network
  isolation, use Windows Firewall rules. Under `local_only` the control API is
  also required to bind to loopback.

- **Windows autostart** (Task Scheduler, logon trigger, least privilege):

  ```bash
  uv run python -m dana_agent autostart install    # register the logon task (idempotent)
  uv run python -m dana_agent autostart status
  uv run python -m dana_agent autostart uninstall
  ```

- **System tray** (with `--control`) — mute/unmute the mic (a real hard mute at
  the capture source), open the dashboard, quit. Needs the `control` extra
  (`pystray` + `pillow`); absent, the control API still runs headless.

- **Speech-to-speech** — set `DANA_CONVERSATION__MODE=s2s`. The `mock` S2S proves
  the path offline; `openai_realtime` is the reference cloud provider (live).

See [../docs/RUNBOOK.md](../docs/RUNBOOK.md) for install, autostart, tray, the
`local_only` walkthrough, and the manual soak / reboot procedures.

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
