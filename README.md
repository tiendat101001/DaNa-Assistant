# DaNa — always-on voice assistant

[![CI](https://github.com/tiendat101001/DaNa-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/tiendat101001/DaNa-Assistant/actions/workflows/ci.yml)

Always-listening desktop voice assistant for Windows. Wake on **"Hey DaNa"**, hold a
low-latency spoken conversation, act on the browser and OS through gated tool calls,
return to idle on **"Bye DaNa"** or a 30-second timeout. Every heavy component — STT,
LLM, TTS, wake detection, actions — is swappable from config with no code changes.

```
mic → Silero VAD → AssemblyAI wake detection
    → AssemblyAI STT → LangGraph brain (Gemini) → Gemini TTS → speaker
```

## Repository layout

```
dana-agent/   Python agent process (this is what you run)
dana-ui/      React dashboard (Phase 6, not yet started)
docs/         Architecture decisions and the phased roadmap
backlog/      Task-level tracking, one file per feature
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | 3.13 recommended |
| uv | any recent | replaces pip + venv |
| pre-commit | any recent | optional — `uv tool install pre-commit && pre-commit install` for the same lint/type/secret gates locally |
| Microphone | — | any Windows audio input device |

**Install uv on Windows** (PowerShell, one-time):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installing so `uv` is on `PATH`.

PortAudio (needed for PyAudio) ships bundled in the PyAudio Windows wheel —
no separate install required.

---

## Installation

```bash
cd dana-agent
uv sync --dev          # installs core + dev deps (brain, test suite)
uv sync --extra cloud  # adds pipecat, PyAudio, AssemblyAI, Gemini, ElevenLabs
```

> The `--dev` sync alone is enough for the offline test suite and `--validate-only`.
> Add `--extra cloud` before running a live session.

**If uv is not yet on PATH** — the venv already exists at `dana-agent/.venv/`.
Invoke Python directly until uv is installed:

```powershell
.\.venv\Scripts\python.exe -m dana_agent --validate-only
.\.venv\Scripts\python.exe -m pytest
```

---

## Configuration

Copy the example and fill in your API keys:

```bash
cp .env.example .env
```

API keys use standard vendor names — one variable covers all stages that use the
same provider:

```bash
# .env — minimum for the cloud stack (AssemblyAI + Gemini)
ASSEMBLYAI_API_KEY=your_key   # STT + wake detection
GEMINI_API_KEY=your_key       # LLM + Gemini TTS
```

Select providers via `DANA_` prefixed settings:

```bash
DANA_CONVERSATION__STT__PROVIDER=assemblyai   # assemblyai | deepgram | mock
DANA_CONVERSATION__LLM__PROVIDER=gemini       # gemini | openai | anthropic | mock
DANA_CONVERSATION__TTS__PROVIDER=gemini       # gemini | elevenlabs | mock
```

See [`.env.example`](dana-agent/.env.example) for the full list of options.

---

## Running

### 1 — Validate config (no keys, no mic needed)

Checks that providers can be assembled from the current config and exits. Good for
CI or confirming a new `.env` is wired correctly.

```bash
cd dana-agent
uv run python -m dana_agent --validate-only
```

### 2 — Development mode (skip wake phrase)

Bypasses wake-word detection — every mic frame goes straight to STT. Speak freely
without saying "Hey DaNa" first. Requires `ASSEMBLYAI_API_KEY` + `GEMINI_API_KEY`.

```bash
uv run python -m dana_agent --bypass-wake
```

### 3 — Full mode (wake-phrase gated)

```bash
uv run python -m dana_agent
```

Full session lifecycle:

1. Say **"Hey DaNa"** — Silero VAD clips the utterance, AssemblyAI transcribes it,
   the gate checks for the phrase and enters ACTIVE.
2. Speak normally — AssemblyAI streams your speech to the LangGraph brain.
3. DaNa responds via Gemini TTS (streams audio chunks progressively).
4. Say **"Bye DaNa"** or wait 30 s of silence — gate returns to IDLE.

Press **Ctrl+C** for a clean shutdown at any point.

### Mock mode (fully offline, no keys)

Set all providers to `mock` in `.env` or run with defaults — no API keys or
network access required. The mock stack validates config and provider wiring.

```bash
# No .env needed — defaults are all mock
uv run python -m dana_agent --validate-only
```

---

## Provider quick-reference

| Stage | Default | Alternatives |
|---|---|---|
| VAD | Silero (on-device) | — |
| Wake detection | AssemblyAI STT + phrase check | — |
| STT | AssemblyAI streaming | Deepgram, faster-whisper (local) |
| LLM | Gemini (via LangChain) | OpenAI, Anthropic, any OpenAI-compat endpoint |
| TTS | Gemini (streaming chunks) | ElevenLabs, Piper (local) |
| Actions | mock | Playwright (Phase 5) |

Switch providers by changing `DANA_CONVERSATION__<STAGE>__PROVIDER` in `.env`.

---

## Logs

Each session appends to `dana-agent/logs/YYYY-MM-DD.log` (JSON, one line per event). The directory is created automatically on startup. Entries from both DaNa (structlog) and Pipecat (loguru) land in the same file.

**Tail live** (PowerShell, from the repo root):

```powershell
Get-Content dana-agent\logs\2026-08-17.log -Wait
```

**Pretty-print a past session** (requires `jq`):

```powershell
Get-Content dana-agent\logs\2026-08-17.log | jq .
```

The `logs/` folder is gitignored — log files never end up in the repository.

---

## Tests

```bash
cd dana-agent
uv run pytest                     # offline suite (~79 tests, no network)
uv run pytest -m live             # live provider tests (needs real API keys)
uv run pytest --no-cov -q        # faster, no coverage report
```

The offline suite uses mock providers throughout — no API keys, no microphone.
Live tests are opt-in and marked `@pytest.mark.live`.

---

## Development

```bash
uv run ruff check src/     # lint
uv run ruff format src/    # format
uv run mypy src/           # type-check (strict)
```

---

## Documentation

- **[docs/PLAN.md](docs/PLAN.md)** — phased roadmap, latency budget, security commitments, control-API contract
- **[docs/adr/ADR-0001-realtime-engine.md](docs/adr/ADR-0001-realtime-engine.md)** — why Pipecat
- **[docs/adr/ADR-0002-ui-stack.md](docs/adr/ADR-0002-ui-stack.md)** — FastAPI + React dashboard plan
- **[backlog/](backlog/)** — task-level tracking (TASK-001 through TASK-010)
