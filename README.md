# DaNa — always-on voice assistant

[![CI](https://github.com/tiendat101001/DaNa-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/tiendat101001/DaNa-Assistant/actions/workflows/ci.yml)

Always-listening desktop voice assistant for Windows. Wake on **"Hey DaNa"**, hold a
low-latency spoken conversation, act on the browser and OS through gated tool calls,
return to idle on **"Bye DaNa"** or a 30-second timeout. Every heavy component — STT,
LLM, TTS, wake detection, actions — is swappable from config with no code changes.

```
mic → Silero VAD → AssemblyAI wake detection
    → AssemblyAI STT → LangGraph brain (Gemini) → Gemini TTS → speaker

control API (127.0.0.1:8760) → Talk UI + dashboard (served by the agent)
```

## Repository layout

```
dana-agent/   Python agent process (this is what you run)
dana-ui/      React Talk UI + dashboard (built bundle served by the agent)
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
| Node.js | ≥ 20 | optional — only for developing the UI (`npm run dev`); the agent serves the pre-built bundle otherwise |
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
uv sync --dev --extra all     # everything: cloud + brain + actions + control
```

> ⚠️ **`uv sync` REPLACES the extra set** — it installs exactly what you ask
> for and *uninstalls* the rest. A bare `uv sync` leaves only the dev deps
> (no pipecat/uvicorn/Gemini SDK) and the agent won't start with
> `ModuleNotFoundError`. Always include your extras, or just use `--extra all`
> as above. Individual extras (`cloud`, `brain`, `actions`, `control`, …)
> are listed in [`pyproject.toml`](dana-agent/pyproject.toml) if you want a
> leaner set.

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

### 1 — Talk UI (recommended)

The agent hosts the dashboard itself — voice conversation, live transcript,
confirmations, typed fallback, all in the browser:

```bash
cd dana-agent
uv run python -m dana_agent --control
```

1. Open **http://127.0.0.1:8760** (the startup log prints the URL).
2. First login only: paste the bearer token from `dana-agent/.dana/control-token`
   (auto-generated on first start; the browser remembers it).
3. On the **🎙 Talk** tab, press **Talk** and just speak — no wake phrase needed.
   Or type in the message box; DaNa answers out loud either way.

The same command runs the full voice pipeline (wake-gated for hands-free use)
plus the control API. The **🛠 Dashboard** tab has config editing, the latency
waterfall, and the action audit trail. Stop with **Ctrl+C**.

Make it permanent with `DANA_CONTROL__ENABLED=true` in `.env` — then plain
`uv run python -m dana_agent` starts it too.

**UI development mode** (hot reload; only when changing UI code) — keep the
agent running in one terminal, then:

```bash
cd dana-ui
npm install
npm run dev        # http://localhost:5173, proxies /api to the agent
```

After UI changes, refresh the agent-served bundle with `npm run build`.

### 2 — Validate config (no keys, no mic needed)

Checks that providers can be assembled from the current config and exits. Good for
CI or confirming a new `.env` is wired correctly.

```bash
cd dana-agent
uv run python -m dana_agent --validate-only
```

### 3 — Development mode (CLI, skip wake phrase)

Bypasses wake-word detection — every mic frame goes straight to STT. Speak freely
without saying "Hey DaNa" first. Requires `ASSEMBLYAI_API_KEY` + `GEMINI_API_KEY`.

```bash
uv run python -m dana_agent --bypass-wake
```

#### With UI

```bash
uv run python -m dana_agent --bypass-wake --control
```

### 4 — Full mode (wake-phrase gated, no UI)

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

## Self-hosted models & realtime

Run speech on-device (LiteRT) and pick a fast LLM, all from the UI. Latency
target: first audio within **600 ms** of you finishing a sentence.

1. Install everything: `cd dana-agent && uv sync --dev --extra all`, build the UI
   (`cd dana-ui && npm run build`) and start the agent as usual.
2. Open the UI, paste the control token, and go to the **Models** tab.
3. **Pull & load**: browse `litert-community` (filter ASR / TTS / LLM), open a
   model to see the recommended files, then *Pull & Load* (e.g.
   `moonshine-tiny` for ASR, `kitten-tts-nano-0.8` for English TTS). Public
   repos need no token; set `HF_TOKEN` only for gated ones.
4. **Assign**: in *Local models* click *Use as ASR* / *Use as TTS*. The change
   is written to config and hot-applied where possible.
5. **LLM hosting**: pick a preset (Together, SGLang/vLLM, Ollama, ...), *Apply*,
   then *Measure latency* to see time-to-first-token. Together needs
   `TG_API_KEY`; see `dana-agent/.env.example` for the variables.
6. **Latency budget**: the top card splits the 600 ms goal into VAD stop, STT,
   LLM first token and TTS first byte; tune `DANA_CONVERSATION__LATENCY__*`.
7. **Computer use**: with `DANA_ACTIONS__PROVIDER=browser` and
   `ALLOWED_DOMAINS` set, the agent uses `browser_snapshot` (numbered element
   refs) then `browser_act(ref, action)`. Destructive clicks (delete, pay, ...)
   ask for confirmation; links leaving the allowlist are refused.
8. **MCP**: add servers under `DANA_ACTIONS__MCP_SERVERS` (stdio or HTTP). Their
   tools appear to the agent through the same gate: read-only tools run, the
   rest require confirmation.

Vietnamese TTS uses Piper (`TTS__PROVIDER=piper`); LiteRT TTS is English only.

---

## Provider quick-reference

| Stage | Default | Alternatives |
|---|---|---|
| VAD | Silero (on-device) | — |
| Wake detection | AssemblyAI STT + phrase check | Talk button (UI, no key needed) |
| STT | AssemblyAI streaming | Deepgram, faster-whisper (local) |
| LLM | Gemini (via LangChain) | OpenAI, Anthropic, any OpenAI-compat endpoint |
| TTS | Gemini (streaming chunks) | ElevenLabs, Piper (local) |
| Actions | mock | `browser` (Playwright), `os` (Windows), `browser_os` (both) |

Switch providers by changing `DANA_CONVERSATION__<STAGE>__PROVIDER` in `.env`.

---

## What DaNa can do (voice commands)

Enable the action layer in `.env` (domains/apps are **default-deny** — opt in
per site; OS actions are off unless enabled):

```bash
DANA_ACTIONS__PROVIDER=browser_os
DANA_ACTIONS__OS_ACTIONS_ENABLED=true
DANA_ACTIONS__ALLOWED_DOMAINS=["youtube.com","google.com","duckduckgo.com","wikipedia.org","arxiv.org"]
```

Then, in English or Vietnamese:

- **Media** — "open YouTube and play lofi hip hop" (one command), "pause",
  "jump forward 42 seconds", "tua lui 5 giây", "skip to 3:20", "1.5× speed",
  mute, next/previous, "what's playing?"
- **Web** — search Google/Wikipedia/DDG/arXiv, read and summarize pages, translate
  paragraphs and explain formulas (read the page, the brain does the rest),
  page zoom in/out/reset, scroll
  ("kéo xuống phần bình luận"), tabs ("open a new tab", "switch to the
  YouTube tab"), history ("go back", "that video from earlier"), downloads
  (executables refused)
- **Windows** — system volume ("volume 30"), launch allowlisted apps, focus/
  snap/minimize windows, global media keys ("what song is this?" needs the
  optional `winsdk` package)

**Safety model:** the LLM only proposes structured tool calls; a non-LLM gate
classifies each by consequence — read-only runs free, destructive actions
(send/buy/delete/close-window, incl. Vietnamese labels) pause for voice or UI
confirmation, off-allowlist is refused. Every decision lands in an append-only
audit log (Dashboard tab). Set `DANA_ACTIONS__FORBID_PURCHASES=true` to refuse
purchase clicks outright.

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
uv run pytest                     # offline suite (~370 tests, no network)
uv run pytest -m live             # live provider + adapter probes (real keys;
                                  #   adapter probes also need:
                                  #   uv sync --extra actions &&
                                  #   uv run playwright install chromium)
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

**Model-routing evals** (run when tool descriptions or prompts change — they
call the real Gemini model from `.env`):

```bash
uv run python scripts/action_routing_eval.py     # 40 utterances (EN+VI) → expected tool calls
uv run python scripts/action_chaining_eval.py    # 6 multi-turn scenarios through the real Brain
```
