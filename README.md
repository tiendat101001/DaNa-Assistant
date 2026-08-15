# DaNa — always-on voice assistant

An always-listening desktop voice assistant: wakes on **"Hey DaNa"**, holds a
low-latency spoken conversation, can act on the browser and OS through gated
tool calls, and returns to idle on **"Bye DaNa"** or a timeout. Every heavy
component (wake word, STT, LLM, TTS, actions) is provider-swappable from config.

This is a monorepo with two codebases and one shared config contract:

- **[dana-agent/](dana-agent/)** — the Python agent process (wake gate, session,
  brain, actions, provider plugins). **Phase 1 delivered:** provider framework,
  config schema, mock providers, offline test suite.
- **dana-ui/** — the management dashboard (Phase 6, not yet started).

## Documentation

- **[docs/PLAN.md](docs/PLAN.md)** — clarifying answers, phased roadmap, repo
  layout, control-API contract, latency budget, security commitments.
- **[docs/adr/ADR-0001-realtime-engine.md](docs/adr/ADR-0001-realtime-engine.md)**
  — real-time engine decision: adopt **Pipecat** (LiveKit fallback).
- **[docs/adr/ADR-0002-ui-stack.md](docs/adr/ADR-0002-ui-stack.md)** — UI stack:
  FastAPI control plane in the agent + React dashboard.

## Quick start

```bash
cd dana-agent
uv sync --dev
uv run python -m dana_agent   # validate + assemble the mock stack (no keys needed)
uv run pytest                 # offline test suite
```

## Target profile (from the planning turn)

Windows-first, background/tray process + localhost web dashboard · latency-first
(cloud fast-path default, opt-in local-only privacy mode) · English + Vietnamese
· standard action gates (destructive actions require confirmation).
