# ADR-0002 — Management UI stack: FastAPI control plane in the agent + React web dashboard

- **Status:** Proposed
- **Date:** 2026-08-15
- **Decision drivers:** user chose Windows-first with a **web dashboard** (clarifying Q1); config schema must be the single shared contract (§5–6); solo-maintainable; the agent is the always-on process — the UI must never be required for the assistant to run.

## Context

The brief (§6) requires `dana-ui` to view/edit all config, start/stop/monitor the agent, and stream live status/transcripts/latency, over a local control API (REST for config CRUD, WebSocket for live events). Candidate stacks: FastAPI-in-agent + React web app; Tauri desktop shell; Electron desktop shell.

## Decision

1. **Control plane lives inside `dana-agent`:** a FastAPI app (REST + WebSocket) mounted in the same asyncio process as the pipeline, bound to `127.0.0.1` only. One process = one source of truth for config, state, and events; no IPC layer to design; the pydantic-settings schema is served as JSON Schema directly.
2. **`dana-ui` is a React + TypeScript + Vite single-page app.** In production the agent serves the built static bundle at `http://127.0.0.1:<port>`; in development Vite proxies to the agent. The config editor renders **from the JSON Schema** (with capability flags driving valid choices), so adding a provider or field never requires hand-editing UI forms.
3. **Tray integration** (mic mute, open dashboard, quit) is a lightweight tray icon in the agent process (e.g. `pystray`) — not part of the UI stack.
4. **Electron/Tauri rejected for v1:** a second runtime/build chain and packaging burden to reproduce what a browser tab already does, for a localhost-only tool. A Tauri shell can wrap the same SPA later with near-zero rework if a native window is ever wanted — that path stays open by keeping the UI a plain web app.

## Consequences

- The WebSocket event stream and REST surface must be specified as a versioned contract (see PLAN.md §5) since two codebases depend on it.
- Auth: localhost binding + a random bearer token written to a local file readable by the UI (defense against other local processes/CSRF from browsers); no remote exposure in v1.
- The agent must run headless (UI closed) with zero degradation; the UI is an observer/controller only.
- **Revisit triggers:** wanting remote (LAN/phone) management → add authenticated non-loopback binding; wanting a packaged native app → Tauri wrapper around the existing SPA.
