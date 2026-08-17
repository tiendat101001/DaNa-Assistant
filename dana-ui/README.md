# dana-ui — DaNa management dashboard

React + Vite + TypeScript single-page app for the DaNa control API
(docs/PLAN.md §5, ADR-0002). Dependency-light: only `react` / `react-dom` at
runtime, plain CSS, no UI kit. The config form is rendered **from the JSON
Schema** the agent serves at `GET /config/schema`, so adding a provider or field
never needs a UI change.

## Views

- **Config editor** — schema-driven form from `/config/schema` + `/config`, with
  capability-aware provider dropdowns from `/providers`; `Save` shows each change
  labelled `applied` or `restart_required`.
- **Session** — Start / Stop + hard **mic mute** toggle.
- **State machine** — current FSM node + recent transitions (WS `state_transition`).
- **Transcript** — live partial/final transcripts + assistant turns (WS).
- **Latency waterfall** — per-turn per-stage bars (WS `latency_span`).
- **Audit trail** — gate decisions/executions (`/audit` + WS `audit_entry`).
- **Pending confirmations** — destructive-action prompts (WS `confirmation_request`
  → `POST /confirmations/{id}`).

## Auth

The agent writes a random bearer token to `dana-agent/.dana/control-token` on
first run. Paste it into the token screen; it is stored in `localStorage` and
sent as `Authorization: Bearer <token>` on every REST call and as `?token=` on
the WebSocket (browsers can't set WS headers). The agent binds to `127.0.0.1`
only.

## Develop

```bash
npm install
npm run dev        # Vite dev server; proxies /api (REST + WS) to 127.0.0.1:8760
```

Run the agent with the control plane enabled in another terminal:

```bash
cd ../dana-agent
uv run python -m dana_agent --control    # or DANA_CONTROL__ENABLED=true
```

## Build (production)

```bash
npm run build      # tsc -b && vite build → dist/
```

The agent serves `dist/` at `http://127.0.0.1:8760/` when
`control.ui_dist_dir` (default `../dana-ui/dist`) exists.

## Type-check only

```bash
npm run typecheck
```
