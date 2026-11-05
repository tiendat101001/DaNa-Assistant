# Lead B — product status board

| Task | Owner | Status | Evidence |
|---|---|---|---|
| UI build | Lead B | DONE | `npm run build` ok (190 kB js) |
| E2E desktop: hub list, TTS filter (Kokoro, kitten), drawer, pull&load moonshine-tiny, local table, Use as ASR, presets | UX/C1 | PASS | screenshots/desk-*.png, real HF |
| E2E: Measure latency (Together) | C5 | FIXING: probe lacked env api key when provider switched | "Missing credentials" |
| E2E phone 360px | UX | PENDING | |
| Computer use real check (BrowserActions + ActionGate) | C6 | PASS | fill ALLOW/REVERSIBLE; Search ALLOW; Delete account CONFIRM/DESTRUCTIVE; external link REFUSE/OUT_OF_SCOPE; Delete not executed |
| MCP real stdio check | C5 | PASS after fix | mcp 2.x snake_case hints bug fixed by C5: read-only ALLOW, destructive CONFIRM, unannotated CONFIRM, unknown REFUSE |
| Docs: .env.example + README section | Lead B | DONE | no secrets |
