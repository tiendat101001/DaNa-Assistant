# Lead B — product status board

| Task | Owner | Status | Evidence |
|---|---|---|---|
| UI build | Lead B | DONE | `npm run build` ok (190 kB js) |
| E2E desktop: hub list, TTS filter (Kokoro, kitten), drawer, pull&load moonshine-tiny, local table, Use as ASR, presets | UX/C1 | PASS | screenshots/desk-*.png, real HF |
| E2E: Measure latency (Together) | C5 | PASS after fix (C5: probe now resolves env keys via AppConfig) | TTFT 1281-1413 ms via sandbox proxy (API-only call 497 ms) |
| E2E phone 360px (all flows) | UX | PASS | screenshots/phone-*.png, no h-scroll |
| Cold catalog empty grid | UX | FIXED | loading status line added |
| Computer use real check (BrowserActions + ActionGate) | C6 | PASS | fill ALLOW/REVERSIBLE; Search ALLOW; Delete account CONFIRM/DESTRUCTIVE; external link REFUSE/OUT_OF_SCOPE; Delete not executed |
| MCP real stdio check | C5 | PASS after fix | mcp 2.x snake_case hints bug fixed by C5: read-only ALLOW, destructive CONFIRM, unannotated CONFIRM, unknown REFUSE |
| Docs: .env.example + README section | Lead B | DONE | no secrets |

## Review & QA round
| Finding | Sev | Owner | Status |
|---|---|---|---|
| QA1: unknown repo -> 403 "gated" instead of 404 (anonymous HF 401 ambiguous) | medium | C1 | routed, fixing |

## Deferred
(none yet)
