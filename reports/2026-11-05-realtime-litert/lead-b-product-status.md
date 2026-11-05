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
| QA1 unknown repo -> 403 instead of 404 | med | C1 | routed |
| QA2 PullJob bytes_done > bytes_total (companions) | low-med | C1 | routed |
| WIN1 rmtree ignore_errors on mmapped files; os.replace retry; safe_relpath reserved names/ADS; store._read PermissionError wipes registry | med | C1 | routed |
| WIN2 litert close must release mmap | med | C2 | FIXED (close()+gc, unload_under) |
| WIN3 .gitattributes eol=lf for .sh | low | Lead B | FIXED (.gitattributes added) |
| SEC1 click_consequence NFD Vietnamese / vocab bypass | med | C6 | routed |
| SEC2 MCP annotations trusted from remote servers; unfenced descriptions | med | C5 | routed |
| SEC3 probe sends stored api_key to overlay base_url | low-med | C5 | routed |
| SEC4 _REPO_RE fullmatch; revision unvalidated; Link:next host; https-only redirects | low | C1 | routed (quick ones) |

## Deferred
- pull.py max_bytes cap vs hostile repo (low; needs cfg field = lead's config.py, risky before 14:30).
- /models/file ?token= uses long-lived control token in URL (low; needs HMAC ticket, UI+API change).
- Vietnamese STT quality/latency (whisper-tiny int8 poor, 30 s fixed window ~1.2-2 s): no provider bug; documented in README, model upgrade is a product decision.
- C6: submit-button classification (snapshot lacks button type; would prompt on every Search).
