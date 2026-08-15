# TASK-006 — Phase 5: action layer (safe computer use)

- **Status:** TODO
- **Priority:** P2
- **Blocked by:** TASK-005 (brain emits the tool calls).
- **Source:** docs/PLAN.md §3 (Phase 5), §8, safe-computer-use skill.

## Goal

The `ActionProvider` that turns structured tool calls into real browser/OS
actions — safely. The LLM proposes; a trusted non-LLM layer validates and
disposes. Standard gate policy (destructive requires confirmation).

## Acceptance criteria

- [ ] Playwright browser tools: role/text locators, auto-wait (no sleeps), page
      objects, dedicated profile, storage-state auth, domain allowlist.
- [ ] Bounded OS actions: launch allowlisted apps, focus window, media keys,
      clipboard — off by default, APIs preferred over GUI.
- [ ] Reversibility classification → gate: read-only free; reversible logged;
      destructive requires confirmation (voice + UI).
- [ ] Allowlist / default-deny; every argument validated; rate/scope limits.
- [ ] Append-only JSONL audit log of every action, surfaced to UI.
- [ ] Secrets injected at execution from vault; never seen by model; redacted.
- [ ] Gate/validation layer unit-tested hardest; adversarial prompt-injection
      fixtures (page content is not an instruction).
