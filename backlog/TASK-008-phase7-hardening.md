# TASK-008 — Phase 7: hardening

- **Status:** TODO
- **Priority:** P3
- **Source:** docs/PLAN.md §3 (Phase 7), §8.

## Goal

Turn the working assistant into an always-on, installable, secure setup.

## Acceptance criteria

- [ ] Windows autostart (Task Scheduler) + tray icon (mic mute, open dashboard, quit).
- [ ] Secrets moved to Windows Credential Manager via `keyring`; `.env` for dev only.
- [ ] `local_only` mode: flips to local provider set + blocks non-allowlisted egress.
- [ ] One S2S provider wired end to end (proves cascade-or-S2S abstraction).
- [ ] Survives reboot; 24 h soak run with no leak/crash.
- [ ] README + runbooks; version pinning for Pipecat and vendor SDKs.
