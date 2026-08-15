# TASK-009 — Dev tooling: pre-commit + CI

- **Status:** TODO
- **Priority:** P3
- **Source:** python-service-scaffold skill (ship-readiness checklist).

## Goal

Make quality gates automatic, not by memory.

## Acceptance criteria

- [ ] `.pre-commit-config.yaml`: ruff (lint+format), mypy, trailing-whitespace,
      end-of-file, check for committed `.env`/`dana.toml`.
- [ ] CI pipeline (GitHub Actions or equivalent): `uv sync`, ruff check, ruff
      format --check, mypy, pytest (offline, `-m 'not live'`), coverage report.
- [ ] CI runs on the pinned Python version; `uv.lock` respected.
- [ ] Badge/status wired into README once a remote exists.
