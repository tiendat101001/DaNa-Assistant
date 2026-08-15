---
name: python-service-scaffold
description: Best-practice scaffold and conventions for a production-grade async Python service or application — project layout, uv dependency management, pyproject config, ruff + mypy + pytest, structured logging, typed settings, async patterns, and testing with mock providers/fixtures. Use this skill whenever starting a new Python project, package, backend service, agent, or CLI from scratch; whenever the user asks how to structure a Python codebase, set up tooling, or make code clean/testable/maintainable; and whenever you are about to create the initial files of a Python repo. Consult it before scaffolding so the project starts on solid, idiomatic, testable footing instead of a loose pile of scripts.
---

# Python Service Scaffold

A project that starts clean stays clean. This skill is the default shape for a new async Python service/app so it is idiomatic, typed, testable offline, and pleasant to extend. Adapt, don't cargo-cult — but deviate deliberately.

## Layout: src layout, one package

```
myproject/
├── pyproject.toml
├── README.md
├── .env.example              # documents every env var; never commit real .env
├── .gitignore
├── src/
│   └── myproject/
│       ├── __init__.py
│       ├── config.py         # typed settings (pydantic-settings)
│       ├── logging.py        # structured logging setup
│       ├── app.py            # composition root: wire everything here
│       ├── <domain modules>/ # one folder per bounded concern
│       └── providers/        # swappable backends (see provider-plugin-architecture)
└── tests/
    ├── conftest.py
    ├── fakes/                # mock/fake providers used across tests
    └── ...
```

- **Use the `src/` layout.** It forces you to test the *installed* package, catching import/packaging bugs that a flat layout hides.
- **A composition root** (`app.py`) is the one place that reads config and constructs concrete objects; everything else receives its dependencies. This keeps modules unit-testable and swap-friendly.
- Keep modules organized by **concern/domain**, not by type (avoid a giant `utils.py` and a `models.py` that everything imports).

## Dependency management: uv

Use **uv** — it is the current fast standard for Python env + dependency management.

```bash
uv init                     # or set up pyproject by hand
uv add pydantic-settings structlog httpx
uv add --dev ruff mypy pytest pytest-asyncio pytest-cov
uv run pytest               # runs in the managed env
uv sync                     # reproduce the locked env
```

- Commit `uv.lock` for apps (reproducible installs); libraries stay lock-free.
- Pin a Python version in `pyproject.toml` (`requires-python`).

## pyproject.toml (single config surface)

Configure the whole toolchain in one file:

```toml
[project]
name = "myproject"
requires-python = ">=3.11"
dependencies = ["pydantic-settings", "structlog", "httpx"]

[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "ASYNC", "S"]  # incl. import-sort, async, security

[tool.mypy]
strict = true
warn_unused_ignores = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = "-q --cov=myproject"
```

- **ruff** for lint + format + import sorting (replaces black/isort/flake8) — one fast tool.
- **mypy --strict** (or pyright). Type everything; types are the cheapest documentation and catch whole bug classes.
- Wire these into **pre-commit** and **CI** so they run automatically, not by memory.

## Typed configuration

One settings object, validated on load, env-overridable. See the `provider-plugin-architecture` skill for the multi-provider version.

```python
from pydantic import SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    log_level: str = "INFO"
    api_key: SecretStr | None = None
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
```

- Precedence: **env > .env file > defaults.**
- Secrets are `SecretStr`; never log or serialize them. Keep `.env.example` current as the canonical list of variables.
- Fail fast: a bad config raises at startup, not at first use.

## Structured logging

Use **structlog** (or stdlib `logging` with a JSON formatter). Log key/value events, not f-string prose.

```python
import structlog
log = structlog.get_logger()
log.info("turn_completed", latency_ms=412, provider="openai", tokens=284)
```

- JSON logs in production (searchable/aggregatable), pretty console logs in dev.
- Attach request/session IDs via contextvars so events correlate across a turn.
- Never log secrets or full user audio/PII; redact.

## Async patterns

- Pick async **or** sync deliberately and stay consistent within a layer; don't block the event loop with sync I/O inside async code.
- Use a single async HTTP client (`httpx.AsyncClient`) with connection pooling and explicit timeouts on **every** external call — no unbounded awaits.
- Bound concurrency with a semaphore; wrap external calls in retry-with-backoff (put this in the provider base, not scattered).
- Manage lifecycle with async context managers / lifespan hooks so clients and tasks are opened and closed cleanly; ensure graceful shutdown cancels background tasks.
- Never fire-and-forget a task without keeping a reference and handling its exceptions.

## Testing: offline by default

- **pytest + pytest-asyncio.** The suite must run with **no network and no secrets** — this is the whole point of the provider abstraction.
- Put **fakes/mocks** in `tests/fakes/` and inject them via fixtures in `conftest.py`. Fakes are the default backend in tests.
- Test behavior at the interface (the ABC), parametrized across providers where relevant.
- Use `tmp_path`, `monkeypatch`, and recorded fixtures instead of touching the real filesystem/clock/network.
- Track coverage but chase *meaningful* tests (error paths, edge cases), not a percentage.

## Errors

- Define a small hierarchy of domain exceptions (`ConfigError`, `ProviderError`, `ActionRefused`) so callers can handle intent, not string-match messages.
- Fail fast and loud on programmer errors; degrade gracefully on expected external failures (a provider timing out shouldn't crash the app).

## Ship-readiness checklist

- [ ] `src/` layout; a single composition root wires concrete objects.
- [ ] uv for deps; `uv.lock` committed for the app; Python version pinned.
- [ ] ruff + mypy(strict) + pytest configured in `pyproject.toml`, run in pre-commit + CI.
- [ ] Typed settings; env > file > defaults; secrets as `SecretStr`; `.env.example` current.
- [ ] Structured logging with correlation IDs; no secrets/PII in logs.
- [ ] Async: explicit timeouts on all external calls; bounded concurrency; clean shutdown.
- [ ] Test suite runs offline with fakes; error paths covered.
- [ ] Domain exception hierarchy; fail-fast on bugs, graceful on external failures.
