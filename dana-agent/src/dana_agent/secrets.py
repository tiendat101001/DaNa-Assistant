"""Secret resolution — Windows Credential Manager (``keyring``) with env fallback.

A tiny, import-guarded resolver so real API keys can live in the OS credential
vault (Windows Credential Manager via ``keyring``) instead of a plaintext
``.env``. It degrades cleanly:

    keyring (if installed AND a backend is available)  →  os.environ

``keyring`` is an optional dependency (the ``control`` extra). When it is not
installed, or its backend is unavailable (headless CI, WSL without the Windows
session), :func:`resolve_secret` silently falls through to the process
environment — so the offline test suite and the headless agent work unchanged
with nothing installed.

Precedence for API keys stays exactly as documented in :mod:`dana_agent.config`:
an explicit ``DANA_*__API_KEY`` (parsed by pydantic-settings) wins, then this
resolver is consulted for the well-known vendor names, and within the resolver
the vault beats a plain environment variable. ``.env`` remains a dev-only
fallback (it is loaded into ``os.environ`` before the resolver ever runs).

The tiny CLI (``python -m dana_agent secrets set|get|delete NAME``) writes to the
vault without echoing the key into shell history — the value is read from an
interactive prompt or piped stdin, never taken as an argv argument.
"""

from __future__ import annotations

import os

# The keyring "service" name every DaNa secret is stored under. Matches the
# vendor variable name as the "username", e.g.
#   service="dana-agent", username="ASSEMBLYAI_API_KEY".
DEFAULT_SERVICE = "dana-agent"


def _keyring_get(name: str, service: str) -> str | None:
    """Best-effort read from the OS vault. Returns None on ANY failure (keyring
    absent, no backend, locked vault) so callers fall through to the env."""
    try:
        import keyring  # noqa: PLC0415
    except ImportError:
        return None
    try:
        value = keyring.get_password(service, name)
    except Exception:  # noqa: BLE001 - a failing/locked backend must fall back, not crash
        return None
    return value or None


def resolve_secret(name: str, *, service: str = DEFAULT_SERVICE) -> str | None:
    """Resolve a secret by its well-known name.

    Tries the OS credential vault first (via ``keyring``), then the process
    environment (which already includes anything loaded from ``.env``). Returns
    ``None`` when neither source has it.
    """
    value = _keyring_get(name, service)
    if value:
        return value
    return os.environ.get(name)


# --- CLI-facing vault mutators (only reached from the `secrets` subcommand) ---


class KeyringUnavailableError(RuntimeError):
    """Raised by the mutating helpers when ``keyring`` cannot be used, so the CLI
    can print an actionable message instead of a stack trace."""


def _require_keyring() -> object:
    try:
        import keyring  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise KeyringUnavailableError(
            "keyring is not installed. Install the control extra: "
            "uv sync --extra control (or: pip install keyring)"
        ) from exc
    return keyring


def set_secret(name: str, value: str, *, service: str = DEFAULT_SERVICE) -> None:
    """Store ``value`` under ``name`` in the OS credential vault."""
    keyring = _require_keyring()
    keyring.set_password(service, name, value)  # type: ignore[attr-defined]


def get_secret(name: str, *, service: str = DEFAULT_SERVICE) -> str | None:
    """Read ``name`` from the OS credential vault (vault only — no env fallback)."""
    keyring = _require_keyring()
    return keyring.get_password(service, name)  # type: ignore[attr-defined, no-any-return]


def delete_secret(name: str, *, service: str = DEFAULT_SERVICE) -> bool:
    """Delete ``name`` from the vault. Returns False if it was not present
    (delete-of-missing is treated as a success by the CLI)."""
    keyring = _require_keyring()
    try:
        keyring.delete_password(service, name)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - keyring raises PasswordDeleteError when absent
        return False
    return True
