"""Bearer-token auth for the loopback control plane.

Loopback binding alone is not enough: any local process (or a browser hitting
127.0.0.1 via a malicious page) could reach the API. So every request except
``/health`` must carry ``Authorization: Bearer <token>`` where the token is a
random secret written to a local file (``cfg.control.token_path``) that only the
user — and the dashboard they point at it — can read (ADR-0002).

The token is generated once and persisted; the UI reads the same file. The
comparison is constant-time.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
from pathlib import Path

from dana_agent.observability import get_logger

log = get_logger(__name__)

_TOKEN_BYTES = 32  # 256 bits of entropy, URL-safe base64


def load_or_create_token(token_path: str | Path) -> str:
    """Return the bearer token, generating and persisting one on first use.

    The file is created with owner-only permissions where the OS honours them
    (POSIX 0600; on Windows the bits are advisory but still set). A pre-existing
    file is trusted as-is so the UI and agent agree on the same value.
    """
    path = Path(token_path)
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token

    token = secrets.token_urlsafe(_TOKEN_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
    # Best-effort owner-only perms — a filesystem that rejects chmod (some
    # Windows setups) must not stop the agent from starting.
    with contextlib.suppress(OSError):
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    log.info("control_token_generated", path=str(path))
    return token


def token_matches(expected: str, header_value: str | None) -> bool:
    """Constant-time check of an ``Authorization`` header against the token."""
    if not header_value:
        return False
    scheme, _, presented = header_value.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        return False
    return secrets.compare_digest(presented.strip(), expected)
