"""Windows autostart via Task Scheduler (idempotent install/uninstall/status).

Task Scheduler is preferred over the Startup folder: it runs reliably at logon,
survives reboots, and is scriptable and idempotent. We register an ``ONLOGON``
task that launches the agent with least privilege (``/rl limited``).

Design rules (research §C.1):

- **list argv, never a shell string** — every ``schtasks`` call passes a Python
  list to :func:`subprocess.run` with ``shell=False`` (the default), so no
  model- or path-supplied string is ever interpreted by a shell.
- **idempotent create** — ``/f`` force-overwrites an existing definition, so
  re-running ``install`` just replaces it.
- **delete-of-missing is success** — ``uninstall`` treats "task not found" as an
  already-satisfied goal rather than an error.

The functions here perform REAL registration when called. They are invoked only
from the ``autostart`` CLI subcommand; the test suite monkeypatches
``subprocess.run`` so nothing is ever registered during tests.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

# The Task Scheduler task name. Stable — install/uninstall/status all key off it.
TASK_NAME = "DaNa Voice Assistant"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of a schtasks invocation, surfaced to the CLI for a clean message."""

    ok: bool
    returncode: int
    stdout: str
    stderr: str


def _agent_command() -> str:
    """The command Task Scheduler runs at logon: the current interpreter + the
    package entrypoint. The exe path is quoted because it may contain spaces —
    this inner quoting is what ``/tr`` expects (a single command string)."""
    return f'"{sys.executable}" -m dana_agent'


def _run(argv: list[str]) -> CommandResult:
    proc = subprocess.run(  # noqa: S603 - fixed argv, shell=False, no user string
        argv,
        capture_output=True,
        text=True,
        check=False,
    )
    return CommandResult(
        ok=proc.returncode == 0,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def build_install_argv(task_name: str = TASK_NAME) -> list[str]:
    """The exact ``schtasks`` argv used to (re)create the logon task. Exposed so
    ``autostart status``/dry-runs and tests can inspect it without executing."""
    return [
        "schtasks",
        "/create",
        "/tn",
        task_name,
        "/tr",
        _agent_command(),
        "/sc",
        "onlogon",
        "/rl",
        "limited",
        "/f",  # force-overwrite → idempotent
    ]


def install_autostart(task_name: str = TASK_NAME) -> CommandResult:
    """Create (or replace) the logon autostart task. Idempotent via ``/f``."""
    return _run(build_install_argv(task_name))


def uninstall_autostart(task_name: str = TASK_NAME) -> CommandResult:
    """Delete the autostart task. Delete-of-missing is reported as success."""
    result = _run(["schtasks", "/delete", "/tn", task_name, "/f"])
    if not result.ok and _looks_like_missing(result):
        return CommandResult(ok=True, returncode=0, stdout=result.stdout, stderr=result.stderr)
    return result


def is_autostart_installed(task_name: str = TASK_NAME) -> bool:
    """True if the task exists (``schtasks /query`` returns 0)."""
    return _run(["schtasks", "/query", "/tn", task_name]).ok


def _looks_like_missing(result: CommandResult) -> bool:
    """schtasks returns non-zero with 'cannot find the file' / 'does not exist'
    when the task is absent — treat that as an already-uninstalled success."""
    text = f"{result.stdout} {result.stderr}".lower()
    return "cannot find" in text or "does not exist" in text or "not exist" in text
