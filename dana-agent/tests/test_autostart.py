"""Windows autostart helper — schtasks argv shape + idempotency semantics.

Nothing here registers a real task: ``subprocess.run`` is monkeypatched so the
helper's argv and return-code handling are verified without touching Task
Scheduler.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from dana_agent.platform_win import autostart


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def record_run(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture argv passed to subprocess.run; default to a success proc."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kw: Any) -> _FakeProc:
        calls.append(argv)
        return _FakeProc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_install_argv_shape() -> None:
    argv = autostart.build_install_argv()
    assert argv[:4] == ["schtasks", "/create", "/tn", autostart.TASK_NAME]
    assert "/sc" in argv and argv[argv.index("/sc") + 1] == "onlogon"
    assert "/rl" in argv and argv[argv.index("/rl") + 1] == "limited"
    assert argv[-1] == "/f"  # force → idempotent
    tr = argv[argv.index("/tr") + 1]
    assert "-m dana_agent" in tr and tr.startswith('"')  # exe path quoted


def test_install_runs_schtasks_create(record_run: list[list[str]]) -> None:
    result = autostart.install_autostart()
    assert result.ok
    assert record_run[0][:2] == ["schtasks", "/create"]


def test_status_true_on_zero_exit(record_run: list[list[str]]) -> None:
    assert autostart.is_autostart_installed() is True
    assert record_run[0][:2] == ["schtasks", "/query"]


def test_status_false_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeProc(returncode=1))
    assert autostart.is_autostart_installed() is False


def test_uninstall_ok(record_run: list[list[str]]) -> None:
    result = autostart.uninstall_autostart()
    assert result.ok
    assert record_run[0][:2] == ["schtasks", "/delete"]


def test_uninstall_of_missing_is_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **_kw: Any) -> _FakeProc:
        return _FakeProc(returncode=1, stderr="ERROR: The system cannot find the file specified.")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = autostart.uninstall_autostart()
    assert result.ok  # delete-of-missing treated as success


def test_uninstall_real_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **_kw: Any) -> _FakeProc:
        return _FakeProc(returncode=1, stderr="ERROR: Access is denied.")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = autostart.uninstall_autostart()
    assert not result.ok
