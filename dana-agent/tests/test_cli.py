"""CLI subcommand routing for `python -m dana_agent secrets|autostart`.

Handlers are driven with monkeypatched vault/subprocess so nothing touches the
real Credential Manager or Task Scheduler.
"""

from __future__ import annotations

import io
import sys
from typing import Any

import pytest

from dana_agent import __main__ as cli


def test_parse_run_defaults() -> None:
    args = cli._parse_args([])
    assert args.command is None
    assert args.control is False


def test_parse_secrets_subcommand() -> None:
    args = cli._parse_args(["secrets", "set", "ASSEMBLYAI_API_KEY"])
    assert args.command == "secrets"
    assert args.action == "set"
    assert args.name == "ASSEMBLYAI_API_KEY"


def test_parse_autostart_subcommand() -> None:
    args = cli._parse_args(["autostart", "install"])
    assert args.command == "autostart"
    assert args.action == "install"


def test_secrets_set_reads_stdin_not_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    stored: dict[str, str] = {}
    monkeypatch.setattr(cli, "_read_secret_value", lambda name: "piped-secret")
    import dana_agent.secrets as secrets_mod

    monkeypatch.setattr(secrets_mod, "set_secret", lambda n, v: stored.__setitem__(n, v))
    rc = cli.main(["secrets", "set", "GEMINI_API_KEY"])
    assert rc == 0
    assert stored == {"GEMINI_API_KEY": "piped-secret"}


def test_secrets_get_missing_returns_1(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana_agent.secrets as secrets_mod

    monkeypatch.setattr(secrets_mod, "get_secret", lambda n: None)
    assert cli.main(["secrets", "get", "NOPE"]) == 1


def test_autostart_status_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana_agent.platform_win import autostart

    monkeypatch.setattr(autostart, "is_autostart_installed", lambda: True)
    assert cli.main(["autostart", "status"]) == 0


def test_autostart_install_routes_to_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana_agent.platform_win import autostart

    calls: list[str] = []

    def fake_install() -> Any:
        calls.append("install")
        return autostart.CommandResult(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(autostart, "install_autostart", fake_install)
    assert cli.main(["autostart", "install"]) == 0
    assert calls == ["install"]


def test_read_secret_value_from_piped_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("piped-key\n"))
    # A StringIO's isatty() is False → reads the line rather than prompting.
    assert cli._read_secret_value("ANY") == "piped-key"
