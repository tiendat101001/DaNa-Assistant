"""A fake Controller so the control API can be tested with no pipeline/pipecat.

Records the calls the routes make and implements config apply with the real
``diff_and_label`` logic, so PUT /config round-trips honestly without a running
agent."""

from __future__ import annotations

from typing import Any

from dana_agent.config import AppConfig
from dana_agent.control.config_apply import (
    ChangeStatus,
    FieldChange,
    diff_and_label,
    get_path,
    set_path,
)
from dana_agent.control.controller import ConfirmationInfo, Controller, StatusInfo
from tests.support import clean_config


class FakeController(Controller):
    def __init__(self, cfg: AppConfig | None = None) -> None:
        # Deterministic defaults, independent of any dev .env.
        self._cfg = cfg if cfg is not None else clean_config()
        self._muted = False
        self._running = False
        self.calls: list[str] = []
        self._pending: dict[str, ConfirmationInfo] = {}
        self._audit: list[dict[str, Any]] = []

    # test seams -----------------------------------------------------------
    def add_pending(self, info: ConfirmationInfo) -> None:
        self._pending[info.id] = info

    def add_audit(self, entry: dict[str, Any]) -> None:
        self._audit.append(entry)

    # Controller -----------------------------------------------------------
    def config(self) -> AppConfig:
        return self._cfg

    def status(self) -> StatusInfo:
        return StatusInfo(
            state="idle",
            session_running=self._running,
            muted=self._muted,
            uptime_s=1.5,
            session_id="fake-session" if self._running else None,
            language=self._cfg.language,
            mode=self._cfg.conversation.mode,
            providers={"stt": self._cfg.conversation.stt.provider},
        )

    async def start_session(self) -> None:
        self.calls.append("start")
        self._running = True

    async def stop_session(self) -> None:
        self.calls.append("stop")
        self._running = False

    async def mute(self) -> None:
        self.calls.append("mute")
        self._muted = True

    async def unmute(self) -> None:
        self.calls.append("unmute")
        self._muted = False

    def is_muted(self) -> bool:
        return self._muted

    async def resolve_confirmation(self, confirmation_id: str, *, approved: bool) -> bool:
        self.calls.append(f"resolve:{confirmation_id}:{approved}")
        return self._pending.pop(confirmation_id, None) is not None

    def pending_confirmations(self) -> list[ConfirmationInfo]:
        return list(self._pending.values())

    def audit_entries(self, *, since: float | None = None) -> list[dict[str, Any]]:
        return [e for e in self._audit if since is None or float(e["at"]) >= since]

    def apply_config(self, new_config: AppConfig) -> list[FieldChange]:
        changes = diff_and_label(self._cfg, new_config)
        for change in changes:
            if change.status is ChangeStatus.APPLIED:
                set_path(self._cfg, change.path, get_path(new_config, change.path))
        return changes
