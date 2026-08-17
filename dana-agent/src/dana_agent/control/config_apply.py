"""Hot-apply policy for ``PUT /config``.

A config write is validated as a whole (a full ``AppConfig`` round-trip), then
each *changed* leaf is labelled:

- ``applied``          — the change takes *genuine live effect* on the running
                         process. Only fields the runtime re-reads each cycle
                         qualify: the log level (root logger), the gate policy
                         (``ActionGate.decide`` re-reads it per action), and the
                         brain step budget (``Brain._stream`` re-reads it per
                         turn). The controller both writes the value into the
                         running config *and* pushes it into the live object.
- ``restart_required`` — the value is baked into an already-constructed provider,
                         socket, audio device, compiled LangGraph, or the wake
                         gate at build time, so changing the config field cannot
                         affect the running stack (which is built once and reused
                         across sessions). Reported, not applied.

The split is deliberately *honest*: a field is only ``applied`` when a live
re-bind is actually implemented, so the UI never over-promises. Everything else
— provider selection, ports, sample rates, language, allowlists, the system
prompt, wake sensitivity/phrase, history strategy — is ``restart_required`` until
a live re-bind hook exists for it. Secrets are compared and reported already
redacted (the diff runs over ``model_dump(mode="json")``, where ``SecretStr``
renders masked).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from dana_agent.config import AppConfig

# Dotted leaf paths that take GENUINE live effect on a running agent — each has a
# real re-bind hook in AgentController._hot_side_effect. Everything else is
# restart_required (honest labeling: we never mark a field ``applied`` unless the
# running stack actually observes the change without a restart). Grow this set
# only in lock-step with a new live re-bind hook.
HOT_APPLY_PATHS: frozenset[str] = frozenset(
    {
        "log_level",  # root logger .setLevel()
        "actions.gate_policy",  # ActionGate.set_policy() — re-read per action
        "brain.max_steps",  # Brain.set_max_steps() — re-read per turn
    }
)


class ChangeStatus(StrEnum):
    APPLIED = "applied"
    RESTART_REQUIRED = "restart_required"


@dataclass(frozen=True, slots=True)
class FieldChange:
    path: str
    old: Any
    new: Any
    status: ChangeStatus

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "old": self.old,
            "new": self.new,
            "status": self.status.value,
        }


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dotted leaf paths. Lists are leaves (a whole
    allowlist is one setting, not N)."""
    out: dict[str, Any] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, path))
    else:
        out[prefix] = obj
    return out


def diff_and_label(current: AppConfig, new: AppConfig) -> list[FieldChange]:
    """Return one ``FieldChange`` per changed leaf between ``current`` and
    ``new``, each labelled applied|restart_required (secrets pre-redacted)."""
    cur_flat = _flatten(current.model_dump(mode="json"))
    new_flat = _flatten(new.model_dump(mode="json"))
    changes: list[FieldChange] = []
    for path in sorted(cur_flat.keys() | new_flat.keys()):
        old = cur_flat.get(path)
        new_val = new_flat.get(path)
        if old == new_val:
            continue
        status = (
            ChangeStatus.APPLIED if path in HOT_APPLY_PATHS else ChangeStatus.RESTART_REQUIRED
        )
        changes.append(FieldChange(path=path, old=old, new=new_val, status=status))
    return changes


def get_path(cfg: AppConfig, path: str) -> Any:
    """Read a live (unredacted) value from the config by dotted path."""
    obj: Any = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_path(cfg: AppConfig, path: str, value: Any) -> None:
    """Write a value into a nested pydantic config by dotted path."""
    obj: Any = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
