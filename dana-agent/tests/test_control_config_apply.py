"""Hot-apply labeling: which config changes are applied vs restart_required."""

from __future__ import annotations

from dana_agent.control.config_apply import (
    ChangeStatus,
    FieldChange,
    diff_and_label,
    get_path,
    set_path,
)
from tests.support import clean_config as _cfg


def _by_path(changes: list[FieldChange], path: str) -> FieldChange:
    return next(c for c in changes if c.path == path)


def test_no_changes_for_identical_config() -> None:
    cfg = _cfg()
    assert diff_and_label(cfg, cfg) == []


def test_hot_field_labeled_applied() -> None:
    change = _by_path(diff_and_label(_cfg(), _cfg(log_level="DEBUG")), "log_level")
    assert change.status is ChangeStatus.APPLIED
    assert change.old == "INFO"
    assert change.new == "DEBUG"


def test_provider_change_labeled_restart_required() -> None:
    new = _cfg(conversation={"stt": {"provider": "deepgram"}})
    change = _by_path(diff_and_label(_cfg(), new), "conversation.stt.provider")
    assert change.status is ChangeStatus.RESTART_REQUIRED


def test_allowlist_is_a_single_leaf() -> None:
    new = _cfg(actions={"allowed_domains": ["a.com", "b.com"]})
    change = _by_path(diff_and_label(_cfg(), new), "actions.allowed_domains")
    # The whole allowlist is one leaf (not N). It is restart_required: the value
    # is baked into the browser provider at build time, so labelling it
    # ``applied`` would over-promise (there is no live re-bind hook for it).
    assert change.status is ChangeStatus.RESTART_REQUIRED
    assert change.new == ["a.com", "b.com"]


def test_gate_policy_is_hot_applied() -> None:
    # gate_policy has a real live re-bind (ActionGate.set_policy), so it is applied.
    change = _by_path(
        diff_and_label(_cfg(), _cfg(actions={"gate_policy": "conservative"})),
        "actions.gate_policy",
    )
    assert change.status is ChangeStatus.APPLIED


def test_system_prompt_is_restart_required() -> None:
    # Baked into the compiled LangGraph at build → cannot take live effect.
    change = _by_path(
        diff_and_label(_cfg(), _cfg(brain={"system_prompt": "new"})),
        "brain.system_prompt",
    )
    assert change.status is ChangeStatus.RESTART_REQUIRED


def test_secret_change_is_redacted_in_diff() -> None:
    current = _cfg()  # stt.api_key is None
    new = _cfg(conversation={"stt": {"api_key": "supersecret"}})
    change = _by_path(diff_and_label(current, new), "conversation.stt.api_key")
    assert change.old is None
    # Never leak the raw secret in the labelled diff — it is masked.
    assert change.new == "**********"
    assert "supersecret" not in str(change.to_json())


def test_get_and_set_path_roundtrip() -> None:
    cfg = _cfg()
    assert get_path(cfg, "brain.system_prompt") == cfg.brain.system_prompt
    set_path(cfg, "brain.system_prompt", "hi")
    assert cfg.brain.system_prompt == "hi"
