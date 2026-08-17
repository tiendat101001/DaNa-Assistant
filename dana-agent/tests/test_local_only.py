"""privacy.local_only enforcement + the control-API LAN-exposure guard.

local_only is a construction-time policy (NOT an OS firewall): it refuses to
*build* any conversation provider that could egress off-box — asserting both that
each selected provider's base_url is loopback/private AND that it advertises
Capabilities.self_hosted.
"""

from __future__ import annotations

import pytest

from dana_agent.capabilities import check_control_exposure, enforce_local_only
from dana_agent.errors import ConfigError
from tests.support import clean_config


def test_local_only_off_is_noop() -> None:
    # A cloud provider is fine when local_only is not set.
    cfg = clean_config(conversation={"stt": {"provider": "assemblyai"}})
    enforce_local_only(cfg)  # no raise


def test_default_mock_stack_passes_local_only() -> None:
    cfg = clean_config(privacy={"local_only": True})
    enforce_local_only(cfg)  # all-mock is self-hosted → ok


def test_s2s_mock_stack_passes_local_only() -> None:
    cfg = clean_config(privacy={"local_only": True}, conversation={"mode": "s2s"})
    enforce_local_only(cfg)


def test_cloud_stt_rejected() -> None:
    cfg = clean_config(
        privacy={"local_only": True}, conversation={"stt": {"provider": "assemblyai"}}
    )
    with pytest.raises(ConfigError, match="self-hosted"):
        enforce_local_only(cfg)


def test_cloud_brain_llm_rejected() -> None:
    cfg = clean_config(
        privacy={"local_only": True}, conversation={"llm": {"provider": "openai"}}
    )
    with pytest.raises(ConfigError, match="self-hosted"):
        enforce_local_only(cfg)


def test_public_base_url_rejected() -> None:
    cfg = clean_config(
        privacy={"local_only": True},
        vad={"provider": "mock", "base_url": "https://vad.example.com"},
    )
    with pytest.raises(ConfigError, match="non-local base_url"):
        enforce_local_only(cfg)


def test_loopback_base_url_ok() -> None:
    cfg = clean_config(
        privacy={"local_only": True},
        vad={"provider": "mock", "base_url": "http://127.0.0.1:9000"},
    )
    enforce_local_only(cfg)


def test_private_base_url_ok() -> None:
    cfg = clean_config(
        privacy={"local_only": True},
        tts={"provider": "mock"},
        vad={"provider": "mock", "base_url": "http://192.168.1.20:5000"},
    )
    enforce_local_only(cfg)


# --- control exposure guard -------------------------------------------------


def test_control_loopback_is_silent() -> None:
    cfg = clean_config(control={"enabled": True, "host": "127.0.0.1"})
    check_control_exposure(cfg)  # no raise, no warning needed


def test_control_disabled_is_silent() -> None:
    cfg = clean_config(control={"enabled": False, "host": "0.0.0.0"})
    check_control_exposure(cfg)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "::"])
def test_control_lan_under_local_only_is_error(host: str) -> None:
    cfg = clean_config(privacy={"local_only": True}, control={"enabled": True, "host": host})
    with pytest.raises(ConfigError, match="loopback"):
        check_control_exposure(cfg)


def test_control_lan_without_local_only_warns_not_raises() -> None:
    # Loud warning path (no local_only): must NOT raise — the agent still starts.
    cfg = clean_config(control={"enabled": True, "host": "0.0.0.0"})
    check_control_exposure(cfg)  # no raise
