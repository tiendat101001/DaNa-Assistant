"""Startup validation: check the assembled config against declared provider
capabilities and fail fast with an actionable error, instead of crashing
mid-conversation (provider-plugin-architecture skill).
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import dana_agent.providers  # noqa: F401  ensure all providers are registered
from dana_agent.brain.providers import BRAIN_PROVIDERS
from dana_agent.config import AppConfig, ProviderConfig
from dana_agent.errors import CapabilityError, ConfigError
from dana_agent.observability import get_logger
from dana_agent.providers.base import Capabilities
from dana_agent.providers.registry import (
    build_actions,
    build_llm,
    build_s2s,
    build_stt,
    build_tts,
    build_vad,
    build_wake,
)

log = get_logger(__name__)

# Hostnames/addresses that are unambiguously on-box (never leave the machine).
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", ""})


def _require_language(caps: Capabilities, label: str, language: str) -> None:
    if language not in caps.languages:
        raise CapabilityError(
            f"{label} does not support configured language {language!r}; "
            f"it supports {sorted(caps.languages)}"
        )


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise CapabilityError(message)


def validate_app_config(cfg: AppConfig) -> None:
    """Validate that the selected provider stack can actually serve the config.

    Raises ConfigError/CapabilityError on any unknown provider, unsupported
    language, or missing required capability. Constructing a provider here is
    cheap and side-effect-free (no network/model load at __init__).
    """
    lang = cfg.language

    # Wake gate must exist; language-agnostic phrase spotting, so no lang check.
    build_vad(cfg.vad)
    build_wake(cfg.wake)

    conv = cfg.conversation
    if conv.mode == "cascade":
        stt = build_stt(conv.stt)
        tts = build_tts(conv.tts)
        _require_language(stt.capabilities, f"STT provider {conv.stt.provider!r}", lang)
        _require_language(tts.capabilities, f"TTS provider {conv.tts.provider!r}", lang)
        # The brain's LLM is built by init_chat_model, so validate its provider
        # against the brain's supported set rather than the LLMProvider registry.
        if conv.llm.provider not in BRAIN_PROVIDERS:
            known = ", ".join(sorted(BRAIN_PROVIDERS))
            raise ConfigError(f"unknown brain llm provider {conv.llm.provider!r}; known: {known}")
    else:  # s2s
        s2s = build_s2s(conv.s2s)
        _require_language(s2s.capabilities, f"S2S provider {conv.s2s.provider!r}", lang)
        _require(
            s2s.capabilities.streaming,
            f"S2S provider {conv.s2s.provider!r} must stream",
        )

    # Action provider must exist (allowlists/gate policy validated by the schema).
    build_actions(cfg.actions)

    # Privacy: refuse to be *configured* to egress off-box under local_only, and
    # warn (or, under local_only, refuse) if the control API would be LAN-exposed.
    enforce_local_only(cfg)
    check_control_exposure(cfg)


# --- local_only enforcement ---------------------------------------------------


def _host_is_local(host: str) -> bool:
    """True if ``host`` is loopback or an RFC-1918 / private address.

    A hostname that is neither loopback nor a literal private IP (e.g. a public
    domain) is treated as NON-local — we do not resolve DNS at config time, so an
    unresolved public hostname is conservatively rejected under local_only.
    """
    host = host.lower()
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # a public hostname (not a literal IP) → not provably local
    return addr.is_loopback or addr.is_private


def _host_is_loopback(host: str) -> bool:
    """Stricter than :func:`_host_is_local`: ONLY true loopback counts.

    Used for the control API *bind* host, where "local" means the socket is
    unreachable from the network. A private/RFC-1918 bind (or ``0.0.0.0`` /
    ``::``, which mean "all interfaces") still exposes the port to the LAN, so
    those are NOT loopback-safe even though they pass ``_host_is_local``.
    """
    host = host.lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _base_url_is_local(base_url: str | None) -> bool:
    """A provider with no ``base_url`` uses the vendor SDK's baked-in endpoint, so
    "no endpoint" does NOT imply local. Locality of the endpoint is decided by the
    host; the self_hosted capability check (below) is what rejects a cloud vendor
    that happens to leave ``base_url`` unset."""
    if not base_url:
        return True  # host locality is vacuously fine; self_hosted gates the rest
    return _host_is_local(urlparse(base_url).hostname or "")


def _llm_capabilities(cfg: AppConfig) -> Capabilities | None:
    """Capabilities for the cascade LLM, or None when the provider is a real
    brain backend (openai/gemini/anthropic) that is not a self-hosted
    :class:`LLMProvider` in the registry — which, under local_only, means it is
    treated as a cloud provider and rejected."""
    try:
        return build_llm(cfg.conversation.llm).capabilities
    except ConfigError:
        return None


def _require_local_provider(label: str, sub: ProviderConfig, caps: Capabilities | None) -> None:
    if not _base_url_is_local(sub.base_url):
        raise ConfigError(
            f"local_only: {label} provider {sub.provider!r} has a non-local "
            f"base_url {sub.base_url!r} — it would send data off this machine. "
            "Point it at a loopback/private endpoint or switch to a local provider."
        )
    if caps is None or not caps.self_hosted:
        raise ConfigError(
            f"local_only: {label} provider {sub.provider!r} is not a self-hosted "
            "backend (it does not advertise Capabilities.self_hosted). Select a "
            "local/self-hosted provider (e.g. a mock, silero, faster-whisper, piper)."
        )


def enforce_local_only(cfg: AppConfig) -> None:
    """When ``privacy.local_only`` is set, refuse to *build* any conversation
    provider that could send audio/text off the machine.

    For every selected provider we assert BOTH: its ``base_url`` (if any) points
    at a loopback/private address, AND it advertises ``Capabilities.self_hosted``.
    A cloud provider fails the second check even with no ``base_url``; a
    self-hosted provider misconfigured with a public ``base_url`` fails the first.

    This is a *construction-time policy*, not an OS firewall — see the README.
    It cannot stop a dependency that ignores the policy; it guarantees DaNa is
    not *configured* to egress and refuses to start a cloud provider.
    """
    if not cfg.privacy.local_only:
        return

    conv = cfg.conversation
    checks: list[tuple[str, ProviderConfig, Capabilities | None]] = [
        ("vad", cfg.vad, build_vad(cfg.vad).capabilities),
        ("wake", cfg.wake, build_wake(cfg.wake).capabilities),
    ]
    if conv.mode == "cascade":
        checks.append(("stt", conv.stt, build_stt(conv.stt).capabilities))
        checks.append(("tts", conv.tts, build_tts(conv.tts).capabilities))
        checks.append(("llm", conv.llm, _llm_capabilities(cfg)))
    else:
        checks.append(("s2s", conv.s2s, build_s2s(conv.s2s).capabilities))

    for label, sub, caps in checks:
        _require_local_provider(label, sub, caps)

    log.info("local_only_enforced", mode=conv.mode)


def check_control_exposure(cfg: AppConfig) -> None:
    """Carry-over guard: the control API only defends itself with a bearer token.

    If it is enabled on a non-loopback host it would be reachable from the LAN, so
    we make that loud: a startup WARNING normally, or a hard ``ConfigError`` under
    ``local_only`` (where any off-box surface is disallowed by policy)."""
    if not cfg.control.enabled:
        return
    if _host_is_loopback(cfg.control.host):
        return
    if cfg.privacy.local_only:
        raise ConfigError(
            f"local_only: control API host {cfg.control.host!r} is not loopback — "
            "it would expose the control plane to the LAN. Set control.host to 127.0.0.1."
        )
    log.warning(
        "control_api_lan_exposed",
        host=cfg.control.host,
        port=cfg.control.port,
        hint="The control API is bound to a non-loopback address and is protected "
        "ONLY by a bearer token. Anyone on the LAN who obtains the token gets full "
        "control. Bind to 127.0.0.1 unless you intend remote access.",
    )
