"""Typed configuration — the single source of truth (also the UI contract).

One schema for the whole app (pydantic-settings), a sub-model per capability.
Precedence: environment > config file (dana.toml) > defaults. Secrets are
``SecretStr``, redacted on display. Validated on load; a bad provider name or a
missing key fails at startup, not at first request (docs/PLAN.md §6).

Env vars use the ``DANA_`` prefix and ``__`` to descend into nested models,
e.g. ``DANA_CONVERSATION__STT__PROVIDER=deepgram``, ``DANA_LANGUAGE=vi``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

Language = Literal["en", "vi"]


# --- per-capability provider configs -------------------------------------


class ProviderConfig(BaseModel):
    """Common shape: a provider name plus optional credentials/endpoint. A
    self-hosted or OpenAI-compatible backend is the same provider with a
    ``base_url`` override — not a separate class."""

    provider: str = "mock"
    api_key: SecretStr | None = None
    base_url: str | None = None


class VADConfig(ProviderConfig):
    """Voice activity detection — the cheap gate in front of the wake engine."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class WakeConfig(ProviderConfig):
    phrase: str = "Hey DaNa"
    end_phrase: str = "Bye DaNa"
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    inactivity_timeout_s: float = Field(default=30.0, gt=0.0)


class STTConfig(ProviderConfig):
    model: str = "mock-stt"


class LLMConfig(ProviderConfig):
    model: str = "mock-llm"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class TTSConfig(ProviderConfig):
    model: str = "mock-tts"
    voice: str = "default"


class S2SConfig(ProviderConfig):
    model: str = "mock-s2s"
    voice: str = "default"


# --- grouped configs -----------------------------------------------------


class ConversationConfig(BaseModel):
    """A conversation provider is either a cascade (STT+LLM+TTS) or a single
    speech-to-speech model. Default to cascade (docs/PLAN.md §6)."""

    mode: Literal["cascade", "s2s"] = "cascade"
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    s2s: S2SConfig = S2SConfig()


class ActionsConfig(ProviderConfig):
    """Action layer selection plus the allowlists and gate policy that bound
    it (safe-computer-use skill)."""

    allowed_domains: list[str] = Field(default_factory=list)
    allowed_apps: list[str] = Field(default_factory=list)
    gate_policy: Literal["standard", "conservative", "wide"] = "standard"
    max_actions_per_turn: int = Field(default=10, gt=0)


class MemoryConfig(BaseModel):
    """Brain conversation memory (the LangGraph checkpointer).

    - ``memory``  — in-process, non-persistent (LangGraph InMemorySaver).
    - ``sqlite``  — SQLite-backed; ``sqlite_path`` is a file for durable memory
      across restarts, or the literal ``":memory:"`` for a SQLite in-memory DB.
    """

    backend: Literal["memory", "sqlite"] = "memory"
    sqlite_path: str = ".dana/brain.sqlite"


class BrainConfig(BaseModel):
    """The agent brain (LangGraph-style state graph over the LLM provider)."""

    system_prompt: str = (
        "You are DaNa, a concise, helpful voice assistant. Answer briefly and "
        "naturally. Use a tool only when it is needed to act on the user's "
        "browser or computer."
    )
    max_steps: int = Field(default=8, gt=0, le=50)
    memory: MemoryConfig = MemoryConfig()

    # Long-conversation strategy so context doesn't grow unbounded:
    #   none      -> send the whole history every turn
    #   trim      -> send only the last `history_keep_last` messages (default)
    #   summarize -> once history exceeds `history_trigger_messages`, condense the
    #                older ones into a running summary and prune them
    history_strategy: Literal["none", "trim", "summarize"] = "trim"
    history_trigger_messages: int = Field(default=20, gt=0)
    history_keep_last: int = Field(default=8, gt=0)


class ControlConfig(BaseModel):
    """Local control API (agent <-> UI). Bound to loopback only in v1."""

    host: str = "127.0.0.1"
    port: int = Field(default=8760, gt=0, lt=65536)
    # Path to the local bearer-token file (not a secret value itself).
    token_path: str = ".dana/control-token"  # noqa: S105


class PrivacyConfig(BaseModel):
    local_only: bool = False
    audio_retention: Literal["none", "session", "persist"] = "none"


# --- root ----------------------------------------------------------------


class AppConfig(BaseSettings):
    """Root application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DANA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        toml_file="dana.toml",
        extra="forbid",
    )

    language: Language = "en"
    log_level: str = "INFO"
    vad: VADConfig = VADConfig()
    wake: WakeConfig = WakeConfig()
    brain: BrainConfig = BrainConfig()
    conversation: ConversationConfig = ConversationConfig()
    actions: ActionsConfig = ActionsConfig()
    control: ControlConfig = ControlConfig()
    privacy: PrivacyConfig = PrivacyConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest priority first: explicit init kwargs (tests) > env > .env >
        # dana.toml > file secrets/defaults. This is the env > file > defaults
        # precedence the brief requires.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> AppConfig:
    """The shared application settings — one cached instance the whole project
    imports (``from dana_agent.config import get_settings``), loaded from
    environment > .env / dana.toml > defaults. Cached so every caller sees the
    same config; call ``get_settings.cache_clear()`` in tests that need a reload.
    """
    return AppConfig()
