"""Typed configuration — the single source of truth (also the UI contract).

One schema for the whole app (pydantic-settings), a sub-model per capability.
Precedence: environment > config file (dana.toml) > defaults. Secrets are
``SecretStr``, redacted on display. Validated on load; a bad provider name or a
missing key fails at startup, not at first request (docs/PLAN.md §6).

API keys use well-known names (``ASSEMBLYAI_API_KEY``, ``GEMINI_API_KEY``, …) —
the same names the vendor SDKs expect, so one variable does double duty.
App settings use ``DANA_`` prefix with ``__`` for nesting, e.g.
``DANA_CONVERSATION__STT__PROVIDER=assemblyai``.  The explicit
``DANA_*__API_KEY`` vars still work and take priority over the standard names.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
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
    language_code: str = "en"  # propagated from AppConfig.language by _apply_standard_api_keys


class LLMConfig(ProviderConfig):
    model: str = "mock-llm"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class TTSConfig(ProviderConfig):
    model: str = "mock-tts"
    voice: str = "default"
    language_code: str = "en"  # propagated from AppConfig.language
    # How many sentence boundaries to accumulate before sending one TTS request.
    #   0 → whole reply in one request (fewest API calls, slightly higher latency)
    #   1 → one request per sentence (current default, lowest latency, most calls)
    #   N → one request per N sentences (balance)
    chunk_sentences: int = Field(default=1, ge=0)


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
        # "ignore" (not "forbid") because pydantic-settings v2.3+ also passes
        # standard vendor env vars (ASSEMBLYAI_API_KEY, GEMINI_API_KEY, …) through
        # to pydantic as extra fields.  Those are intentional and handled by the
        # _apply_standard_api_keys model_validator below; forbidding them here
        # would prevent the app from starting when those vars are set.
        extra="ignore",
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

    @model_validator(mode="after")
    def _apply_standard_api_keys(self) -> AppConfig:
        """Fill api_key fields from well-known vendor env var names.

        Standard names (``ASSEMBLYAI_API_KEY``, ``GEMINI_API_KEY``, …) are read
        only when the explicit ``DANA_*__API_KEY`` form is absent, so the DANA_
        vars always take priority.  This means a single ``.env`` entry like
        ``ASSEMBLYAI_API_KEY=xxx`` covers both the STT stage and wake detection
        without any duplicated variables.

        Mapping:
          ASSEMBLYAI_API_KEY  → conversation.stt  (if provider=assemblyai)
                                 wake              (always — wake uses AssemblyAI)
          DEEPGRAM_API_KEY    → conversation.stt  (if provider=deepgram)
          GEMINI_API_KEY /
          GOOGLE_API_KEY      → conversation.tts  (if provider=gemini)
                                 conversation.llm  (if provider=gemini)
          ELEVENLABS_API_KEY  → conversation.tts  (if provider=elevenlabs)
          OPENAI_API_KEY      → conversation.llm  (if provider=openai)
          ANTHROPIC_API_KEY   → conversation.llm  (if provider=anthropic)
        """

        def _env(*names: str) -> SecretStr | None:
            for name in names:
                v = os.environ.get(name)
                if v:
                    return SecretStr(v)
            return None

        stt = self.conversation.stt
        tts = self.conversation.tts
        llm = self.conversation.llm
        wake = self.wake

        # Propagate top-level language into STT so providers can set the
        # correct transcription language (better accuracy than auto-detect).
        if stt.language_code == "en" and self.language != "en":
            stt.language_code = self.language

        if not stt.api_key:
            if stt.provider == "assemblyai":
                stt.api_key = _env("ASSEMBLYAI_API_KEY")
            elif stt.provider == "deepgram":
                stt.api_key = _env("DEEPGRAM_API_KEY")

        if tts.language_code == "en" and self.language != "en":
            tts.language_code = self.language

        if not tts.api_key:
            if tts.provider == "gemini":
                tts.api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
            elif tts.provider == "elevenlabs":
                tts.api_key = _env("ELEVENLABS_API_KEY")

        if not llm.api_key:
            if llm.provider == "gemini":
                llm.api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
            elif llm.provider == "openai":
                llm.api_key = _env("OPENAI_API_KEY")
            elif llm.provider == "anthropic":
                llm.api_key = _env("ANTHROPIC_API_KEY")

        if not wake.api_key:
            wake.api_key = _env("ASSEMBLYAI_API_KEY")

        return self

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
