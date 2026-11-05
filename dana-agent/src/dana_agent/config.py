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
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from dana_agent.secrets import resolve_secret

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


class LocalModelFields(BaseModel):
    """Shared knobs for self-hosted, on-box model runtimes (LiteRT, Piper,
    faster-whisper, ...). Ignored by cloud providers.

    ``model`` (on the owning config) names the model: for ``litert`` it is a
    model-registry id (a HuggingFace repo id such as
    ``litert-community/whisper-tiny``) resolved under ``models.models_dir``;
    ``model_path`` overrides that with an explicit local file/directory.
    """

    backend: Literal["cpu", "webgpu"] = "cpu"
    num_threads: int = Field(default=4, ge=1, le=64)
    model_path: str | None = None


class STTConfig(ProviderConfig, LocalModelFields):
    model: str = "mock-stt"
    language_code: str = "en"  # propagated from AppConfig.language by _apply_standard_api_keys


class LLMConfig(ProviderConfig):
    model: str = "mock-llm"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # --- latency knobs (None = provider default, nothing sent) ---
    # Cap reply length: voice replies are short; fewer tokens = earlier end.
    max_tokens: int | None = Field(default=None, gt=0)
    # Gemini 2.5 "thinking" adds hundreds of ms before the first token; 0 turns
    # it off (Flash / Flash-Lite). Ignored by non-Gemini providers.
    thinking_budget: int | None = Field(default=None, ge=-1)
    # OpenAI reasoning models (gpt-5*, o*): "minimal" = fastest first token.
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    # Request timeout (s) so a stalled endpoint fails fast instead of hanging a turn.
    timeout_s: float | None = Field(default=None, gt=0)
    # Escape hatch: extra kwargs passed straight to init_chat_model.
    extra: dict[str, Any] = Field(default_factory=dict)


class TTSConfig(ProviderConfig, LocalModelFields):
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


class LatencyConfig(BaseModel):
    """Realtime budget knobs. Target: first audio < ``target_first_audio_ms``
    after the user stops speaking (end-of-speech -> first TTS byte)."""

    target_first_audio_ms: int = Field(default=600, gt=0)
    # Silence that ends an utterance (Pipecat 1.7 VADParams default: 0.2 s).
    # Every 100 ms here is 100 ms of first-audio latency; raise it only if users
    # get cut off mid-sentence.
    vad_stop_secs: float = Field(default=0.2, ge=0.05, le=3.0)
    vad_start_secs: float = Field(default=0.2, ge=0.05, le=2.0)
    vad_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    # Eager first TTS chunk: flush the first clause (at , ; : - or a sentence
    # end) as soon as it holds >= this many chars, instead of waiting for a full
    # sentence. 0 disables eager flushing (sentence-level only).
    tts_first_chunk_min_chars: int = Field(default=20, ge=0)
    # Pre-warm providers (load local models, open HTTP connections, one tiny
    # LLM call) at startup so the first real turn doesn't pay cold-start cost.
    warmup_on_start: bool = True
    # Instant acknowledgement (opt-in): at end of speech, play a cached short
    # phrase ("Mm-hm.") from the active TTS so the user hears the bot right
    # away while STT/LLM run. Synthesized once per language at startup.
    # Keep the LLM connection warm: ping the endpoint every N seconds while a
    # session runs (token-free models.list() on OpenAI-compatible clients).
    # 0 = off.
    keepalive_s: float = Field(default=0.0, ge=0.0)
    # Speculative STT: start transcribing as soon as ~60 ms of silence follows
    # speech, reuse the result if the VAD then confirms end-of-speech.
    speculative_stt: bool = False
    ack_enabled: bool = False
    ack_phrases: dict[str, str] = Field(default_factory=lambda: {"en": "Mm-hm.", "vi": "Dạ."})


# --- grouped configs -----------------------------------------------------


class ConversationConfig(BaseModel):
    """A conversation provider is either a cascade (STT+LLM+TTS) or a single
    speech-to-speech model. Default to cascade (docs/PLAN.md §6)."""

    mode: Literal["cascade", "s2s"] = "cascade"
    # When True the mic is gated off while the bot is speaking so the bot's
    # own TTS audio cannot loop back into STT and be treated as user input.
    # Trades lower perceived latency (no barge-in) for echo safety.
    no_interrupt: bool = False
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    s2s: S2SConfig = S2SConfig()
    latency: LatencyConfig = LatencyConfig()


class MCPServerConfig(BaseModel):
    """One Model Context Protocol server whose tools DaNa may call (gated)."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    enabled: bool = True
    transport: Literal["stdio", "streamable_http", "sse"] = "stdio"
    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # http transports
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    # Optional allowlist of tool names (empty = all tools the server lists).
    allowed_tools: list[str] = Field(default_factory=list)
    # Trust: "annotations" = honour readOnlyHint/destructiveHint from the server;
    # "confirm_all" = treat every tool as destructive (always ask first).
    trust: Literal["annotations", "confirm_all"] = "annotations"

    @model_validator(mode="after")
    def _check_transport(self) -> MCPServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"mcp server {self.name!r}: stdio transport needs 'command'")
        if self.transport != "stdio" and not self.url:
            raise ValueError(f"mcp server {self.name!r}: {self.transport} transport needs 'url'")
        return self


class ActionsConfig(ProviderConfig):
    """Action layer selection plus the allowlists and gate policy that bound
    it (safe-computer-use skill).

    Safe defaults: OS actions are OFF, and both allowlists are EMPTY — an empty
    allowlist is default-deny (nothing is navigable / launchable) so a fresh
    install can act on nothing until the user opts specific targets in.
    """

    allowed_domains: list[str] = Field(default_factory=list)
    allowed_apps: list[str] = Field(default_factory=list)
    gate_policy: Literal["standard", "conservative", "wide"] = "standard"
    max_actions_per_turn: int = Field(default=10, gt=0)
    # Hard-refuse purchase-committing clicks outright — stronger than the
    # destructive-word confirmation, for accounts that must never be charged
    # by voice. Restart-required on hot-apply.
    forbid_purchases: bool = False

    # --- browser (Playwright) provider ---
    # Dedicated, disposable profile dir — NEVER the user's real Chrome profile.
    browser_profile_dir: str = ".dana/browser-profile"
    # Portable auth blob exported on close (contains live cookies → treat as a
    # secret; lives under .dana/ which is gitignored).
    browser_storage_state_path: str = ".dana/browser-state.json"
    # Visible by default so the user can watch and trust what the agent does.
    browser_headless: bool = False

    # --- OS (Windows) provider ---
    # Bounded OS control is dangerous → OFF by default. Even when the provider is
    # selected, execution is refused unless this is explicitly enabled.
    os_actions_enabled: bool = False

    # --- audit ---
    # Durable append-only JSONL sink for every gate decision/execution.
    audit_path: str = ".dana/audit.jsonl"

    # --- generic computer use ---
    # User-defined site search engines: name -> URL template containing "{q}",
    # e.g. {"shopee": "https://shopee.vn/search?keyword={q}"}. Merged over the
    # built-ins so any (allowlisted) site is searchable without code changes.
    search_engines: dict[str, str] = Field(default_factory=dict)
    # Plug-and-play tool servers (Model Context Protocol). Every MCP tool is
    # routed through the same ActionGate (classified from its annotations).
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)


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
        "browser or computer. When asked about 'this paper', 'bài đang xét', "
        "or what page/article is open, use browser_read_text or browser_get_state "
        "to inspect the open page first. On any site or app you don't know, call "
        "browser_snapshot (or os_ui_snapshot for desktop apps), then act by ref with "
        "browser_act / os_ui_act; snapshot again after the page changes."
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

    # Off by default: the agent runs headless with zero control-plane overhead
    # and no fastapi/uvicorn import unless this is enabled (or --control passed).
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8760, gt=0, lt=65536)
    # Path to the local bearer-token file (not a secret value itself).
    token_path: str = ".dana/control-token"  # noqa: S105
    # Directory holding the built dana-ui bundle to serve in production. When it
    # exists the control app mounts it at "/"; in dev the UI runs under Vite.
    ui_dist_dir: str = "../dana-ui/dist"


class ModelsConfig(BaseModel):
    """Self-hosted model registry (HuggingFace Hub -> local disk -> runtime)."""

    hub_endpoint: str = "https://huggingface.co"
    # Default organisation browsed by the registry (LiteRT community models).
    hub_author: str = "litert-community"
    # Read from HF_TOKEN / HUGGING_FACE_HUB_TOKEN when unset. Needed for gated
    # repos (e.g. Gemma) and higher rate limits. Never logged or returned.
    hf_token: SecretStr | None = None
    models_dir: str = ".dana/models"
    catalog_ttl_s: float = Field(default=600.0, ge=0.0)
    # Files pulled by default when a request names no explicit file list.
    default_include: list[str] = Field(
        default_factory=lambda: [
            "*.tflite",
            "*.litertlm",
            "*.task",
            "*.json",
            "*.txt",
            "*.model",
            "*.md",
        ]
    )
    max_concurrent_pulls: int = Field(default=2, ge=1, le=8)
    request_timeout_s: float = Field(default=30.0, gt=0)


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
    models: ModelsConfig = ModelsConfig()
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
            # resolve_secret consults the OS credential vault (keyring) first,
            # then os.environ (which already includes anything from .env). The
            # explicit DANA_*__API_KEY form still wins — it is parsed by
            # pydantic-settings before this validator runs and we only fill a
            # key that is still unset. So precedence is:
            #   DANA_*__API_KEY  >  keyring  >  plain env / .env
            for name in names:
                v = resolve_secret(name)
                # .strip(): a CRLF .env / `source`d Windows file leaves a trailing
                # "\r" that breaks auth headers and URLs.
                if v and v.strip():
                    return SecretStr(v.strip())
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
            elif llm.provider == "together":
                # Together AI (OpenAI-compatible): TG_* names as provided by the
                # operator, TOGETHER_API_KEY as the vendor-standard fallback.
                llm.api_key = _env("TG_API_KEY", "TOGETHER_API_KEY")
            elif llm.provider in ("sglang", "vllm", "openai_compatible", "ollama", "lmstudio"):
                # Self-hosted OpenAI-compatible servers usually accept any key;
                # honour an explicit one if the operator set it.
                llm.api_key = _env("SGLANG_API_KEY", "OPENAI_COMPATIBLE_API_KEY")

        if llm.provider == "together":
            if llm.base_url is None:
                llm.base_url = os.environ.get("TG_BASE_URL", "").strip() or None
            if llm.model in ("", "mock-llm"):
                llm.model = os.environ.get("TG_MODEL_ID", "").strip() or llm.model

        if not tts.api_key and tts.provider == "openai":
            tts.api_key = _env("OPENAI_API_KEY")

        if not wake.api_key:
            wake.api_key = _env("ASSEMBLYAI_API_KEY")

        if not self.models.hf_token:
            self.models.hf_token = _env("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")

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
