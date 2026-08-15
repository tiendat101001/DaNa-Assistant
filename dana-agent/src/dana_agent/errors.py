"""Domain exception hierarchy.

Callers handle intent, not string-matched messages (python-service-scaffold
skill). Fail fast and loud on programmer/config errors; degrade gracefully on
expected external failures.
"""

from __future__ import annotations


class DanaError(Exception):
    """Base class for all DaNa domain errors."""


class ConfigError(DanaError):
    """Configuration is invalid: unknown provider, missing key, or an
    incompatible capability combination. Raised at startup, never at first use."""


class ProviderError(DanaError):
    """A provider failed at runtime (network, auth, backend). Expected and
    recoverable — the app degrades rather than crashing."""


class CapabilityError(ConfigError):
    """A selected provider does not support something the configured stack
    requires (e.g. a language, streaming, or tool-calling)."""


class ActionRefused(DanaError):
    """The action gate refused a tool call: off-allowlist, unvalidated
    argument, or a destructive action lacking confirmation."""
