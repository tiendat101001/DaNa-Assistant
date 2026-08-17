from dana_agent.providers.vad import mock  # noqa: F401  (registration)

try:
    from dana_agent.providers.vad import silero  # noqa: F401  (registration)
except ImportError:
    pass  # pipecat not installed — silero provider unavailable, mock is used
