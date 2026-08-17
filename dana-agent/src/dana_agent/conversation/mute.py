"""Hard mic mute — a real capture gate, not a soft flag (docs/PLAN.md §8).

``MicGate`` is a tiny, dependency-free state object shared between the control
plane (which flips it) and the audio pipeline (which reads it). When muted, the
:class:`MuteProcessor` in ``conversation.processors`` *drops* every
``InputAudioRawFrame`` at the head of the pipeline, so captured audio never
reaches STT, the brain, or any provider — the honest interpretation of a hard
mute for an in-process pipeline. (True device-level mute is a future
enhancement; dropping frames at the source already guarantees no capture
egresses.)

This module imports nothing from ``dana_agent.control`` (the pipeline must stay
decoupled from FastAPI) and nothing from pipecat (so it is offline-testable). The
control layer injects an ``on_change`` callback to emit a ``mic_state`` event.
"""

from __future__ import annotations

from collections.abc import Callable


class BotSpeakingGate:
    """Shared state so BrainProcessor can signal HalfDuplexGateProcessor.

    Set to ``speaking=True`` when the bot's TTS starts, ``False`` when it
    finishes.  The :class:`~dana_agent.conversation.processors.HalfDuplexGateProcessor`
    at the head of the pipeline reads this flag to silently drop
    ``InputAudioRawFrame`` and VAD boundary frames while the bot is talking,
    preventing its own audio from reaching STT and being mistaken for user input.

    Only active when ``DANA_CONVERSATION__NO_INTERRUPT=true``.
    """

    def __init__(self) -> None:
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    def set_speaking(self, value: bool) -> None:
        self._speaking = value


class MicGate:
    """Shared mute state for the capture path.

    ``on_change`` (optional) is invoked with the new muted state whenever it
    flips — the control plane wires it to publish a ``mic_state`` event. It is
    never invoked on a no-op toggle.
    """

    def __init__(
        self, *, muted: bool = False, on_change: Callable[[bool], None] | None = None
    ) -> None:
        self._muted = muted
        self._on_change = on_change

    @property
    def muted(self) -> bool:
        return self._muted

    def mute(self) -> bool:
        """Mute capture. Returns True if this changed the state."""
        return self._set(True)

    def unmute(self) -> bool:
        """Resume capture. Returns True if this changed the state."""
        return self._set(False)

    def toggle(self) -> bool:
        """Flip the state; returns the new muted value."""
        self._set(not self._muted)
        return self._muted

    def _set(self, value: bool) -> bool:
        if value == self._muted:
            return False
        self._muted = value
        if self._on_change is not None:
            self._on_change(value)
        return True
