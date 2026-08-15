# Recorded audio fixtures

Phase 2 populates this directory with short recorded clips (wake phrase, a
command, silence, plus Vietnamese variants) used to drive the wake gate and
state machine offline — no live microphone. Assertions target state
transitions and emitted events, never audio bytes (voice-agent-architecture
skill).

Until then, the mock providers accept sentinel PCM (the phrase text encoded as
utf-8) so the state machine can be exercised without real audio.
