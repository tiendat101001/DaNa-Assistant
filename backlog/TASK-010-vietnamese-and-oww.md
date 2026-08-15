# TASK-010 — Vietnamese quality + openWakeWord "Hey DaNa" training

- **Status:** TODO
- **Priority:** P3
- **Source:** docs/PLAN.md §1 (languages), §10 (risks), voice-agent skill.

## Goal

Harden the en + vi requirement and provide a free wake-engine path.

## Acceptance criteria

- [ ] Vietnamese recorded-audio fixtures (wake, command, silence) in the FSM suite.
- [ ] Vietnamese STT/TTS quality pass: Deepgram Nova-3 tones vs faster-whisper
      fallback; ElevenLabs vi vs Piper `vi_VN` naturalness — pick defaults.
- [ ] Train an openWakeWord model for "Hey DaNa" (the free wake path); document
      the training + drop-in steps. Porcupine remains the v1 default until then.
- [ ] Decide on (and defer or implement) automatic en↔vi code-switching within a
      turn — hard problem, explicitly out of v1 scope for now.
