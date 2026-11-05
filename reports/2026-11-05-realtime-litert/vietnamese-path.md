# Vietnamese voice path (piper TTS + LiteRT Whisper-tiny STT)

Verified 2026-11-05 with DaNa's real providers (`PiperTTS`, `LiteRTWhisperSTT`), CPU, 4 threads. Scripts: /tmp/claude-0/vi/s/{run,val}.py. No src edits.

## Voices (rhasspy/piper-voices, vi/vi_VN)
Available: `vais1000/medium` (63 MB, 22.05 kHz), `25hours_single/low` (63 MB, 16 kHz), `vivos/x_low` (28 MB, multi-speaker, not tested).
Recommended: **vais1000/medium** (better quality). `25hours_single/low` is ~3x faster.

## Install
Download both files per voice (public, no token) into one directory, e.g. `.dana/models/`:
```
B=https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium
curl -L -O $B/vi_VN-vais1000-medium.onnx
curl -L -O $B/vi_VN-vais1000-medium.onnx.json    # must sit next to the .onnx
```
Env (voice looked up as `<models_dir>/<voice>.onnx`, so `voice` = file stem; or use model_path):
```
DANA_LANGUAGE=vi
DANA_CONVERSATION__STT__PROVIDER=litert
DANA_CONVERSATION__STT__MODEL=litert-community/whisper-tiny
DANA_CONVERSATION__TTS__PROVIDER=piper
DANA_CONVERSATION__TTS__VOICE=vi_VN-vais1000-medium
# or explicit: DANA_CONVERSATION__TTS__MODEL_PATH=.dana/models/vi_VN-vais1000-medium.onnx
```
STT also needs the whisper tflite plus tokenizer dir `openai__whisper-tiny` (vocab.json, added_tokens.json) under models_dir.

## Measured (warm; text "Xin chào, tôi là DaNa. Hôm nay tôi có thể giúp gì cho bạn?")
| voice | audio | synth total | RTF | TTFC full sentence | TTFC 20-char clause (5 runs) |
|---|---|---|---|---|---|
| vais1000/medium | 3.02 s | 905 ms | 0.30 | 345 ms | 224-377 ms (median 291) |
| 25hours_single/low | 4.00 s | 363 ms | 0.09 | 234 ms | 79-123 ms (median 86) |

Note: medium misses the 600 ms first-audio budget only when combined with slow STT/LLM; TTFC alone is fine. WAVs: /tmp/claude-0/vi/*.wav.

## STT (whisper_tiny_30s_i8.tflite, language_code="vi", warm, 3 runs)
- On vais1000 wav: 1341-2007 ms. Transcript: `Xin chào, tôi là Stanna, không này tôi có thể dút gì cho bạn.`
- On 25hours wav: 1199-1312 ms. Transcript: `Xin chào tôi là dân nạ, mong này tôi có thể chút sị trách bà.`
- Pipeline works end to end and decodes Vietnamese with diacritics, but whisper-tiny int8 accuracy is poor ("DaNa" -> "Stanna", "Hôm nay"/"giúp" garbled). The 30 s fixed window makes latency ~1.2-2 s regardless of utterance length: over the 600 ms target. Consider whisper-base/small or faster-whisper for vi.

## Validator (`validate_app_config`, language=vi; stt/tts live under `conversation`)
- STT litert + TTS piper: PASS.
- TTS litert (Kitten): FAIL, `CapabilityError: TTS provider 'litert' does not support configured language 'vi'; it supports ['en']`. Clear.

## Issues
- No provider bugs found. Quality issue only: whisper-tiny vi accuracy and ~1.2-2 s latency.
- Minor: `TTS voice` must equal file stem (not repo-style `vi_VN/vais1000/medium`); worth a README line.
