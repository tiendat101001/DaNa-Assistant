# Lead A (Backend & Realtime) status

| Task | Owner | Status | Evidence |
|---|---|---|---|
| Full offline suite | Lead A | green | 557 passed, 4 skipped, 58 deselected (live), tray test deselected |
| ruff check / format | C5, Lead A | green | 2 test files unformatted (test_action_browser, test_control_api) reformatted by Lead; test_llm_model by C5 |
| mypy strict | C5 | green except known | 3 test_llm_model.py errors fixed by C5; only the 5 accepted os_win.py windll errors remain |
| Real-model tests (stt_litert, runtime_litert, tts_local) | Lead A | pass | 40 tests, all pass with DANA_TEST_* env |
| HF live test (test_models_live) | Lead A | pass | 1 test passed |
| latency_bench.py | C3 | done, 26 tools bound | scripts/latency_bench.py |

## Measured (first run, Brain showed 0 tools - being fixed)
Moonshine 423, Whisper 748, LLM TTFT 426 (probe 453), Kitten 138 ms (26 tools).
Projected: Moonshine 1187 ms (over 587), Whisper 1511 ms (over 911) vs 600 ms.

## Risks
- Budget missed by ~585 ms with local STT; STT (370) and LLM TTFT (425) dominate.
- tray test fails without X display (pre-existing).

## Latency sprint
### 1. LLM sweep (Together serverless, Brain + 26 tools, max_tokens 80, reasoning disabled, warm median of 3)
Of ~60 priced chat models on /models only 20 are actually serverless for this key; the rest (Llama-3.1-8B-Turbo, Qwen2.5-7B-Turbo, Llama-3.2-3B, gpt-oss-20b, Llama-4-Scout, Qwen3.8-Flash [403]) return "non-serverless" and cannot be used.

| Model | TTFT ms (min/max) | Vietnamese |
|---|---|---|
| deepseek-ai/DeepSeek-V4.1-Flash (current) | 431 (403/453) | good |
| together/Tev1-4B-experimental | 527 (477/612) | poor (truncated, refuses) |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL | 530 (505/657) | fair (claims it opened page) |
| DeepSeek-V4.1-Flash-1017 | 532 (435/571) | good |
| Qwen/Qwen3.5-9B | 590 (531/749) | good |
| MiniMaxAI/MiniMax-M3.1-Flash | 666 (588/874) | good |
| zai-org/GLM-5.3-Flash | 658 (633/980) | empty reply at 80 tok (reasoning eats budget) |
| meta-models/Muse-Glimmer-30B | 817 (507/861) | empty reply |
| openai/gpt-oss-120b | 4126 | n/a (reasoning) |

Recommendation: keep DeepSeek-V4.1-Flash; no serverless small model beats it. Plain probe ~370-450 ms is the network floor.

### 2. Opt-in instant acknowledgement (C3)
Status: done, default off. config.py gained only `conversation.latency.ack_enabled=False` and `ack_phrases={"en":"Mm-hm.","vi":"Dạ."}`. Cached PCM per language at warmup; played on VAD stop in an ACTIVE voice turn only; TurnClock emits `ack_first_audio`, `voice_to_first_audio` unchanged. Tests: tests/test_latency_ack.py, 7 tests (enabled/disabled, cache miss, empty transcript, ordering). Bench projects perceived first audio with ack = 200 + 20 = 220 ms (real reply still ~1.2 s).

### Final gates
Offline suite 566 passed, 4 skipped, 58 live deselected; ruff check/format clean; mypy only the 5 known os_win.py errors.
