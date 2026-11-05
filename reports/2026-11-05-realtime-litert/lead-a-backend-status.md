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
