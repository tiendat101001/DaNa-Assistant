# Implementation, verification & runbook

Low-level companion to `01-architecture-deep-dive.md`. It covers what shipped
(file by file), how to configure and run it, the evidence that it works, the
measured latency, and the concrete plan to close the remaining gap to 600 ms.

---

## 1. What shipped — file map

All paths are under `dana-agent/src/dana_agent/` unless noted.

### 1.1 Model registry (HF `litert-community` → disk → runtime)
| File | Responsibility |
|---|---|
| `models/hub.py` | `HubClient`: whole-org catalog (`/api/models?author=…&full=true`, `Link` pagination, TTL cache keyed by search; task filter **before** limit), `model_detail` (`/tree/{rev}?recursive=true`, LFS sizes), `classify_task`, `recommended_files` (CPU: excludes `MediaTek_/Qualcomm_/_gpu` NPU variants, picks one precision tier, keeps every stage + sidecars) and `recommended_files_webgpu`, `COMPANIONS` (Whisper → `openai/whisper-*` tokenizer; Moonshine → `UsefulSensors/moonshine-*` tokenizer) |
| `models/pull.py` | `Downloader`/`PullJob`: streaming download to `<file>.part` → `os.replace` (atomic), `Range` resume, size verification, manual redirects (Authorization sent to the hub host only), path-traversal rejection before any network call, cancellable |
| `models/store.py` | `model_dir()` = `<models_dir>/<org>__<name>`, `resolve_model_path()`, `ModelStore` with atomic `registry.json`, `safe_relpath`/`resolve_in_model_dir` |
| `models/manager.py` | `ModelManager`: hub listing with `pulled` flags, pull jobs (semaphore `max_concurrent_pulls`, auto-load), **all-or-nothing multi-stage CPU load** into `MODEL_CACHE`, WebGPU "serve to browser" mode, unload, delete, safe file serving |
| `control/models_routes.py` | `/api/v1/models/*` (hub, hub detail, local, pull, jobs, load, unload, delete, assign, file) |

### 1.2 LiteRT runtime + speech providers
| File | Responsibility |
|---|---|
| `runtime/litert.py` | `ai_edge_litert` → `tflite_runtime` → `tensorflow.lite` fallback chain; `LiteRTModel` (signatures, runners, `invoke_default`, JSON `info()`); thread-safe `ModelCache`/`MODEL_CACHE`; `webgpu` → `ValueError` (browser-side runtime) |
| `runtime/audio.py` | numpy Whisper front-end: resample → 400-pt Hann STFT, hop 160 → Slaney mel (80/128) → log10 clamp → dynamic range → `(x+4)/4` → pad 3000 frames |
| `runtime/whisper_tokenizer.py` | byte-level BPE decoder (Whisper) and `SentencePieceDecoder` (Moonshine); `load_tokenizer(dir)` auto-selects |
| `providers/stt/litert.py` | `@register("stt","litert")`: strategy by introspection (Whisper encode/decode, Moonshine raw-audio windows, single-graph generate); greedy decoding with language prefix (`<|en|>` 50259 / `<|vi|>` 50278); energy gate (−45 dBFS); `warmup()` |
| `providers/stt/faster_whisper.py` | `@register("stt","faster_whisper")`: CPU int8 fallback (en/vi) |
| `providers/tts/litert.py` | `@register("tts","litert")`: KittenTTS 3-stage (predictor → prosody → vocoder) + bundled neural G2P (dictionary + DeepPhonemizer tflite, memoised), leading-silence clamp (`LEAD_FRAMES=3`), tail trim (100 ms), per-graph shape cache, 150 ms PCM chunks, `warmup()` |
| `providers/tts/piper.py` | `@register("tts","piper")`: Piper ONNX (new and legacy APIs), en + **vi** voices |
| `providers/tts/openai.py` | `@register("tts","openai")`: `gpt-4o-mini-tts` streaming PCM 24 kHz, odd-byte carry |

### 1.3 Realtime pipeline
| File | Responsibility |
|---|---|
| `conversation/processors.py` | `ProviderSTTProcessor` (bridges any `STTProvider`: 300 ms pre-roll ring, VAD-bounded utterance, background inference, 30 s cap); `ProviderTTSProcessor(first_chunk_min_chars)` + `eager_split()` (first clause at `, ; : — –` once ≥ N chars, number-safe) |
| `conversation/pipeline.py` | `SileroVADAnalyzer(VADParams(...from latency config))`; any registered non-Pipecat STT → `ProviderSTTProcessor` |
| `conversation/latency.py` | `TurnClock`: `stt`, `llm_ttft`, `tts_ttfb`, `voice_to_first_audio` (+ `over_budget`) latency events, VAD-delay compensated |
| `conversation/warmup.py` | `warmup_stack()`: concurrent, timed and fault-tolerant warm-up of STT/TTS/S2S `warmup()` and the brain's chat model |
| `conversation/session.py` | shared clock; warm-up before the pipeline starts (10 s bound) |

### 1.4 LLM hosting
| File | Responsibility |
|---|---|
| `brain/providers.py` | aliases `sglang`, `vllm`, `together` (→ OpenAI client); `DEFAULT_BASE_URLS` (self-hosted), `CLOUD_BASE_URLS` (Together), `SELF_HOSTED_PROVIDERS` |
| `brain/model.py` | per-provider knobs (`thinking_budget` → Gemini, `reasoning_effort` → OpenAI, `max_tokens`, `timeout`, `extra`); `EMPTY` key for self-hosted; `warmup_chat_model()` |
| `brain/probe.py` | `probe_llm()` → `{ok, ttft_ms, total_ms, text, error}` |
| `control/llm_routes.py` | `/api/v1/llm/presets` (Together, Gemini ×2, OpenAI ×3, SGLang ×3), `/probe`, `/health` |
| `scripts/serve_sglang.sh`, `.ps1` (repo root) | SGLang launcher (Linux/WSL2 + NVIDIA) |
| `brain/agent.py` | `Brain.model` exposed for warm-up |

### 1.5 Generic computer use
| File | Responsibility |
|---|---|
| `providers/actions/browser.py` | `browser_snapshot` (in-page DOM/shadow-DOM walker → `data-dana-ref` refs, ≤150 elements, fenced as untrusted data), `browser_act` (click/type/fill/select/check/uncheck/hover/focus/press by ref, stale-ref detection, live re-classification at click time), generic site search and first-result finder |
| `providers/actions/adapters.py` | built-in engines merged with `actions.search_engines`; `click_consequence()` shared vocabulary (destructive / purchase words, EN + VI) |
| `providers/actions/os_win.py` | `os_ui_snapshot` / `os_ui_act` via pywinauto UIA (dedicated thread), gated by `os_actions_enabled` + `allowed_apps` (process name) |
| `providers/actions/mcp.py` | `MCPActions`: stdio / streamable-http / sse sessions, namespaced tools, annotation-based classification, schema → `ToolSpec`, truncated, untrusted output |
| `brain/tools.py` | ToolSpecs for the new tools; `browser_tool_specs(cfg)` lists the configured engines |
| `app.py` | `_build_mcp_tools()` (own `ActionGate`, shared audit log), `_with_local_model_path()` (registry id → pulled dir) |
| `control/controller.py` | gate-policy hot-apply also updates the MCP gate |

### 1.6 UI (`dana-ui/`)
`src/api/models.ts`, `src/features/models/{ModelsView,HubBrowser,PullJobs,LocalModels,LlmHosting,LatencyBudget,WebGpuPanel}.tsx`, `webgpu.ts` (MediaPipe `@mediapipe/tasks-genai@0.10.30`, dynamically imported), `public/manifest.webmanifest`, `public/icon.svg`, and responsive CSS (760 px breakpoint).

---

## 2. Configuration reference (all new knobs)

Env form: `DANA_<SECTION>__<FIELD>`. Everything is also editable from the dashboard,
which renders `/api/v1/config/schema`.

| Key | Default | Meaning |
|---|---|---|
| `models.hub_author` | `litert-community` | HF org the registry browses |
| `models.models_dir` | `.dana/models` | pulled models (gitignored) |
| `models.hf_token` | from `HF_TOKEN` | gated repos / rate limits; never logged or returned |
| `models.catalog_ttl_s` | 600 | catalog cache TTL |
| `models.max_concurrent_pulls` | 2 | parallel pull jobs |
| `conversation.latency.target_first_audio_ms` | 600 | budget used for `over_budget` |
| `conversation.latency.vad_stop_secs` | 0.2 | end-of-utterance silence |
| `conversation.latency.tts_first_chunk_min_chars` | 20 | eager first-clause flush (0 = off) |
| `conversation.latency.warmup_on_start` | true | warm STT/TTS/LLM before the first turn |
| `conversation.latency.speculative_stt` | false | start STT after 60 ms of silence and reuse it at VAD stop (measured −240 ms) |
| `conversation.latency.ack_enabled` / `ack_phrases` | false / `{"en":"Mm-hm.","vi":"Dạ."}` | instant acknowledgement at end-of-speech (perceived ≈ 220 ms) |
| `conversation.latency.keepalive_s` | 0 (off) | token-free `models.list()` ping to keep the LLM connection warm during sessions |
| `conversation.stt.provider` | — | `litert` \| `faster_whisper` \| `assemblyai` \| `deepgram` |
| `conversation.stt.model` | — | e.g. `litert-community/moonshine-tiny` (EN) or `litert-community/whisper-tiny` (VI) |
| `conversation.{stt,tts}.num_threads` | 4 | = physical cores is best (measured) |
| `conversation.{stt,tts}.backend` | `cpu` | `webgpu` = browser runtime |
| `conversation.tts.provider` | — | `litert` (Kitten, EN) \| `piper` (VI) \| `openai` \| `gemini` \| `elevenlabs` |
| `conversation.llm.provider` | — | `together` \| `gemini` \| `openai` \| `sglang` \| `vllm` \| `ollama` \| … |
| `conversation.llm.{max_tokens, thinking_budget, reasoning_effort, timeout_s, extra}` | None | latency knobs (see presets) |
| `actions.search_engines` | `{}` | `{"name": "https://site/search?q={q}"}` |
| `actions.mcp_servers` | `[]` | `[{name, transport, command/args or url, allowed_tools, trust}]` |

Recommended low-latency English stack:
```bash
DANA_CONVERSATION__STT__PROVIDER=litert
DANA_CONVERSATION__STT__MODEL=litert-community/moonshine-tiny
DANA_CONVERSATION__TTS__PROVIDER=litert
DANA_CONVERSATION__TTS__MODEL=litert-community/kitten-tts-nano-0.8
DANA_CONVERSATION__TTS__VOICE=expr-voice-2-m
DANA_CONVERSATION__LLM__PROVIDER=together          # with TG_API_KEY / TG_BASE_URL / TG_MODEL_ID
DANA_CONVERSATION__LLM__MAX_TOKENS=160
DANA_CONVERSATION__LLM__EXTRA={"extra_body": {"reasoning": {"enabled": false}}}
DANA_CONVERSATION__LATENCY__SPECULATIVE_STT=true    # −240 ms measured
DANA_CONVERSATION__LATENCY__ACK_ENABLED=true        # optional: "Mm-hm." at ≈ 220 ms
```
Vietnamese: `STT__MODEL=litert-community/whisper-tiny`, `TTS__PROVIDER=piper` (a vi_VN voice).

---

## 3. Runbook

```bash
cd dana-agent
uv sync --dev --extra all                 # now includes litert, piper, models, mcp extras
cp .env.example .env                      # add HF_TOKEN (optional), TG_* or GEMINI/OPENAI keys
uv run python -m dana_agent --control     # http://127.0.0.1:8760 → 🧩 Models tab
```
1. **Models → Hub**: filter ASR / TTS, open `moonshine-tiny` → **Pull & Load**.
   Then do the same for `kitten-tts-nano-0.8`.
2. **Local models** → *Use as ASR* / *Use as TTS* (writes the config; restart-labelled fields are shown).
3. **LLM Hosting** → pick a preset (e.g. Together) → **Apply** → **Measure latency**.
4. **Talk** tab → speak, then watch the latency waterfall (`voice_to_first_audio`, red when over budget).

One-request API equivalent:
```bash
curl -X POST http://127.0.0.1:8760/api/v1/models/pull \
  -H "Authorization: Bearer $(cat .dana/control-token)" -H "Content-Type: application/json" \
  -d '{"repo_id":"litert-community/kitten-tts-nano-0.8","load":true,"backend":"cpu","num_threads":4}'
```

SGLang (Linux/WSL2 + NVIDIA): `MODEL=Qwen/Qwen2.5-1.5B-Instruct ./scripts/serve_sglang.sh`,
then apply the "SGLang Qwen2.5-1.5B" preset and use **Check server**.

---

## 4. Verification evidence

### 4.1 Backend gate (Team Lead A, final re-run)
| Gate | Result |
|---|---|
| Offline suite `pytest -m 'not live'` | **586 passed, 4 skipped, 0 failed** (final re-run after both latency sprints and the catalog-classification fix) (58 live deselected; `test_tray` import-guard test deselected: no X display in the container, already failing before this work) |
| `ruff check src tests scripts` | clean |
| `ruff format --check` | clean (157 files) |
| `mypy` (strict, whole project) | clean except 5 accepted `ctypes.windll` errors in `os_win.py` that only Linux mypy reports (already present before this work) |
| Real-model tests (whisper-tiny i8, moonshine-tiny i8, kitten-tts-nano fp16; `DANA_TEST_*` env) | passed |
| Live HF registry test (`test_models_live.py`: catalog, detail, real pull) | passed |

New offline tests by area: registry ≈ 62, LiteRT runtime/ASR ≈ 18, TTS ≈ 22,
latency pipeline 25, LLM + MCP 41, generic computer use 36, multi-stage load 2.

### 4.2 Product gate (Team Lead B, real backend + real huggingface.co, Playwright)
Run at 1280 px desktop and a 360 px phone viewport: **all flows pass on both**, with
no JS errors and no horizontal scroll. Screenshots are in `screenshots/`
(`desk-*` / `phone-*`).

| Flow | Result |
|---|---|
| Hub catalog loads (live HF; 100 cards per page) | PASS |
| TTS filter shows Kokoro-82M and kitten-tts-nano-0.8 (14 TTS repos) | PASS |
| Detail drawer: files listed, recommended pre-checked (README + `moonshine_tiny_5s_i8.tflite`) | PASS |
| **Pull & Load** `moonshine-tiny` in one click: progress bar → loaded (49.6 MB, cpu, 4 threads) | PASS |
| Local models table → **Use as ASR** (shows the config diff, flagged restart-required) | PASS |
| LLM Hosting: 9 presets render; **Measure latency** with the Together preset | PASS (see §5.2 note) |
| **Generic computer use**, real `BrowserActions` through `ActionGate` on a local page: snapshot → refs e1..e5; fill by ref → ALLOW (reversible) and executed; click "Search" → ALLOW; click **"Delete account" → CONFIRM (destructive), not executed**; click external link → **REFUSE (out of scope)** | PASS |
| **MCP bridge**, real stdio FastMCP server: tools namespaced `mcp_<server>_<tool>`; read-only → ALLOW and output fenced as untrusted; destructive / unannotated → CONFIRM; unknown → REFUSE | PASS after a fix |

Bugs found by this gate and fixed by their owners:
1. **MCP 2.x annotations are snake_case** (`read_only_hint`). Read-only tools were
   being classified destructive. That failed safe, but it was wrong. Both
   spellings are now read (C5, with a test).
2. **LLM probe ignored env credentials** when the UI switched provider. The probe now
   resolves config exactly like startup, and it never reuses one vendor's key for
   another provider (C5, with a test).
3. **Catalog misclassification:** MoGe-2 (`depth-estimation`) appeared under TTS
   because of substring keyword matching. `pipeline_tag` now decides, and the
   fallback matches whole tokens only. Re-verified over all 443 live repos: no
   vision repo is in ASR/TTS (C1, +15 tests built from real entries).
4. **Cold hub fetch (~15 s) showed an empty grid.** A loading status line and
   360 px layout guards were added (UX).


Owner-verified during integration (live, this session):
- **Registry, live HF:** the catalog listed 443 repos. `POST /pull {load:true}` for
  `kitten-tts-nano-0.8` downloaded 10/10 files (62.9 MB) in ~10 s and loaded.
  A bug found in this run is fixed: the task filter ran after `limit`, which hid
  Kokoro/Kitten. The partial-load issue is fixed too: only 1 of 3 Kitten stages
  had loaded, and loading is now all-or-nothing across all stages
  (`tests/test_models_multistage.py`).
- **LLM, live Together:** CRLF in the credentials file broke the URL. Env values
  are now stripped, and there is a test. Measured warm TTFT is ~375 ms with
  reasoning off.

---

## 5. Measured latency and the gap-closing plan

### 5.1 Measured (real components, warm, median of 3; 4-vCPU container)
Produced by `dana-agent/scripts/latency_bench.py`
(`uv run python scripts/latency_bench.py --runs 3`; reads `TG_*` from env only).

| Stage | Component | median ms (min / max) |
|---|---|---|
| End-of-speech | Silero VAD `stop_secs` | 200 (config) |
| STT | LiteRT **Moonshine-tiny** i8, 3.6 s utterance | **423** (415 / 454) |
| STT | LiteRT Whisper-tiny i8, 3.6 s utterance | 748 (720 / 761) |
| LLM TTFT | Together DeepSeek-V4.1-Flash via the real `Brain`, **26 tools bound**, reasoning off, max_tokens 80 | **426** (420 / 435) |
| LLM TTFT | same model, plain probe (no tools) | 453 (421 / 596) |
| TTS first chunk | LiteRT **KittenTTS** fp16, 21-char first clause | **138** (131 / 174) |
| **voice → first audio** | VAD + Moonshine + LLM + Kitten | **≈ 1,187 ms** (budget 600; over by ≈ 590) |
| voice → first audio | VAD + Whisper + LLM + Kitten | ≈ 1,511 ms |

**Before → after, same leg definitions:** TTS first audio went from ~1 s+ (Gemini TTS; ~1 s for
Kitten before the silence clamp) to 138 ms. LLM went from ~500 ms–1.5 s (reasoning/thinking
on; 7.5 s cold) to 426 ms warm with tools bound. STT is now on-box (no network), and silence
costs ~1 ms. End-to-end went from ≈ 2–4 s to ≈ 1.2 s.

**Accuracy note:** the bench clip is synthetic espeak speech. Moonshine:
"The low, what is the weather like today in Hanoi?"; Whisper-tiny clipped "Hanoi" →
"Panor". Real-voice WER still needs measuring on the user's microphone (KPI in
`business-case.md`).


### 5.2 Latency sprint (Team Lead A, after the first measurement)

**LLM sweep on Together.** Measured through the real `Brain` with 26 tools bound,
max_tokens 80, reasoning off, warm median of 3. Of ~60 listed chat models, 20 are
serverless for this key. Several small "Turbo" models are not serverless here
(403 / non-serverless).

| Model | TTFT ms (min / max) | Vietnamese |
|---|---|---|
| **deepseek-ai/DeepSeek-V4.1-Flash** (configured) | **431** (403 / 453) | good |
| together/Tev1-4B-experimental | 527 (477 / 612) | poor (truncated / refuses) |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL | 530 (505 / 657) | fair (hallucinated an action) |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | 532 (435 / 571) | good |
| Qwen/Qwen3.5-9B | 590 (531 / 749) | good |
| zai-org/GLM-5.3-Flash | 658 (633 / 980) | empty reply at 80 tokens |
| MiniMaxAI/MiniMax-M3.1-Flash | 666 (588 / 874) | good |
| meta-models/Muse-Glimmer-30B | 817 (507 / 861) | empty reply |
| openai/gpt-oss-120b | 4126 | n/a |

**Decision:** keep DeepSeek-V4.1-Flash. The plain probe sits at ~370–450 ms, which
appears to be the network round-trip floor from this container. A local GPU LLM
(SGLang) is the only way below it.

**Opt-in instant acknowledgement** (`conversation.latency.ack_enabled`, default
**off**). At warm-up, the active TTS synthesises `ack_phrases[lang]` ("Mm-hm." /
"Dạ.") once and caches it. At end-of-speech in an active voice turn, the cached
audio plays immediately. Too-short or too-quiet speech gets no ack, and typed
turns never do. It is logged as `ack_first_audio`, while `voice_to_first_audio`
keeps measuring the *real* reply, so the metric stays honest. **Perceived first
audio ≈ 220 ms** (200 ms VAD stop + one 20 ms frame). The real reply still lands
at ≈ 1.2 s with Moonshine on this container. Covered by `tests/test_latency_ack.py`.

**UI probe vs direct (Team Lead B).** "Measure latency" first showed 1.3–1.4 s,
because the UI sent only `{provider, model, base_url}` and dropped the preset's
`reasoning.enabled=false` and `max_tokens`. After the fix, the probe sends the
full preset and runs twice, reporting **warm 368 ms** (desktop) / 394 ms (phone)
and **cold 1,133 ms**. With the minimal body, warm is ≈ 470–540 ms, so hidden
reasoning costs ≈ 100–150 ms. The cold ≈ +750 ms is **connection setup**
(TLS/proxy), and it recurs after any idle gap longer than the HTTP keep-alive
window. That is why session warm-up and a connection keep-warm matter as much as
model choice (§5.3 #8).

### 5.3 Closing the remaining gap (ranked by ms saved per effort)
| # | Lever | Expected saving | Effort | Notes |
|---|---|---|---|---|
| 1 | **Incremental STT**: run Moonshine on ≤1–2 s windows *during* speech, so only the tail remains at end-of-speech | −150…−250 ms | M | encode ≈ 30 ms/window; reuse `ProviderSTTProcessor`'s ring buffer |
| 2 | **Local GPU LLM** (SGLang, Qwen2.5-1.5B / Qwen3-1.7B) or Gemini Flash-Lite with thinking 0 | −200…−300 ms | S (config) | presets exist; needs an NVIDIA GPU (WSL2) |
| 3 | **Speculative first clause**: a short acknowledgement synthesised from a cache while the LLM thinks | −150…−250 ms perceived | S | pre-synthesised PCM per language |
| 4 | **Prompt-prefix caching**: keep system prompt + 2.3k-token tool schema byte-identical across turns | −50…−150 ms | S | OpenAI-compatible servers cache by prefix |
| 5 | **Smaller first clause** (`tts_first_chunk_min_chars` 12–15) | −50…−100 ms | XS | Kitten costs ≈ 7 ms/char |
| 6 | **Vietnamese ASR**: whisper-tiny has a static 128-token decoder (≈ 65 ms/token). Options: a `Qwen3-ASR-0.6B` LiteRT trial, or a cloud streaming STT for VI only | −500 ms+ (VI) | M | per-language STT routing is a config change |
| 7 | Hardware: the container has 4 vCPU; a desktop CPU with 8+ performance cores | −20…−40 % compute | — | set `num_threads` = physical cores |
| 8 | **Connection keep-warm** during sessions (shipped as `keepalive_s`, opt-in) | up to −750 ms on post-pause turns where the provider/proxy closes idle connections | done | no gain measured on Together after a 90 s idle; kept as a safeguard |

### 5.4 Final sprint (shipped, opt-in, tested)
| Feature | Mechanism | Measured |
|---|---|---|
| **Speculative STT** (`speculative_stt`) | per-frame RMS in `ProviderSTTProcessor`; after 60 ms of low energy, transcribe the buffered audio in the background; cancel if speech resumes; reuse at VAD stop (re-run on error) | real Moonshine, VAD stop → transcript: **498 / 380 / 442 ms → 196 / 224 / 200 ms (≈ −240 ms)** |
| **Keep-warm** (`keepalive_s`) | token-free `root_async_client.models.list()` on the model's own pool every N s during a session; cancelled on stop | Together after 90 s idle: no cold penalty observed (off 376 / 383 ms vs keepalive 488 / 448 ms), so no benefit on this endpoint |

**Final projection (EN, this 4-vCPU container, everything on):**
VAD 200 + speculative STT ≈ 223 + LLM 426 + Kitten 138 ≈ **≈ 0.99 s** to the real reply's
first audio. **≈ 220 ms** to the instant acknowledgement. Started at ≈ 2–4 s.
Costs and risks: speculative STT spends extra compute on mid-utterance pauses,
and a cancelled run can briefly hold the LiteRT lock.

---

## 6. Known limitations (honest list)

- Sub-600 ms is **not yet** met end-to-end on this container (≈ 1.1 s measured
  sum). See §5.
- No Vietnamese LiteRT TTS exists in the org, so Vietnamese uses Piper. A Piper
  vi_VN voice must be downloaded by the user (not done here).
- Whisper-tiny Vietnamese ASR takes ~1–1.5 s, because the decoder window is static.
- `os_ui_*` (Windows UIA) and the Windows-specific paths were validated with fakes
  only; this container is Linux.
- The WebGPU runner uses MediaPipe with wasm from jsDelivr (needs internet). Hosting
  the wasm under `dana-ui/public/` makes it air-gapped.
- There is no service worker yet: the PWA is installable but not offline.
- The wake-gate's first utterance after "Hey DaNa" is consumed by wake detection
  (pre-existing behaviour).
- The HF token pasted in chat should be **rotated**. It was never written to any
  file in the repo.
