# DaNa realtime upgrade — architecture deep dive

From the high-level concept down to the mechanism of each component. Companion
files in this folder: `README.md` (index + executive summary),
`02-implementation-and-verification.md` (what shipped, tests, measurements),
`00-team-contract.md` (API contracts), research/business/UX/mobile notes.

---

## 1. The problem in one picture

The voice loop is a relay race; the user only feels the **sum of the legs up to
the first audio byte**:

```
user stops talking
   │  ① end-of-speech detection (VAD silence window)
   ▼
STT final transcript
   │  ② STT finalisation
   ▼
LLM first token
   │  ③ LLM time-to-first-token (TTFT): network + prompt prefill + "thinking"
   ▼
first speakable chunk of text
   │  ④ chunking policy (how much text we wait for before calling TTS)
   ▼
TTS first audio byte
   │  ⑤ TTS time-to-first-byte (TTFB)
   ▼
speaker plays            ← target: ① + ② + ③ + ④ + ⑤ < 600 ms
```

### Where the time went (current stack, before this work)

| Leg | Current component | Typical cost | Why |
|---|---|---|---|
| ① | Silero VAD, Pipecat 1.7 default `stop_secs=0.2` | ~200 ms | fixed silence window (fine) |
| ② | AssemblyAI streaming STT | ~300–500 ms | cloud round-trip + vendor turn finalisation |
| ③ | Gemini via LangChain, **thinking left at default** | ~500–1500 ms | Gemini 2.5 "thinking" runs before the first token; ~2.3k tokens of tool schema in every prompt |
| ④ | full-sentence flush to TTS | +100–400 ms | waits for `.`/`?`/`!` before TTS starts |
| ⑤ | **Gemini TTS (preview)**, one request per sentence | ~800–1500 ms (estimate; no vendor TTFB published) | generative TTS model behind a cloud request |
| | **Sum** | **≈ 1.9 – 4.1 s** | |

The two biggest legs are **⑤ cloud generative TTS** and **③ LLM thinking**.
Neither can be fixed by tuning the pipeline alone. They need different
components, which is why the user's instinct to self-host ASR/TTS is right.

### Target budget (after)

| Leg | New component | Budget | Mechanism |
|---|---|---|---|
| ① | Silero VAD, `latency.vad_stop_secs = 0.2` (configurable) | 200 ms | config-driven `VADParams` |
| ② | **LiteRT Moonshine-tiny i8 (EN)** / Whisper-tiny i8 (VI) on CPU — or keep AssemblyAI | measured 250–480 ms (EN, Moonshine) · 1.0–1.5 s (VI, Whisper) on a 4-vCPU container | utterance-level local inference, no network; model pre-loaded + warmed; energy gate skips silence (~1 ms) |
| ③ | Together DeepSeek-V4.1-Flash **reasoning off** (measured) · Gemini 2.5 Flash-Lite thinking 0 · gpt-4.1-nano · **SGLang Qwen2.5-1.5B** local | **measured 325–440 ms warm** (Together, reasoning off; 505 ms with reasoning; **7.5 s cold**) · 30–120 ms local GPU (estimate) | thinking off, `max_tokens` cap, connection pre-warm (warm-up), prompt-prefix caching |
| ④ | **eager first-clause flush** (`tts_first_chunk_min_chars=20`) | ~0–150 ms | flush at the first `, ; :` once ≥ 20 chars |
| ⑤ | **LiteRT KittenTTS-nano on CPU** (EN) · Piper (VI) · OpenAI `gpt-4o-mini-tts` (cloud fallback) | **measured 169 / 233 / 335 ms** to first speech for 10 / 20 / 40-char first clause (was ~1 s: 0.8 s of leading silence, now clamped) | ~32 MB fp16 3-stage model, pre-loaded + warmed |
| | **Sum (measured legs, EN, 4-vCPU container)** | **≈ 200 + 300 + 375 + 233 ≈ 1.1 s** (vs ≈ 2–4 s before) | |

**Honest read.** Every leg is now 2–5× faster than before, and the whole loop
can be measured. On this 4-vCPU cloud container, though, the sum is about
**1.1 s, not 0.6 s**. The remaining gap is about 500 ms, split between local STT
(~300 ms) and cloud LLM TTFT (~375 ms). The levers that close it are in
`02-implementation-and-verification.md` §5. In priority order:
(1) **streaming/incremental STT**: transcribe Moonshine windows *while* the user
speaks, so only the last ≤1 s is left at end-of-speech (≈ −200 ms);
(2) a **local LLM on GPU** via SGLang (TTFT 30–120 ms, ≈ −250 ms), or Gemini
Flash-Lite with thinking 0;
(3) **speculative first clause**: start TTS on a short acknowledgement while
the LLM streams;
(4) a desktop CPU with more cores than this container.

---

## 2. Design principles (unchanged; we extended them)

1. **Provider plugin system.** Every capability is an ABC (`providers/base.py`).
   Backends self-register (`@register("stt","litert")`). Config picks one, and the
   composition root (`app.py`) builds it. New LiteRT, Piper, OpenAI-TTS and
   faster-whisper backends are *just more providers*: no `if/elif` ladders were
   added to callers.
2. **Config is the single source of truth** and also the UI contract. All new
   knobs (`models.*`, `conversation.latency.*`, LLM latency fields,
   `actions.search_engines`, `actions.mcp_servers`) live in the one
   pydantic-settings schema, so `/api/v1/config/schema` and the dashboard
   editor picked them up automatically.
3. **Fail at startup, not mid-conversation.** Capabilities (`self_hosted`,
   `languages`, `latency_class`) are declared as data and validated.
   `privacy.local_only` refuses any provider that could egress.
4. **The LLM proposes, a non-LLM gate disposes.** Every new computer-use tool,
   including every MCP tool from any third-party server, goes through
   `ActionGate`: classify → allow / confirm / refuse → audit.

---

## 3. Component A — Model registry (HuggingFace `litert-community` → disk → runtime)

### Concept
A local "app store" for models: **browse** the Hub org, **pull** the right files,
**load** them into the host process (CPU) or **serve** them to the browser
(WebGPU), then **assign** them to a role (ASR / TTS). Each step is one REST call,
and `POST /models/pull {load:true}` does pull + load in a single request.

### Data flow
```
UI ──GET /api/v1/models/hub──▶ HubClient ──GET /api/models?author=litert-community&full=true──▶ HF
UI ──POST /models/pull──────▶ ModelManager.start_pull ─▶ PullJob (asyncio task, semaphore)
                                   │  for each file: GET /{repo}/resolve/{rev}/{path}
                                   │  stream → <file>.part → os.replace (atomic), Range-resume
                                   ▼
                     .dana/models/<org>__<name>/…   + registry.json (atomic manifest)
UI ──POST /models/load──────▶ MODEL_CACHE.get_or_load(path, cpu, threads)  (ai_edge_litert)
UI ──POST /models/assign────▶ config patch {conversation.stt.provider:"litert", model:repo}
Browser (WebGPU) ──GET /models/file?repo_id&path&token──▶ raw bytes → MediaPipe/LiteRT.js
```

### Key mechanisms
- **Catalog**: one paged call with `full=true` returns `siblings` (file names),
  so formats (`tflite`/`litertlm`/`task`) and "pulled" flags are computed without
  N extra requests. TTL cache (`models.catalog_ttl_s`).
- **Task classification**: `pipeline_tag` first, then keyword fallback on name and
  tags.
- **File recommendation** (important for real repos): litert-community repos ship
  many hardware-specific NPU variants (`*_MediaTek_MT6989.tflite`,
  `*_Qualcomm_SM8650.tflite`, `*_gpu_*`). For a CPU host we exclude those and pick
  the generic `*_i8`/`*_fp16` variant **plus every stage and sidecar** of
  multi-stage models. For example, KittenTTS needs predictor + prosody + vocoder +
  `voices.npz` + `g2p/*`.
- **Companion files**: Whisper `.tflite` files have no tokenizer inside, so the
  registry also pulls `openai/whisper-<size>` tokenizer files next to them.
- **Safety**: file paths are validated (no `..`, no absolute paths, resolved path
  must stay inside the model directory). The HF token is sent only to the hub
  host, never across a CDN redirect, and is never logged or returned
  (`SecretStr`).
- **Gated repos**: 401/403 are mapped to a clear "accept the license on HF and set
  `HF_TOKEN`" error (HTTP 403 to the UI).

---

## 4. Component B — LiteRT runtime and the realtime speech providers

### 4.1 Runtime (`runtime/litert.py`)
- `ai-edge-litert` `Interpreter(model_path, num_threads)`. XNNPACK is the default
  CPU delegate, so INT8/FP16 kernels are used automatically.
- `ModelCache`: a process-wide, thread-safe cache keyed by
  `(path, backend, threads)`. Loading through the registry **pre-warms** the
  same instance the provider later uses, so the first turn pays no load cost.
- `backend="webgpu"` is intentionally *not* a server-side backend. WebGPU lives
  in the browser, so the registry serves model bytes to the UI runtime instead.

### 4.2 ASR: LiteRT Whisper & Moonshine (`providers/stt/litert.py`)

One provider, two strategies chosen by **signature introspection**:
- **Moonshine-tiny** (`moonshine_tiny_5s_i8.tflite`, English): the encoder takes
  *raw* 16 kHz audio `[1, 80000]` (5 s windows). The decoder is
  `[1,64]` tokens + additive mask. Encode ≈ 30 ms, ≈ 21 ms per decode step, and
  **250–480 ms** for a short command. The SentencePiece tokenizer is pulled from
  `UsefulSensors/moonshine-tiny` as a companion.
- **Whisper-tiny** (`whisper_tiny_30s_i8.tflite`, multilingual incl. Vietnamese):
  described below. Its decode graph has a **static** `[1,128]` token window, so
  `resize_tensor_input` is impossible. Every step costs ≈ 65 ms regardless of
  prefix length, so a short Vietnamese command takes ≈ 1–1.5 s.
- **Energy gate**: an utterance below −45 dBFS skips inference (≈ 1 ms). This
  also removes Whisper's classic "you" hallucination on silence.
`litert-community/whisper-tiny` → `whisper_tiny_30s_i8.tflite` (41 MB, INT8) exposes
two signatures:

| signature | inputs | outputs |
|---|---|---|
| `encode` | log-mel `[1, 80, 3000]` f32 | encoder states `[1, 1500, 384]` |
| `decode` | encoder states, token ids, additive causal mask | vocabulary logits |

Mechanism:
1. **Front-end** (`runtime/audio.py`, numpy): 16 kHz mono, 400-point Hann STFT,
   10 ms hop, 80-bin Slaney power mel, `log10(max(x,1e-10))`, dynamic-range
   clamp `max(x, max−8)`, `(x+4)/4`, zero-pad to 3000 frames (30 s).
2. **Encode once** per utterance.
3. **Greedy stateless decode**: seed `[<|startoftranscript|>=50258, <|lang|>,
   <|transcribe|>=50359, <|notimestamps|>=50363]` where `<|en|>=50259` and
   `<|vi|>=50278` (DaNa's language config picks the token). Re-run the decoder on
   the whole prefix each step (the graph is stateless), take the argmax, and stop
   at `<|endoftext|>=50257` or on trailing repetition.
4. **Detokenise** with byte-level BPE (`vocab.json` + `added_tokens.json` from
   `openai/whisper-tiny`).

Pipeline integration: Whisper is utterance-level, not streaming. A new
`ProviderSTTProcessor` buffers mic frames between `VADUserStartedSpeaking` and
`VADUserStoppedSpeaking` (with ~300 ms pre-roll so the first phoneme is not
clipped), runs the provider in a worker thread, and emits a `TranscriptionFrame`.
On CPU, a 30 s-padded tiny encoder plus a short decode costs well under a cloud
round-trip (see §7 for the measurement).

Alternatives in the registry: `moonshine-tiny` (5 s window, English-only,
faster), `parakeet-*`, `Qwen3-ASR-0.6B` (Vietnamese tagged), and the
`faster_whisper` provider as a non-LiteRT CPU fallback.

### 4.3 TTS: LiteRT KittenTTS (`providers/tts/litert.py`)
`litert-community/kitten-tts-nano-0.8` (StyleTTS2/iSTFTNet family, 24 kHz):
1. **G2P**: text → IPA phonemes. Uses the bundled neural G2P
   (`g2p/dp_g2p_matcha_fp16.tflite` + dictionary), so no espeak-ng is needed on
   Windows. `phonemizer`/espeak is the fallback.
2. **Tokenise** phonemes → symbol ids.
3. **Style vector** = `voices.npz[voice][min(len(text), N−1)]`.
4. **predictor → prosody → vocoder** (3 fp16 `.tflite` stages) → float waveform
   → int16 PCM, chunked at ~200 ms.
Because `ProviderTTSProcessor` hands the provider one clause at a time, the first
audio is produced after the **first clause**, not the whole reply.

**Measured tuning.** Profiling showed the vocoder dominates compute
(≈ 60 ms per second of audio). The real problem, though, was **perceptual**:
the BOS token's predicted duration produced ~0.8 s of leading silence.
Clamping the first token to 3 frames (envelope correlation 0.97–0.98 with the
unclamped output) and trimming tail silence to 100 ms cut time-to-first-speech
from ~1 s to **169 / 233 / 335 ms** for 10 / 20 / 40-char clauses. Two other
ideas were measured and **rejected** because they degraded audio (correlation
0.45–0.79): padding inputs to fixed shape buckets, and windowed vocoding.
4 threads (= physical cores) was best.

Vietnamese: no Vietnamese LiteRT TTS exists in the org today, so `language=vi`
routes to the **Piper** provider (`vi_VN` voices, CPU ONNX). OpenAI
`gpt-4o-mini-tts` (PCM streaming) is the low-latency cloud fallback for both
languages.

---

## 5. Component C — The pipeline's own latency mechanics

1. **VAD from config**: `SileroVADAnalyzer(params=VADParams(stop_secs=latency.vad_stop_secs,
   …))`. Every 100 ms of silence window is 100 ms of latency.
2. **Eager first chunk**: for the first chunk of each reply, flush to TTS at the
   first clause boundary (`, ; : — –`, which must be followed by whitespace so
   "3,5" is never split) once the buffer holds ≥ `tts_first_chunk_min_chars`, or at
   any sentence end. Later chunks keep `chunk_sentences` batching.
3. **Warm-up at session start**: load local models and run one dummy inference;
   send one tiny LLM call so TLS/HTTP2 connections and server-side prompt caches
   are hot. This is bounded by a timeout, and failures are logged, not fatal.
4. **Speculative STT** (opt-in): after 60 ms of low energy inside an utterance,
   transcription of the buffered audio starts in the background. VAD's 200 ms
   stop window then overlaps STT instead of preceding it, and the run is
   cancelled if speech resumes. Measured −240 ms on real Moonshine.
5. **Instant acknowledgement** (opt-in): a cached "Mm-hm." / "Dạ." plays at
   end-of-speech (≈ 220 ms perceived). It is tracked separately as
   `ack_first_audio`.
6. **Connection keep-warm** (opt-in): a token-free `models.list()` ping keeps the
   LLM's HTTP connection alive between turns.
7. **Per-turn clock**: `TurnClock` stamps end-of-speech, STT final, LLM first
   token and TTS first audio. It emits `latency` events (`stt`, `llm_ttft`,
   `tts_ttfb`, `voice_to_first_audio` + `over_budget`), which flow to the UI's
   latency waterfall over the existing log→WebSocket bridge.

---

## 6. Component D — LLM hosting: SGLang (self-hosted) + OpenAI/Gemini (primary)

- **One code path.** SGLang, vLLM, Ollama and LM Studio are OpenAI-compatible, so
  they are aliases of the `openai` chat model with a default `base_url`
  (`sglang → http://127.0.0.1:30000/v1`). An `EMPTY` placeholder key is used when
  none is set.
- **Latency fields** map per provider: `thinking_budget` → Gemini only;
  `reasoning_effort` → OpenAI reasoning models only; `max_tokens`, `timeout_s`
  and `extra` are passed through.
- **Together AI** (`provider=together`): a hosted OpenAI-compatible cloud for
  open models. It reads `TG_API_KEY`/`TOGETHER_API_KEY`, `TG_BASE_URL` and
  `TG_MODEL_ID`, and it is **not** classed as self-hosted, so `local_only` still
  refuses it. Measured live on `deepseek-ai/DeepSeek-V4.1-Flash`: the cold first
  call took **7.5 s**; warm TTFT was **~505 ms** with hidden reasoning and
  **~375 ms** with `extra_body.reasoning.enabled=false` (the preset's default).
  `reasoning_effort="minimal"` is rejected by this model (it accepts
  low/high/xhigh/max). Env values are `.strip()`ped, because the provided file had
  CRLF line endings that broke the URL.
- **Presets** (`GET /api/v1/llm/presets`) are curated one-click config patches.
  **Probe** (`POST /api/v1/llm/probe`) measures real TTFT and total time against
  the live endpoint. **Health** checks an SGLang server's `/v1/models`.
- **Serving SGLang**: `scripts/serve_sglang.sh` (Linux/WSL2 + NVIDIA). SGLang has
  no native Windows support; its CPU path targets Intel Xeon AMX servers, not
  laptops. For a CPU-only laptop, the realistic "self-hosted LLM" is a LiteRT LLM
  in the **browser on WebGPU** (Gemma 3 1B / Qwen `.litertlm`/`.task` via
  MediaPipe), or a cloud LLM.

---

## 7. Component E — Generic computer use (no more hard-coded sites)

### Before
Per-site adapters (`adapters.py`) held YouTube/Google/arXiv URL templates and
CSS selectors, and `browser_search` took a fixed engine enum. Any new site meant
new code.

### After: three layers, all behind the same gate
1. **Observe → act-by-reference (any website).** `browser_snapshot` runs an
   in-page script that walks the DOM (and open shadow roots), keeps visible
   interactive and landmark elements, computes role + accessible name + state,
   stamps `data-dana-ref="eN"`, and returns a compact numbered list (≤150 items,
   wrapped as *untrusted page data*). `browser_act(ref, action, text)` then clicks,
   fills, selects or checks *by ref*. This is the pattern used by Playwright MCP
   and browser-use: the model reasons over a semantic map, not pixels or brittle
   selectors.
   - The **gate classifies from the snapshot map**, not the page:
     hover/focus = read-only, fill/select = reversible, a click whose accessible
     name matches destructive or purchase vocabulary = destructive (confirm), and
     a link to a non-allowlisted host = out-of-scope (refuse).
2. **Configurable search** (`actions.search_engines`): user URL templates such as
   `{"shopee": "https://shopee.vn/search?keyword={q}"}`, plus *generic site
   search* on any allowlisted domain, which finds the page's search box via the
   same semantic heuristics.
3. **Desktop apps via Windows UI Automation**: `os_ui_snapshot` / `os_ui_act`
   apply the same ref pattern to any native app's accessibility tree
   (pywinauto/UIA), gated by `os_actions_enabled`, the app allowlist, and
   destructive-name detection.
4. **MCP bridge (plug-and-play standard)**: any app with a Model Context Protocol
   server (Playwright MCP, filesystem, Spotify, Office, internal tools…) is
   declared in `actions.mcp_servers`. Its tools are discovered at startup,
   namespaced `mcp_<server>_<tool>`, and **classified from MCP annotations**
   (`readOnlyHint` → read-only, `destructiveHint=false` → reversible, the MCP
   default → destructive = confirm). `trust="confirm_all"` and an optional
   `allowed_tools` list are available. **Default trust is `confirm_all`**: every
   MCP call needs confirmation unless the user opts a trusted server into
   `trust="annotations"`. Otherwise a hostile server could label a destructive
   tool read-only (security review finding).

Vision-based computer-use models (Gemini/OpenAI/Anthropic CUA) are the planned
fallback only when there is no accessibility tree (canvas apps, games). They are
slower and costlier, and they cannot meet the voice latency budget.

---

## 8. Component F — UI (Models tab) and in-browser WebGPU

- **🧩 Models tab**: Hub browser (search, task chips, gated/pulled badges, file
  drawer with recommended files pre-checked, CPU/WebGPU + threads selector,
  *Pull* / *Pull & Load*), live pull jobs with progress, local models (load,
  unload, delete, *Use as ASR/TTS*), LLM hosting presets with *Check server* and
  *Measure latency*, and a visual 600 ms latency budget fed by live
  `latency_span` events.
- **WebGPU runner**: detects `navigator.gpu`, fetches pulled `.task`/`.litertlm`
  bytes from the agent (`/api/v1/models/file`, bearer-authenticated), and runs
  them with MediaPipe `LlmInference` on WebGPU, measuring TTFT in the browser.
- **PWA manifest + responsive CSS**: the Talk view is installable on a phone. This
  is the Phase-1 mobile client (see `mobile-app-plan.md`).

---

## 9. Security and privacy notes

- **HF token**: read from the `HF_TOKEN` env/`.env` (gitignored) or the OS keyring
  into `models.hf_token: SecretStr`. It is never written to tracked files, logs or
  API responses. **The token pasted in chat should be rotated** at
  huggingface.co/settings/tokens, because it now sits in the conversation
  history.
- Model downloads are untrusted data. Path traversal is blocked, and executables
  are never pulled by default (only `tflite/litertlm/task/json/txt/model/npz/bin/gz/md`).
- `/api/v1/models/file` accepts `?token=` (browsers cannot set headers for some
  loaders), mirroring the existing WebSocket auth. The API stays loopback-only.
- The new computer-use and MCP tools never bypass `ActionGate`. Every decision is
  audited in `.dana/audit.jsonl`.
