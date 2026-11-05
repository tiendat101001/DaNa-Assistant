# LiteRT ecosystem research for DaNa real-time ASR/TTS (R1)

Date: 2026-11-05. Author: Research Agent R1.

## Executive summary

> **⚠ Owner correction (2026-11-05, after huggingface.co was unblocked):** items 1–3
> below were written blind and are **superseded**. The live HF API lists **443**
> litert-community repos including **26 ASR** (whisper-tiny/base/medium/large-v3(-turbo),
> moonshine-tiny, parakeet-*, Qwen3-ASR, SenseVoice, …) and **13 TTS** repos
> (Kokoro-82M, kitten-tts-nano-0.8, Matcha-TTS, Qwen3-TTS-12Hz-0.6B-Base, sopro-v2-turbo,
> Audio8-TTS, …). `ai-edge-litert` 2.3.0 ships `win_amd64` wheels for cp310–cp312+
> (verified on PyPI). See `litert-community-catalog.md` for the verified snapshot.


1. **Access limit:** `huggingface.co` did not resolve from the WebFetch tool in this session (`getaddrinfo ENOTFOUND`). Hub facts below come from WebSearch snippets and third-party pages, so most HF-specific claims are UNVERIFIED. Allow `huggingface.co` in the session network settings and re-run section 1 and section 2.
2. **litert-community ASR:** The only litert-community speech repo I could confirm is `litert-community/whisper-tiny` (Apache-2.0, LiteRT tag). A PR adding 10s/15s variants was opened and closed. I did not confirm a `litert-community/moonshine-*` repo. A third-party repo points to one that may not exist (UNVERIFIED).
3. **litert-community TTS:** None found. No LiteRT TTS model was located in the org.
4. **litert-community LLMs:** Org listing (search snapshot, possibly stale) includes Gemma3-1B-IT, gemma-3-270m-it, Gemma2-2B-IT, Gemma3-4B-IT, Gemma3-12B-IT, Gemma3-27B-IT, MedGemma-27B-IT, embeddinggemma-300m, plus Qwen, Phi-4-mini, and SmolVLM. Exact repo ids and sizes are UNVERIFIED.
5. **Python runtime:** `ai-edge-litert` is the standalone package. I could not confirm its Python `Interpreter` signature. Search returned only TensorFlow `tf.lite.Interpreter` docs (`num_threads`, `experimental_delegates`), which are probably the same API but UNVERIFIED for `ai_edge_litert`.
6. **LiteRT-LM Python:** Official docs exist (`litert_lm.Engine`, `audio_backend`, `send_message_async`). Platforms listed are Linux and macOS; Windows is pending. This is a blocker for a Windows-first DaNa.
7. **MediaPipe LLM Inference is deprecated for the web.** The official guide says it is in maintenance-only mode and recommends migrating to the LiteRT-LM JavaScript API.
8. **@litertjs/core (browser):** `loadLiteRt(wasmPath)` then `loadAndCompile(model, { accelerator: 'webgpu' })`, and inputs via `moveTo('webgpu')`. Versions are inconsistent across sources (0.1.1 vs 2.0.0). Verify against the npm README.
9. **Latency:** Moonshine's vendor benchmark (CPU, Linux x86): Moonshine Tiny Streaming 69 ms vs Whisper Tiny 1,141 ms for a short-phrase test. This is vendor-reported and not per 3 s utterance. A 3 s utterance number is UNVERIFIED.
10. **Vietnamese:** Piper has vi_VN voices (`vnhellboy/vietnamese-piper-voices`, 3 voices; `quangdung/Piper_checkpoint`, 14 voices, gated). Kokoro Vietnamese support is UNVERIFIED (no results). Whisper and Moonshine Vietnamese support is UNVERIFIED for the LiteRT ports (Whisper is multilingual by design; Moonshine v1 is English-only per my prior knowledge, UNVERIFIED here).

## Recommendation table (role -> model -> format -> runtime -> expected latency -> languages)

| Role | Model | Format | Runtime | Expected latency | Languages |
|---|---|---|---|---|---|
| ASR (CPU, Python, Windows) | `litert-community/whisper-tiny` (or Moonshine Tiny Streaming via UsefulSensors, non-LiteRT) | .tflite | `ai-edge-litert` Interpreter, XNNPACK CPU | Whisper Tiny ~1.1 s/phrase on Linux x86 (vendor-reported comparison); Moonshine Tiny Streaming ~69 ms (vendor) | Whisper: multilingual incl. vi (UNVERIFIED for this port); Moonshine: verify vi |
| ASR (browser, WebGPU) | Whisper via LiteRT.js | .tflite | `@litertjs/core`, `accelerator: 'webgpu'` | UNVERIFIED (one source: 5.6x speedup WebGPU vs CPU for ASR) | Depends on model |
| LLM (CPU/GPU, Python) | Gemma 3 1B IT or Qwen 2.5/3 0.6-1.5B (LiteRT-LM variant) | .litertlm | LiteRT-LM Python (`litert_lm.Engine`) | UNVERIFIED | en + vi (Gemma 3 multilingual; vi quality UNVERIFIED) |
| LLM (browser) | Gemma 3 1B / Gemma 3n | .task or .litertlm | MediaPipe `@mediapipe/tasks-genai` (deprecated) or LiteRT-LM JS | UNVERIFIED | en (vi UNVERIFIED) |
| TTS (CPU, Python, Windows) | Piper vi_VN (`vnhellboy/vietnamese-piper-voices`) or en Piper voice | .onnx (not LiteRT) | piper / sherpa-onnx | UNVERIFIED (Piper is typically real-time on CPU) | vi + en via separate voices |
| TTS (LiteRT) | none found | n/a | n/a | n/a | n/a |

Note: a LiteRT TTS path does not exist in what I could find. For TTS, Piper (ONNX) or Kokoro via onnxruntime is the practical option. This does not meet the "LiteRT for TTS" goal, so flag it to the team.

## 1. litert-community on Hugging Face

Status: **HF unreachable from WebFetch** (`getaddrinfo ENOTFOUND huggingface.co`). Results below are from WebSearch snippets.

| Repo id | Task | Format | Size | Notes |
|---|---|---|---|---|
| `litert-community/whisper-tiny` | ASR (presumed) | .tflite | UNVERIFIED | Apache-2.0, LiteRT tag. PR for 10s/15s variants opened then closed. |
| `litert-community/moonshine-tiny` | ASR | .tflite | UNVERIFIED | Mentioned by a third-party repo; NOT confirmed to exist. |
| `litert-community/models` (org index) | n/a | n/a | n/a | Search snippet listed "18 models" and mostly May-June updates; snapshot may be stale. |
| `litert-community/Gemma3-1B-IT` | LLM | .litertlm/.task (UNVERIFIED) | UNVERIFIED | Listed in org snapshot. |
| `litert-community/gemma-3-270m-it` | LLM | UNVERIFIED | UNVERIFIED | Listed. |
| `litert-community/Gemma2-2B-IT` | LLM | UNVERIFIED | UNVERIFIED | Listed. |
| `litert-community/Gemma3-4B-IT` | LLM (multimodal) | UNVERIFIED | UNVERIFIED | Listed. |
| `litert-community/Gemma3-12B-IT` | LLM | UNVERIFIED | UNVERIFIED | Listed. |
| `litert-community/Gemma3-27B-IT` | LLM | UNVERIFIED | UNVERIFIED | Listed. Too large for CPU. |
| `litert-community/MedGemma-27B-IT` | LLM | UNVERIFIED | UNVERIFIED | Listed, not relevant. |
| `litert-community/embeddinggemma-300m` | embeddings | UNVERIFIED | UNVERIFIED | Listed, not relevant. |
| `litert-community/datasets` | datasets | n/a | n/a | Exists per search. |
| Qwen, Phi-4-mini, SmolVLM repos | LLM/VLM | UNVERIFIED | UNVERIFIED | Listed in snapshot; exact ids unknown. |

Third-party (not litert-community, do not treat as official):
- `devendradhakad/autodroid-litert-community-moonshine-tiny` (float32, int8, NPU-compiled .tflite; points at a `litert-community/moonshine-tiny` source).
- `AndStarTut/whisper-tflite-models` (no model card).
- Upstream `UsefulSensors/moonshine` has `tflite/tiny` float files (https://huggingface.co/UsefulSensors/moonshine/tree/main/tflite/tiny). Not verified in-session.

Speech models with no confirmed LiteRT port: Gemma 3n audio (only via `.litertlm` in MediaPipe/LiteRT-LM examples, UNVERIFIED for ASR use), Moonshine (only third-party). TTS: none.

### 1b. HF API shape (UNVERIFIED, from memory and third-party code; confirm with live calls)

- `GET https://huggingface.co/api/models?author=litert-community&limit=500&full=true`. Fields expected: `id`, `pipeline_tag`, `tags`, `downloads`, `likes`, `lastModified`, `gated`, `private`, `siblings[].rfilename` (search listings may omit `siblings`; the single-model endpoint is the reliable source; UNVERIFIED).
- `GET https://huggingface.co/api/models/{repo_id}` returns `siblings` manifest.
- `GET https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true` returns file entries with `path`, `size`, and `lfs` (object with `size`, `sha256`) for LFS files. Use `lfs.size` when present, else `size`.
- Download: `https://huggingface.co/{repo_id}/resolve/{revision}/{path}`.
- Auth: `Authorization: Bearer hf_...` header.
- Gated: a gated repo returns 401 or 403 without an accepted-license token (UNVERIFIED). One source says `gated` is a string (`"manual"`) on gated repos and a boolean on open ones, so code defensively: `bool(gated)` is wrong for `"false"`; check `gated not in (False, None)`.
- Pagination: `Link: <...>; rel="next"` header, GitHub-style. One source reports a limit cap around 1000 rows per page (UNVERIFIED).

## 2. Python LiteRT runtime

- **Package:** `ai-edge-litert` (PyPI). Import path `ai_edge_litert.interpreter` (per task brief; module path UNVERIFIED in-session).
- **Interpreter args (from TensorFlow docs, probably the same API):** `model_path`, `num_threads` (0/1 disables multithreading; -1 implementation-defined; default implementation-dependent), `experimental_delegates` (list from `load_delegate`, experimental), `experimental_disable_delegate_clustering`, `experimental_default_delegate_latest_features`. Source: https://ai.google.dev/edge/api/tflite/python/tf/lite/Interpreter (TF docs, not the ai-edge-litert docs).
- **Signatures:** `get_signature_list()` and `get_signature_runner(name)` (from TF docs, UNVERIFIED for ai-edge-litert).
- **XNNPACK:** default CPU delegate in TFLite/LiteRT (UNVERIFIED for Python package specifics).
- **Platforms:** Windows x86_64 support is UNVERIFIED. Check PyPI wheels for `win_amd64` and Python 3.11-3.13 before committing.

### LiteRT-LM (.litertlm)

- Python API docs exist: https://ai.google.dev/edge/litert-lm/python
- Install: PyPI package `litert-lm-api-nightly` (nightly); stable name UNVERIFIED.
- Platform: **Linux and macOS**; Windows pending (per docs snippet). This is a blocker for Windows-first DaNa.
- Usage shape: `litert_lm.Engine(model_path, audio_backend=..., vision_backend=...)`; messages with `content` list (audio file path + text); response `response["content"][0]["text"]`; `send_message_async` for streaming. Speculative decoding via `enable_speculative_decoding=True`. Tool calling supported.
- Gemma 3n audio transcription through LiteRT-LM: UNVERIFIED (docs example describes audio, not transcription).

### MediaPipe LLM Inference (.task)

- Python: MediaPipe tasks (UNVERIFIED in detail for this session).
- Web: maintenance-only, migrate to LiteRT-LM JS. So do not start new work on it.

## 3. Browser WebGPU

### (a) LiteRT.js, `@litertjs/core`

Pattern (from third-party guides and the package README snippet; UNVERIFIED against current API):

```js
import { loadLiteRt, loadAndCompile } from '@litertjs/core';

await loadLiteRt('/wasm/');                        // host the wasm files at this path
const model = await loadAndCompile('/models/model.tflite', { accelerator: 'webgpu' });

const input = new Float32Array(/* ... */);
const inputTensor = new Tensor(input, [1, N]).moveTo('webgpu');   // UNVERIFIED constructor
const [out] = model.run([inputTensor]);
const result = await out.moveTo('wasm').toTypedArray(); // UNVERIFIED method names
```

- `loadLiteRt(wasmPath)`: point it at a directory that serves the LiteRT wasm files. Must be same-origin or CORS-allowed. Exact file names UNVERIFIED.
- `loadAndCompile(modelUrl, { accelerator: 'webgpu' | 'wasm' })`: no automatic fallback. Catch the error and recompile with `'wasm'` in app code.
- Partial delegation: one source says the whole graph must run on one backend, so one unsupported WebGPU op fails compile (single source, UNVERIFIED).
- Version: 0.1.1 (Socket page) vs 2.0.0 (pkgstats). Pin and verify.
- Speedup claim (one source): WebGPU vs CPU 5.6x for ASR, up to 30.8x for segmentation. UNVERIFIED.

### (b) MediaPipe `@mediapipe/tasks-genai` (deprecated for web, still documented)

```js
import { FilesetResolver, LlmInference } from '@mediapipe/tasks-genai';

const genai = await FilesetResolver.forGenAiTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm');
const llm = await LlmInference.createFromOptions(genai, {
  baseOptions: { modelAssetPath: '/models/gemma-3n-E2B-it-int4-Web.litertlm' },
  maxTokens: 1024,
  topK: 40,
  temperature: 0.8,
  randomSeed: 101,
});

// streaming
llm.generateResponse('prompt', (partial, done) => { /* append partial */ });

// or blocking
const text = await llm.generateResponse('prompt');
```

- Model files: `.task` and `.litertlm` (per guide). Gemma 3 1B and Gemma 3n examples named in guide (exact filenames UNVERIFIED).
- Requirements: WebGPU-capable browser.
- Gemma 3n multimodal options: `maxNumImages`, `supportAudio`; `generateResponse` takes an ordered array of strings and media sources (UNVERIFIED details).
- LoRA: `loadLoraModel()` then pass the LoRA reference to `generateResponse` (UNVERIFIED).
- Status: maintenance-only. Plan to migrate.

## 4. CPU latency numbers

Vendor-reported (Moonshine README, aggregator copies; CPU-only; `two_cities.wav` split by VAD into phrases; Whisper via faster-whisper). These are per-phrase numbers, not strictly per 3 s utterance.

| Model | MacBook Pro (M-series) | Linux x86 | Raspberry Pi 5 | WER |
|---|---|---|---|---|
| Moonshine Tiny Streaming (34M) | 34 ms | 69 ms | 237 ms | 12.00% |
| Whisper Tiny (39M) | 277 ms | 1,141 ms | 5,863 ms | 12.81% |
| Moonshine Small Streaming | 73 ms | 165 ms | 527 ms | n/a |
| Whisper Small | 1,940 ms | 3,425 ms | 10,397 ms | n/a |

- Non-streaming Moonshine Tiny latency: not found (UNVERIFIED).
- Whisper-base, Moonshine-base: UNVERIFIED.
- Expected LiteRT (`ai-edge-litert`, XNNPACK) latency for a 3 s utterance on a Windows laptop CPU: **UNVERIFIED**. Measure it in-repo before deciding. The 600 ms end-of-speech-to-first-audio budget must cover ASR + LLM first token + TTS first chunk, so ASR should stay well under ~150-200 ms.
- Vietnamese: Moonshine v1 is believed English-only (UNVERIFIED); Whisper is multilingual (vi supported in OpenAI weights).

### TTS

- **Piper:** ONNX (not LiteRT). vi_VN voices exist: `vnhellboy/vietnamese-piper-voices` (3 medium voices, sherpa-onnx compatible; requires espeak-ng-data with `vi`). `quangdung/Piper_checkpoint` (14 voices, MIT per listing, gated). Latency: UNVERIFIED.
- **Kokoro-82M:** Vietnamese support UNVERIFIED. Check the model card before relying on it.
- **VieNeu-TTS-v2:** Vietnamese-first, CPU-only, 7 preset voices (North/South). Source not verified beyond a search snippet. Latency UNVERIFIED.
- **LiteRT TTS:** none found.

## 5. Recommendations and next steps

1. Unblock `huggingface.co` in session network settings, then re-run: `GET https://huggingface.co/api/models?author=litert-community&full=true&limit=500` and `GET https://huggingface.co/api/models/litert-community/whisper-tiny`. Record exact ids, sizes, and file formats.
2. Prototype ASR with `ai-edge-litert` + Whisper-tiny `.tflite` on the Windows target. Measure the 3 s utterance latency. Compare with Moonshine Tiny (ONNX via sherpa-onnx, if LiteRT port not confirmed).
3. Check whether `ai-edge-litert` has a `win_amd64` wheel for the project's Python version (3.11+).
4. LLM: use `.litertlm` via LiteRT-LM Python only if Windows is supported or a Linux/WSL path is acceptable; otherwise keep the existing cloud or Ollama path.
5. TTS: keep Piper (ONNX) for vi and en; a LiteRT TTS route is not available today.
6. Browser: do not adopt the deprecated MediaPipe web LLM; evaluate LiteRT-LM JS first.

## Verification status

| Claim | Status |
|---|---|
| litert-community/whisper-tiny exists | Confirmed by search snippet, not by fetching the page |
| Other litert-community repo ids and sizes | UNVERIFIED |
| litert-community moonshine repo | UNVERIFIED (not found) |
| HF API field names and pagination | UNVERIFIED (third-party code only) |
| ai-edge-litert Interpreter signature | UNVERIFIED (TF docs used as proxy) |
| LiteRT-LM Python platforms (Linux/macOS, Windows pending) | Confirmed by docs snippet |
| MediaPipe web LLM deprecation | Confirmed by docs snippet |
| @litertjs/core API calls | Partly confirmed (third-party guides); verify against npm README |
| @mediapipe/tasks-genai API | Confirmed by docs snippet |
| Moonshine/Whisper CPU latency | Vendor-reported, aggregator copies |
| Piper vi_VN voices | Confirmed by HF search snippets (HF pages not fetched) |
| Kokoro Vietnamese | UNVERIFIED |

Sources consulted (WebSearch): https://ai.google.dev/edge/litert-lm/python ; https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/web_js ; https://ai.google.dev/edge/api/tflite/python/tf/lite/Interpreter ; https://github.com/moonshine-ai/moonshine ; https://www.mintlify.com/moonshine-ai/moonshine/models/benchmarks ; https://huggingface.co/vnhellboy/vietnamese-piper-voices ; https://huggingface.co/litert-community/models (search snippet only).
