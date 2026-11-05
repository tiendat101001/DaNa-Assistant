# Market case studies: how production voice agents reach low latency

Author: Research Agent R3 (technical case studies). Tracing lead: Lead C.
Date: 2026-11-05. Method: web search and fetch only. No code, git or uv was touched.
Labels: "UNVERIFIED" = vendor claim, secondary source, forum post, or a number
derived by us rather than measured. Figures from search snippets were not
re-fetched. The Daily blog fetch failed (host not resolvable in this session).

## Executive summary

1. The industry's latency gap is mostly in three places: the end-of-turn wait
   (fixed VAD silence), the LLM network round-trip, and STT finalisation. TTS is
   already cheap (DaNa's Kitten at 138 ms is in line with Cartesia's claims).
2. Semantic turn detection is cheap on CPU. Pipecat Smart Turn v3 reports
   ~10-95 ms per decision on CPU (vendor, UNVERIFIED on our 4-vCPU box). It can
   replace part of DaNa's fixed 200 ms `stop_secs` wait.
3. Speculative LLM work on stable partial transcripts is the best-supported
   lever that keeps the cloud LLM. Estimates run 100-200 ms (vendor and blog,
   UNVERIFIED). It costs 50-70% more LLM calls (Deepgram's claim).
4. Speech-native LLMs (Ultravox, Moshi, gpt-realtime, Gemini Live) remove the STT
   leg, but the published numbers are on A100-class GPUs or hosted APIs with
   1.2-2.3 s time-to-first-audio. None fits DaNa's CPU-only, Vietnamese-first
   constraints this sprint.
5. Colocated inference is the one lever that removes the ~200 ms network leg.
   It needs a GPU. DaNa already has SGLang presets, so a local GPU is the
   cleanest path under 600 ms.
6. On-device audio LLMs (Gemma 3n) have no published audio time-to-first-token.
   The text prefill rates on CPU (E2B: 111 tok/s, Google, secondary source) rule
   out using 3n as the tool-calling LLM on CPU. Its audio token rate (6.25/s) is
   cheap, so it adds little beyond STT.
7. Measured DaNa floor (02 §5): VAD 200 + Moonshine 423 + LLM 426 + Kitten 138
   ≈ 1,187 ms. Reaching < 600 ms needs streaming STT (no full-utterance decode)
   and either overlap (speculative LLM) or a colocated LLM. Most published
   vendor numbers for the same stack sit at 0.5-0.6 s only under ideal conditions.
8. Recommendation for the next sprint: (1) Smart Turn, (2) speculative LLM on
   stable partials, (3) prompt-prefix caching and tool-schema trimming, (4) a
   local GPU LLM, (5) per-leg p50/p90 instrumentation and a real-voice test
   before and after each change.

## 1. Technique table

ms saved = estimated reduction on DaNa's critical path. Effort is for DaNa.
"Fit" means fit for DaNa's current stack (Pipecat, CPU, LiteRT, Together/cloud LLM).

| # | Technique (case study) | Mechanism | ms saved (source) | Effort | Prereqs | Fit for DaNa | Hardware |
|---|---|---|---|---|---|---|---|
| T1 | Semantic turn detection, Pipecat Smart Turn v3 (Daily/Pipecat) | Audio classifier says "user finished" on acoustic and prosodic cues, then ends the turn early instead of waiting for fixed silence | Replaces the fixed 200 ms `stop_secs`. Derived estimate ~100-150 ms net of ~10-95 ms inference: UNVERIFIED | S-M (Pipecat analyzer, ONNX model) | VAD still gates the classifier; audio context window | High. DaNa already uses Pipecat | CPU is enough: 12 ms on modern CPUs, ~60 ms on c8g.medium, ~95 ms on t3.medium (Daily, UNVERIFIED for our box) |
| T2 | Semantic turn detection, LiveKit turn detector (LiveKit docs) | Text model over the transcript plus VAD. Full model is hosted. Lite v1-mini runs locally | No official ms found. VAD-only mode is the documented lowest-latency option | S | LiveKit agents framework | Low. DaNa is on Pipecat; the docs say the full model is hosted | Local mini model on CPU (UNVERIFIED timing) |
| T3 | Eager end-of-turn with speculative LLM prefill (Deepgram Flux) | Fire the LLM on an "eager" end-of-turn signal, cancel if the user keeps talking | 100-200 ms off the critical path (Deepgram guide, UNVERIFIED). Flux end-of-turn detection itself ~260 ms P50 (Deepgram docs) | M | Streaming STT that emits eager EOT (Flux). Costs 50-70% more LLM calls (Deepgram claim) | Medium. Needs a Flux-class STT; Moonshine/Kitten path does not emit it | Cloud STT. Flux claims reduce response latency 200-600 ms vs pipeline (vendor, UNVERIFIED) |
| T4 | Speculative LLM on stable partial transcripts (CallSphere guide; RelayS2S paper) | Start the LLM on partial STT text once it is stable. Cancel and restart if the final transcript differs | 200-400 ms if the transcript matches (CallSphere estimate, UNVERIFIED). RelayS2S paper: P90 first chunk 1,006 → 81 ms, avg −479 ms on real voice, different architecture (arXiv 2603.23346) | M | Partial-result STT (Moonshine streaming or chunked Whisper); cancellable LLM calls; confidence threshold (one guide suggests ≥ 0.85, UNVERIFIED) | High. Biggest lever that keeps the cloud LLM. Watch PII in cancelled prefetches | Any; cost is API calls |
| T5 | Endpoint anticipation (Kyutai Unmute paper, arXiv 2606.13450) | Predict the end of the turn before silence is complete | −505 ms average (paper; units and conditions unclear) | L (research-grade) | Unmute-style STT with semantic VAD | Low this sprint | GPU server |
| T6 | Flush trick (Kyutai Unmute) | Process the buffered audio faster than real time once end-of-turn is predicted | Part of the 500 ms STT delay (Kyutai docs, not quantified) | M | STT server that accepts a flush command | Low; Moonshine streams differently | Server |
| T7 | Speech-native LLM, Ultravox (Fixie/HF) | Audio goes straight into the LLM, so the STT leg disappears | Removes Moonshine's 423 ms. Model TTFT ~200 ms with audio input on A100-40GB (model card). E2E first audio 600 ms in Cerebrium's Pipecat deploy (case study) | L-M | A100-class GPU or hosted; VAD still required | Low. Needs GPU; Vietnamese support UNVERIFIED | A100-40GB (model card) |
| T8 | Full-duplex speech-to-speech, gpt-realtime / Gemini Live | Native speech in and out, no cascade | Plainconcepts benchmark (third-party, Azure-hosted): GPT-Realtime 1,201 ms easy / 2,308 ms hard; Gemini 3.1 Live 1,407 ms ±118 (time to first audio) | M (S2S provider already exists in DaNa config) | Cloud key, network | Low now; the cascade is cheaper and keeps local STT/TTS | Cloud only |
| T9 | Full-duplex on-device, Kyutai Moshi | Native duplex model. 160-200 ms claim is third-party, UNVERIFIED. Kyutai gives no official Moshi ms | Unknown | L | GPU | Low | GPU |
| T10 | Sesame CSM | Conversational speech model with turn-taking. No real-time latency numbers found | Unknown | L | GPU | Low | GPU (UNVERIFIED) |
| T11 | TTS, Cartesia Sonic-3 / Sonic Turbo (Cartesia) | Streaming TTS with low time to first audio | Vendor: ~90 ms TTFA (Sonic-3), ~40 ms (Turbo). Measured: 166 ms median incl. network (Vapi benchmark, secondary, UNVERIFIED). Cartesia changelog: ~190 ms median. DaNa Kitten 138 ms (measured, local). Net gain ≈ 0 to −50 ms | S-M | Cloud key | Low. Kitten already beats the cloud number | Cloud |
| T12 | Fast cloud LLM, Groq / Cerebras (Groq, Cerebras) | Custom silicon, low TTFT | Groq TTFT < 80 ms for Llama/Gemma (one source, UNVERIFIED). No Cerebras ms TTFT found. DaNa's Together floor is 370-450 ms | S | Model must be tool-capable and Vietnamese-capable (not checked) | Medium. Benchmark it; the Together sweep in 02 §5.2 did not include Groq/Cerebras | Cloud |
| T13 | Colocated / edge LLM (LiveKit/Pipecat deployment guidance; DaNa SGLang presets) | Move the LLM to the same region or LAN, or a local GPU | Removes most of the ~200 ms network leg (02 §5, derived). Local GPU TTFT: UNVERIFIED | M | GPU or nearby host, SGLang preset (exists) | High. Only path to < 600 ms with a tool-calling LLM | Local GPU, 8-16 GB VRAM class (UNVERIFIED sizing) |
| T14 | Prompt prefix caching + tool-schema trimming | Cache the system and tool prefix; send fewer tools per turn | DaNa measured 426 ms with 26 tools vs 453 ms plain (02 §5.1), so tools cost little. Prefix cache: UNVERIFIED on Together | S | Provider supports prefix caching | High. Cheap to test | Any |
| T15 | Keep the LLM warm (DaNa already does this) | Warm the connection and model at session start | Cold 7.5 s → warm 426 ms (02 §5) | done | none | Already shipped | Any |
| T16 | Instant acknowledgement (DaNa already does this) | Play a short filler while the LLM runs | Perceived first audio ~220 ms (02 §4) | done | none | Already shipped | Any |
| T17 | Silence threshold tuning (industry guides) | Lower the fixed silence window to 400-600 ms | 200-400 ms apparent gain vs 800-1,000 ms defaults (vendor guides, UNVERIFIED). Costs false interruptions | S | None | Covered by T1, which is better | Any |
| T18 | Streaming STT with partials (Moonshine / Whisper chunked) | Decode while speaking, so only the tail is left at end of speech | Moonshine full-utterance 423 ms (measured). Partial-decode saving: UNVERIFIED for our clip length | M | STT that emits partials | High. Enabler for T4 | CPU (measured on 4 vCPU) |
| T19 | On-device audio LLM, Gemma 3n (LiteRT-LM) | Single model for audio and text | No audio TTFT found. Google (secondary): E2B on S24 Ultra CPU 111 prefill / 16 decode tok/s; GPU 816 / 16. M4 MacBook Pro GPU 2,265 / 47.6 (HF card). Audio = 6.25 tokens/s. Gemma 4 E2B on Pixel 8 Pro ~2× realtime (UNVERIFIED, not 3n) | L | Phone or laptop | Low for tool-calling on CPU. Prefill of a 26-tool prompt on CPU is seconds (derived, UNVERIFIED) | Phone CPU/GPU, Mac GPU |
| T20 | Small on-device STT, Moonshine tiny on RPi5 (third party) | Small model on ARM | 148.67 ms for a 5 s clip (third-party, UNVERIFIED for our CPU; our 3.6 s clip is 423 ms) | S | none | Already in use | RPi5 (UNVERIFIED) |

## 2. Per-case notes

### Pipecat (Smart Turn, interruptions, preemptive generation)
- Smart Turn v3 runs on CPU in ONNX. Vendor numbers: 12 ms modern CPU, ~60 ms c8g.medium, ~95 ms t3.medium (Daily blog, search snippet). Repo README: 10 ms on some CPUs, under 100 ms on most cloud instances (github.com/pipecat-ai/smart-turn). Third-party M5 Pro port: 20-36 ms (UNVERIFIED).
- Prereq: VAD (Silero already in DaNa), then Smart Turn decides "complete" on the audio window.
- DaNa adoption: add the local analyzer in the Pipecat stack, lower `stop_secs`, and A/B on real voice. Cost is one more model on CPU.

### LiveKit Agents
- Recommends its turn-detector model. Full model is hosted; a v1-mini runs locally. VAD-only is the lowest-latency option.
- A community thread reports ~1 s wait with preemptive generation and a fixed endpointing delay (single user, UNVERIFIED). No official ms for the turn detector.
- Lesson: preemptive generation only helps if the LLM actually starts before the final commit. Check this in our traces.

### Deepgram Flux / Voice Agent
- Flux end-of-turn ~260 ms (Deepgram docs). p90 1 s, p95 1.5 s (Deepgram benchmarks, UNVERIFIED). Claims 200-600 ms response-latency reduction vs pipeline and ~30% fewer false interruptions (vendor).
- Eager end-of-turn: 100-200 ms saved at the cost of 50-70% more LLM calls (Deepgram guide, UNVERIFIED).

### Kyutai (Unmute, Moshi, STT)
- STT 1B model: words transcribed 500 ms after speech (Kyutai STT page). 2.6B English: 2.5 s delay.
- Semantic VAD predicts end of turn; flush trick accelerates buffered audio.
- Endpoint anticipation paper (arXiv 2606.13450): −505 ms average (units and setup not clear from the snippet).
- Unmute "below a second" response claim is vendor. A third-party deep dive gives 400-750 ms (UNVERIFIED). Moshi 160-200 ms is third-party (UNVERIFIED).

### Ultravox
- Speech-native: skips STT, feeds audio tokens to the LLM. TTFT ~200 ms with audio on A100-40GB (HF model card). Cerebrium case study: 600 ms first-time-to-audio in a Pipecat pipeline (includes VAD and TTS).
- Hardware is the blocker for DaNa (no GPU, CPU-only container). Vietnamese not verified.

### Realtime and Live APIs (gpt-realtime, Gemini Live)
- Only third-party measurement: GPT-Realtime 1,201 ms easy / 2,308 ms hard; Gemini 3.1 Live 1,407 ms ±118 ms (Plainconcepts, UNVERIFIED for independence). Forum reports of multi-second spikes (UNVERIFIED).
- Lesson: no hosted S2S API in the sample beats DaNa's measured 1.19 s by enough to justify losing local Vietnamese and privacy.

### Cartesia (Sonic-3, Line)
- Vendor TTFA ~90 ms. Measured end-to-end 166 ms median via Vapi (UNVERIFIED, secondary). DaNa Kitten 138 ms measured locally.
- Lesson: TTS is not the bottleneck; do not change it for latency.

### Groq / Cerebras
- Groq TTFT < 80 ms for small Llama/Gemma (UNVERIFIED, one source). Cerebras: no ms TTFT found.
- Model quality and Vietnamese support not checked. Worth a 10-minute probe with DaNa's 26-tool prompt, the same way 02 §5.2 swept Together.

### Vapi, Retell AI (platform benchmarks)
- One test: Vapi avg 539 ms, Retell 714 ms (identical agents). Roundup: Vapi 465 ms, Retell 580-620 ms (vendor best case, single region, low load).
- Hamming production data (secondary, UNVERIFIED): real-world P50 1.4-1.7 s, P90 3.3-3.8 s. Lesson: lab numbers and production numbers differ by roughly 2-3×. Measure p90, not only the median.

### Sesame CSM
- Architecture only; no real-time latency numbers found. Not recommended for DaNa.

### Gemma 3n on LiteRT-LM (item e from the brief)
- No published audio time-to-first-token for Gemma 3n on LiteRT-LM. Searches found none.
- Google's LiteRT-LM overview (secondary, Spanish version): E2B on S24 Ultra CPU 111 prefill / 16 decode tok/s; GPU 816 / 16 (1,024 prefill, 256 decode).
- HF model card: M3 MacBook Pro CPU 232.5 / 27.6; M4 MacBook Pro GPU 2,265 / 47.6.
- Audio encoding: 6.25 audio tokens per second per channel. A 3.6 s clip ≈ 23 tokens. The audio itself is cheap.
- Implication: at ~111 tok/s prefill on phone CPU, a 1,000-token tool prompt costs ~9 s of prefill (derived, UNVERIFIED). Gemma 3n cannot be the tool-calling LLM on CPU. It is at most a replacement for Moonshine on a phone with an NPU, and that is not shown.
- Gemma 4 E2B "2× realtime" audio transcription on Pixel 8 Pro CPU (May 2026 report) is UNVERIFIED and not Gemma 3n.

## 3. What is missing for the 600 ms target

Projected budget for DaNa, not measured (all UNVERIFIED):

| Stage | Today (measured) | With T1 + T4 + T18 | With T13 (colocated LLM) |
|---|---|---|---|
| End-of-speech wait | 200 | ~80-100 (Smart Turn) | ~80-100 |
| STT tail | 423 | ~150-250 (streaming partials) | ~150-250 |
| LLM TTFT | 426 | ~250-430 overlapped with STT tail | ~200-250 (no network leg) |
| TTS first chunk | 138 | 138 | 138 |
| Total | ~1,187 | ~600-700 | ~570-640 |

Conclusion: no single technique reaches 600 ms. Streaming STT and colocated LLM
are required, plus a turn detector. Each needs its own before-and-after measurement.

## 4. Top 5 recommendations for DaNa's next sprint (ranked)

1. **Semantic end-of-turn with Pipecat Smart Turn v3 (T1).** Add the local CPU
   analyzer and cut the fixed 200 ms `stop_secs`. Estimated gain ~100-150 ms,
   UNVERIFIED on our box. Effort S-M. Measure false cut-offs on real voice.
2. **Streaming STT partials and speculative LLM on stable partials (T18 + T4).**
   Start the LLM when the partial is stable, cancel on change, and gate on a
   confidence threshold. Estimated gain 100-200 ms, UNVERIFIED. Effort M. Watch
   LLM call cost (+50-70% per Deepgram) and PII in cancelled calls.
3. **Prefix caching and tool-schema trimming (T14).** Measure TTFT with the
   system and tool prefix cached, and with fewer tools per turn. Low effort, low
   risk. Target: keep the 426 ms warm number and cut the tail.
4. **Colocated or local GPU LLM via the existing SGLang preset (T13).** The only
   path that removes the ~200 ms network leg. Needs a GPU host. Effort M.
   Measure TTFT on the same 26-tool prompt before committing.
5. **Per-leg p50/p90 instrumentation and a real-voice benchmark (new).** Log
   end-of-speech, STT final, LLM first token and TTS first audio for every turn.
   Use p90, not only median (production runs 2-3× the lab numbers). Run this
   before and after items 1-4. It also measures the WER gap the 02 document
   flags as unmeasured.

Not recommended this sprint: speech-native LLMs (T7, T8, T9), Sesame (T10),
endpoint anticipation (T5), and TTS swaps (T11). The first need GPUs or cloud
S2S and lose local Vietnamese and privacy. The last two gain little beyond
Kitten's 138 ms.

## 5. Source list

- Daily, Smart Turn v3 CPU 12 ms: https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/ (fetch failed; snippet only)
- Pipecat Smart Turn repo: https://github.com/pipecat-ai/smart-turn
- Daily, Smart Turn v3.1 accuracy: https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/
- LiveKit turn detection docs: https://docs.livekit.io/agents/build/turn-detection
- LiveKit turns docs: https://docs.livekit.io/agents/logic-structure/turns/
- LiveKit community thread (preemptive generation): https://community.livekit.io/t/high-response-latency-after-short-user-utterances-despite-preemptive-generation-and-aggressive-endpointing-agents-1-6-4/1658
- Deepgram Flux quickstart: https://developers.deepgram.com/docs/flux/quickstart.mdx
- Deepgram Flux intro: https://deepgram.com/blog/introducing-flux-conversational-speech-recognition.md
- Deepgram voice agent architecture (eager EOT guide): https://deepgram.com/blog/voice-agent-architecture-stt-llm-tts-pipeline-design.md
- Together Flux listing (p95 1.5 s): https://together.ai/models/flux
- Kyutai STT: https://kyutai.org/stt
- Kyutai Unmute: https://kyutai.org/unmute/
- Unmute architecture (third party, UNVERIFIED): https://www.mintlify.com/kyutai-labs/unmute/architecture/data-flow
- Endpoint anticipation paper: https://arxiv.org/pdf/2606.13450
- Ultravox model card: https://huggingface.co/fixie-ai/ultravox-v0_2/blob/main/README.md
- Cerebrium Ultravox deployment: https://cerebrium.ai/blog/deploying-ultravox-on-cerebrium
- Pipecat Ultravox docs: https://docs.pipecat.ai/server/services/stt/ultravox
- Plainconcepts realtime benchmark (third party): https://www.plainconcepts.com/conversational-ai-avatar-llm-audio2face/
- Gemini Live forum reports: https://discuss.ai.google.dev/t/live-api-latency-spikes/106814
- Cartesia Sonic 3 overview (secondary): https://invideo.io/blog/cartesia-sonic-ai-voice/
- Groq and Cerebras speed (secondary): https://costbench.com/best/fastest-llm-inference/
- RelayS2S paper: https://arxiv.org/pdf/2603.23346
- Voice-Light paper: https://arxiv.org/pdf/2609.20995
- CallSphere latency guide (vendor blog, UNVERIFIED): https://callsphere.ai/blog/voice-agent-latency-optimization-sub-500ms-response-times
- ElevenLabs latency guide: https://elevenlabs.io/blog/voice-agent-latency-optimization
- Hugging Face voice agent latency playbook: https://huggingface.co/blog/dvalle08/voice-agent-latency-playbook
- Vapi vs Retell comparison (secondary): https://growwstacks.com/blog/voice-ai-showdown-vapi-synthflow-retell-latency-test
- Hamming glossary (secondary, UNVERIFIED): https://hamming.ai/glossary/latency
- LiteRT-LM overview (Google, Spanish): https://ai.google.dev/edge/litert-lm/overview?hl=es-419
- Gemma 3n E2B LiteRT-LM model card: https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/blob/refs%2Fpr%2F2/README.md
- Raspberry Pi LiteRT article (Gemma 4, Moonshine): https://www.raspberrypi.com/news/mastering-edge-ai-on-raspberry-pi-with-litert-and-gemma/

## 6. Gaps and next checks

- No official Smart Turn or Flux number was fetched from its primary page. The Daily fetch failed (host not resolvable). Re-check before quoting in a decision doc.
- Cerebras TTFT in ms and Groq TTFT on the same model are not found. Probe with DaNa's 26-tool prompt.
- Gemma 3n audio TTFT on LiteRT-LM: run `litert-lm benchmark` on a target device.
- Vietnamese quality of Ultravox, Groq and Cerebras models: not checked.
- All "ms saved" figures in §1 except DaNa's own (02 §5) are estimates until measured on DaNa's own voice data.
