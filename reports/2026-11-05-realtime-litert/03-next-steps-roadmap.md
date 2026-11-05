# 03 - Next steps roadmap (latency to < 600 ms, product, mobile)

Owner: Team Lead C. Date: 2026-11-05. Status: draft for the owner.

Legend: "measured" = `scripts/e2e_latency.py` on the 4-vCPU container (no GPU, no AMX).
"est." = engineering estimate, NOT measured; every est. has a go/no-go gate that
replaces it with a measurement. Vendor numbers in R3's report are "claimed" and are
not used here as facts.

## 0. Measured baseline (last speech frame to first REPLY audio)

| Case | End-to-end (ms) |
|---|---|
| Before this work | about 2000-4000 |
| Moonshine STT + Brain (Together DeepSeek-V4.1-Flash, 26 tools) + KittenTTS, plain | 1311 |
| + speculative STT | 1133 (saves 178) |
| + speculative STT + early commit | 1051 (saves 260 vs plain) |
| Instant "Mm-hm." ack | about 220 (a filler, not the reply; do not count it toward the KPI) |

Stages: Moonshine STT 360-570 ms from last speech; LLM TTFT 440-480 ms (a bare warm
GET to Together is already 160-260 ms to first byte); first LLM token to first TTS
audio 260-295 ms; Kitten TTS 138 ms for a 21-char clause.

## 1. Diagnosis: why the real reply cannot be < 600 ms here

LLM TTFT (440-480 ms) plus first-token-to-audio (260-295 ms) is about 700-775 ms
before any STT time is counted, so the target is already missed by the back half of
the chain alone. Three causes are structural, not tuning problems. (a) The LLM is a
remote API: the network floor is 160-260 ms before the model computes anything, and
26 tool schemas add prefill. (b) On 4 vCPUs with no GPU, a local LLM or a better
TTS cannot be faster than the cloud (a 7B+ model decodes at single-digit to low
tens of tokens/s on CPU, and its prefill with tool schemas would take longer than
the whole 600 ms budget). (c) Moonshine STT needs 360-570 ms after speech stops
because it is not truly streaming. Sub-600 ms therefore needs a change of
architecture or hardware, in this order of leverage: (1) put the LLM next to the
audio, on a local GPU (SGLang/vLLM, a small tool-capable model, prefix-cached system
prompt and tool schemas) or at least a provider with a regional endpoint and
sub-150 ms TTFT; (2) overlap work instead of serial stages: streaming STT with a
partial-transcript endpoint, preemptive LLM generation on stable partials, and a
semantic end-of-turn model so the 200 ms VAD hold shrinks; (3) cut TTS first-audio
by streaming sub-clause chunks (5-8 words or a punctuation-less 20-char flush) on a
faster TTS. Est. budget on a GPU PC: endpoint 100 + STT tail 50 + LLM TTFT 120 +
TTS first audio 130 = about 400-500 ms (est.). Without a local GPU, < 600 ms is not
credible for tool-using turns; it is only credible for turns answered by a
pre-generated or cached reply.

## 2. Prioritized roadmap

Columns: ms saved / KPI, effort (S < 2 d, M 3-8 d, L > 2 wk), owner role, go/no-go.
All ms saved are est. unless marked measured.

### NOW (this week, no new hardware)

| # | Item | ms saved / KPI | Effort | Owner | Go/no-go |
|---|---|---|---|---|---|
| N1 | Fix, soak, then default-on speculative STT + early commit (English first, Vietnamese opt-in). Pipeline reviewer found 3 MED bugs in these opt-in paths, Lead A is fixing them: (1) cancel on Cancel/End/Interruption, (2) `voice_to_first_audio` missing for early-committed turns, (3) continued speech after an early commit is dropped. Do NOT flip defaults until all 3 are fixed and a soak test passes | measured 178 / 260 ms; e2e 1051 | S | Backend eng (Lead A) + QA | Go if the 3 fixes are merged, soak test clean (no stuck turns, no dropped speech), median e2e over 10 runs <= 1100 and false early-commit rate < 5% on a 50-utterance set; else keep opt-in |
| N2 | Keep-alive + connection pre-warm to the LLM (`keepalive_s`), HTTP/2, prefix-cached static prompt | up to 100 of the 160-260 ms first-byte (est.) | S | Backend eng | Go if TTFT p50 drops >= 60 ms in `e2e_latency.py` |
| N3 | Trim tool schemas per turn (route to a 6-8 tool subset by intent) to shorten prefill | 30-80 (est.) | M | Backend eng | Go if tool-selection accuracy stays >= current on `action_routing_eval.py` |
| N4 | Lower `tts_first_chunk_min_chars` (try 12-16) and flush on the first clause boundary | 20-60 on the 260-295 ms gap (est.) | S | Backend eng | Go if no audible mid-word cuts in a 30-sample listening check |
| N5 | Report p50 and p95 per stage in the UI waterfall; make the e2e script a CI-optional gate | KPI: p95 visible | S | Backend eng + QA | Go when the table is reproducible within +-10% across 3 runs |
| N6 | Fix doc numbers (speculative saves 178 ms, not 240) in user guide and README.vi | KPI: docs match measurements | S | Tech writer | Done when grep of guides shows measured values only |

### NEXT (2-6 weeks; needs the owner's Windows PC and a GPU)

| # | Item | ms saved / KPI | Effort | Owner | Go/no-go |
|---|---|---|---|---|---|
| X1 | Local GPU LLM via SGLang (or vLLM) behind the existing OpenAI-compatible brain provider (config only: `base_url`), small tool-capable model, FP8/AWQ, prefix cache | LLM TTFT 440-480 down to 100-150 (est.); saves 300+ | M | Backend eng + ML eng | Go if TTFT p50 <= 150 ms with 26 tools AND tool-call accuracy >= 95% of the cloud model on `action_routing_eval.py`; no-go: stay on cloud with regional endpoint |
| X2 | Speculative LLM on stable partials (R3 T4): start the LLM when the partial is unchanged, cancel if the final text differs. Cost: +50-70% LLM calls (Deepgram claim, UNVERIFIED), so budget tokens | 100-200 (R3, UNVERIFIED) | M | Backend eng | Go if wasted-call rate < 20% and no wrong action ever executes (gate layer unchanged, reads only until the final transcript confirms) |
| X3 | Streaming STT with partials (replace Moonshine tail 360-570 ms) | 200-350 (est.) | M | ML eng | Go if WER on EN and VI test set within +10% relative of Moonshine and tail < 150 ms |
| X4 | Semantic turn detection: Pipecat Smart Turn v3 replaces the fixed 200 ms stop (R3 T1; 10-95 ms CPU inference per decision, vendor, UNVERIFIED on our box); A/B on real EN and VI voice | about 100-150 net (R3, UNVERIFIED) and fewer false cuts | S-M | ML eng | Go if the false-cut rate <= current and hold <= 100 ms |
| X5 | Faster/streaming TTS on GPU with sub-clause chunks (first audio from 5-8 words) | 138 down to 60-90 (est.) | M | ML eng | Go if MOS-style listening check shows no regression for EN and VI |
| X6 | Windows PC measurement pass: record e2e, per-stage p50/p95, CPU/GPU load, sleep/battery behaviour | KPI: first real number on target hardware | S | QA + owner | Go/no-go for everything above: run before buying or installing anything |
| X7 | R4 Bet 2, Vietnamese voice quality gate (do first): 200-utterance VI/EN KAM phrase set, WER and first-audio for Moonshine/Whisper vs cloud STT | KPI: VI WER <= 15%, p50 first audio <= 1.0 s (R4 est.) | S-M | ML eng + Product | GO if both met; NO-GO (keep cloud STT for VI) if WER > 15% |
| X7b | R4 Bet 3, preview-then-confirm write path: dry-run preview, read-back of names and numbers, CONFIRM on every external or financial action, audit log in UI | KPI: unsafe actions 0, rework <= 10% | M | Backend eng + Product | GO if 0 unsafe actions; NO-GO if any external send or promo submission happens without CONFIRM |
| X7c | R4 Bet 1, KAM skill pack (product track): seller-center reads, competitor price scans, promo-calendar reading, dashboard Q&A, email drafts; read-only tier first, writes only after X7b | KPI: task success >= 80% on 30 scripted KAM tasks | M-L | Product + Backend | GO >= 80% with 0 unsafe actions; NO-GO < 70% or any unsafe action. Sequence: X7, X7b, X7c |
| X8 | PWA remote client (mobile-app-plan Phase 1) to the home PC agent | KPI: remote e2e <= local + 150 ms (network) | M | Mobile/Frontend | Go if X1 is done and RTT p95 < 100 ms on target networks |

### LATER (1-3 months, bets)

| # | Item | ms saved / KPI | Effort | Owner | Go/no-go |
|---|---|---|---|---|---|
| L1 | Speech-native LLM / S2S (audio in, audio out) via the existing `S2SProvider` slot | removes STT and TTS hops; est. 300-500 total, vendor-claimed figures only | L | ML eng | Go only if tool use is reliable, Vietnamese quality is acceptable and the gate layer can still validate every action |
| L2 | Hybrid native mobile (mobile-app-plan Phase 2) with on-device wake + VAD | KPI: wake false-accept < 1/day, battery drain < 5%/h idle | L | Mobile eng | Go if the PWA pilot shows weekly active use by >= 3 of 5 users |
| L3 | On-device standalone (mobile-app-plan Phase 3, LiteRT-LM Gemma class) for offline/privacy mode | KPI: works offline for 5 core intents | L | Mobile + ML eng | Go if on-device decode >= 25 tok/s on a mid phone and quality passes the 5-intent eval |
| L4 | Extend the KAM pack to the other R4 use cases and more portals after the 30-task gate passes | KPI: weekly retained pilot users, hours saved per week (est.) | L | Product | Go if X7c passes and pilot retention >= 60% at week 4 (est.) |
| L5 | Distillation: small local model fine-tuned on our 26-tool traces for routing only | LLM prefill and decode cost down further | L | ML eng | Go if it beats the N3 router on accuracy at lower latency |

## 3. Fastest credible path to < 600 ms on the owner's Windows PC

Assumption (must be confirmed by step 1): the PC has an NVIDIA GPU with >= 12 GB VRAM
(RTX 3060 12 GB or better), Windows 11 with WSL2 or Docker, and a wired or good Wi-Fi
connection. If it has no discrete GPU, this path does not reach 600 ms with tools;
use the fallback at the end.

Target config (all via `.env` / `dana.toml`, no code change; env prefix `DANA_`,
nested delimiter `__`):

1. Measure first. On the PC run `cd dana-agent && uv run python scripts/e2e_latency.py --runs 5`
   with the current cloud config. Record the table. This is the real baseline; the
   container numbers above are only a proxy.
2. Turn on the free wins: `speculative_stt=true`, `early_commit_ms` (start at 150),
   `keepalive_s=20`, `warmup_on_start=true`, `ack_enabled=false` for the KPI run
   (the ack is not the reply). Re-run the script; expect about 1050 ms or better.
3. Serve a small tool-capable instruct model locally with SGLang in WSL2 or Docker,
   exposing an OpenAI-compatible `/v1` endpoint, with prefix caching on, quantized
   to fit VRAM, `max_new_tokens` capped. Model choice by tool-accuracy test, not by
   size (candidates to test: a 7-9B class instruct model, FP8 or AWQ).
4. Point the brain at it by config only: set the LLM `base_url` to
   `http://localhost:<port>/v1`, same OpenAI-compatible provider, and set the
   model name. Re-run the script. Go/no-go: LLM TTFT p50 <= 150 ms and
   `scripts/action_routing_eval.py` accuracy within 5 points of the cloud model.
5. Replace the STT tail: use a streaming/partial-capable STT (or Moonshine on GPU
   if it exports to it) so the final transcript lands <= 150 ms after end of speech.
   Re-run. Go/no-go: WER on the EN and VI sample set not worse than +10% relative.
6. Move TTS to GPU, lower `tts_first_chunk_min_chars` to 12-16. Re-run. Go/no-go:
   first LLM token to first audio <= 130 ms.
7. Enable preemptive generation (X2) and the semantic turn detector (X4) once built.
8. Acceptance: `uv run python scripts/e2e_latency.py --runs 10` reports median
   last-speech-to-first-reply-audio < 600 ms and p95 < 800 ms on the PC, and
   `uv run pytest` plus `uv run mypy` stay green.

Expected: about 400-500 ms (est.) once steps 3-6 land. Each step has its own gate;
if step 4 fails, the cloud path with steps 1-2 (about 1050 ms, measured in the
container) is the fallback and the < 600 ms target for tool-using turns is
postponed. Fallback for no GPU: pair steps 1-2 with a regional/faster hosted
endpoint and report honestly that the reply is about 0.8-1.0 s (est.), with the ack
at about 220 ms for perceived responsiveness.

## 4. Open items

- R3 and R4 are merged (X2, X4, X7, X7b, X7c). R3's file was reconciled to the true
  e2e figures (1311 / 1133 / 1051 ms); its ~600-700 ms projection stays UNVERIFIED.
- R4 caveats: Vietnamese support for ChatGPT, Gemini Live and Apple Intelligence is
  unverified; prices are secondary-source.
- R3 also lists streaming STT with partials (T18) as the enabler for X2; it maps to X3.
- All "est." figures need a measurement on the PC (step 1) before being quoted.
