# DaNa realtime upgrade — report index (2026-11-05)

**Scope requested:** host LiteRT ASR/TTS models to speed up responses (goal: first
audio < 0.6 s), a model registry API that browses and pulls from HuggingFace
`litert-community` with one request, a UI to manage it all (CPU/WebGPU), LLMs via
SGLang or OpenAI/Gemini (plus Together AI, added mid-session), generic
plug-and-play computer use instead of hard-coded sites, and a mobile plan.

## Executive summary

1. **Root cause of the slowness:** two legs dominated. Cloud generative TTS
   (Gemini TTS, ~1 s+ to first byte) and LLM hidden thinking (Gemini default
   thinking; the Together model reasons by default). End-to-end was ≈ 2–4 s to
   first audio.
2. **Self-hosted speech on LiteRT works for real.** It was tested on the actual
   `litert-community` models, not mocks:
   - **ASR:** Moonshine-tiny (EN) takes **250–480 ms**; Whisper-tiny (EN+VI)
     takes ~0.8–1.5 s. Silence costs ~1 ms thanks to the energy gate.
   - **TTS:** KittenTTS-nano (EN) reaches first speech in **169–335 ms**. It was
     ~1 s before we found and clamped 0.8 s of model-generated leading silence.
   - **Vietnamese TTS:** Piper (no Vietnamese LiteRT TTS exists in the org).
3. **The LLM is tuned and measured live.** Together DeepSeek-V4.1-Flash with
   reasoning off has a warm TTFT of **~375 ms** (cold: 7.5 s, which is why DaNa
   now warms the LLM at session start). SGLang, vLLM, OpenAI, Gemini and Together
   are one-click presets with a live "Measure latency" probe.
4. **Honest latency status:** measured legs add up to ≈ **1.1 s** on this 4-vCPU
   container (vs ≈ 2–4 s before). That is a 2–4× improvement, but not yet
   < 0.6 s. The remaining ~500 ms has a concrete plan: incremental STT while the
   user speaks, a local GPU LLM via SGLang, and a speculative first clause. See
   `02-…` §5.
5. **Model registry:** browse the 443-repo catalog, see file recommendations
   (NPU variants filtered out; multi-stage models complete; tokenizer companions
   added), and **pull + load in one request**. Verified live against
   huggingface.co (62.9 MB KittenTTS pulled and loaded in ~10 s).
6. **Generic computer use:** `browser_snapshot` → `browser_act(ref)` works on any
   website (validated in real Chromium). There are Windows UI Automation tools
   for any desktop app, and an **MCP bridge** that plugs in any MCP server.
   Everything still passes through the safety gate (confirm destructive, refuse
   out-of-scope, audit).
7. **UI:** a new 🧩 Models tab (hub, pull jobs, local models, assign ASR/TTS,
   LLM hosting, latency budget, in-browser WebGPU runner) and a PWA manifest
   for phones.
8. **Mobile:** a 3-phase plan. Phase 1 is a PWA remote client (≈ 20–25
   person-days); Phase 2 is native with on-device LiteRT; Phase 3 is standalone.

## Documents (read in this order)

| # | File | What it covers |
|---|---|---|
| 1 | `01-architecture-deep-dive.md` | High-level concept → mechanisms of every component, latency budget math |
| 2 | `02-implementation-and-verification.md` | What shipped (files, APIs, config), test evidence, measured latency, gap-closing plan, runbook |
| 3 | `litert-community-catalog.md` | Live snapshot of the 443 litert-community repos (all ASR/TTS listed) |
| 4 | `00-team-contract.md` | Team structure, ownership and the REST contracts the team built against |
| 5 | `lead-a-backend-status.md`, `lead-b-product-status.md` | Team-lead task boards with evidence |
| 6 | `research-latency-llm-computer-use.md` | Latency budget research, SGLang, computer-use standards |
| 7 | `research-litert-ecosystem.md` | LiteRT runtime / WebGPU research (§1–3 superseded by the catalog, see note) |
| 8 | `ux-models-ui.md` | Models-tab UX: screens, flows, states |
| 9 | `mobile-app-plan.md` | Mobile app phases, audio WebSocket contract, store constraints |
| 10 | `business-case.md` | Cost per heavy user (cloud vs hybrid vs self-hosted), KPIs, rollout, KAM use cases |

## Team

The owner integrated and decided. Two team leads (Sonnet) handled verification and
coordination. Six coding agents (Sonnet) each owned disjoint files (registry,
LiteRT runtime + ASR, latency pipeline, TTS, LLM + MCP, computer use). One UX
agent (Sonnet) built the UI. Two research agents (Haiku) and one business agent
(Haiku) wrote the research, business and mobile documents.
