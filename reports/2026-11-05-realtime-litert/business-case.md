# DaNa Business Case: Cost, Value, and Rollout

Status: draft for owner review · Date: 2026-11-05 · Owner: Business Agent
Prices were checked with web search on 2026-11-05 against aggregator pages, not the vendors' own price sheets. Verify before any purchase decision.

## Executive summary

- At 2 hours of conversation per day, the current cloud stack (AssemblyAI + Gemini LLM + Gemini TTS) costs about **$19 per month** (base case, estimate), with a range of $19-42.
- Text-to-speech is the largest cost line (about 70% of the base case). Speech recognition is small. The LLM is negligible.
- The hybrid option (local speech models + a cheap cloud LLM such as GPT-4.1 nano) costs about **$4-6 per month** (estimate), mostly electricity.
- Fully self-hosted only saves money if the owner already has a capable GPU. Buying one for this (about $1,800, estimate) puts the cost near $58 per month, which is more than the cloud stack.
- Cost is not the decision driver at this usage level. Privacy, latency, and task success are. Confidential supplier and pricing data is the main reason to move speech and LLM work local.
- Recommendation: move to the **hybrid** option. Keep the cloud STT as a fallback. Do not buy GPU hardware until the pilot shows a clear need.
- Value case: if DaNa saves the owner about 15 minutes per workday, that is roughly 5.5 hours per month. At an assumed loaded cost of $50/hour (estimate), that is about $275 per month, which dwarfs any provider cost.
- KPIs: first-audio latency (p50/p95), WER in English and Vietnamese, task success rate, and cost per hour of conversation.
- Rollout: 4 stages over about 10 weeks, each with explicit go/no-go criteria (Section 6).

---

## 1. Usage assumptions (estimate)

| Parameter | Value | Basis |
|---|---|---|
| Conversation time | 2 h/day, 30 days = 60 h/month | Given in the brief |
| User speech sent to STT | 1.0 h/day = 30 h/month | Assumes about half of conversation time is user speech (VAD-gated) |
| DaNa spoken output | 0.5 h/day = 15 h/month | Assumes about a quarter of conversation time is TTS playback |
| Turns | 60 per day = 1,800 per month | Assumes about 30 turns per conversation hour |
| LLM input tokens per turn | 1,500 (history + tool schema) | Estimate. Prompt caching not modelled |
| LLM output tokens per turn | 150 | Estimate |
| Monthly LLM tokens | 2.7 M input, 0.27 M output | Derived |
| TTS audio tokens | 15 h x 3,600 s x 25 tokens/s = 1.35 M | 25 audio tokens per second is from Google Cloud TTS pricing page |

Change these numbers in the model to test other usage levels. Costs scale linearly for STT and TTS and by tokens for the LLM.

## 2. Unit prices used

| Item | Price used | Source | Confidence |
|---|---|---|---|
| AssemblyAI Universal streaming | $0.15 per hour (base); $0.45 per hour for the higher-accuracy realtime tier (high case) | [Activepieces AssemblyAI 2026 guide](https://www.activepieces.com/blog/assemblyai-pricing-2026-rates-and-add-on-costs.md); [Cekura AssemblyAI pricing](https://www.cekura.ai/blogs/assemblyai-pricing); [usagepricing blueprint](https://usagepricing.com/blueprint/assemblyai) | Medium. Sources disagree on the higher tier. Official page not reached |
| Gemini 3.1 Flash-Lite | $0.25 input / $1.50 output per 1M tokens | [anotherwrapper Gemini 3.1 Flash-Lite](https://anotherwrapper.com/tools/llm-pricing/gemini-3.1-flash-lite-preview); [pricepertoken](https://pricepertoken.com/pricing-page/model/google-gemini-3.1-flash-lite) | Medium. One aggregator lists $0.125 / $0.75 |
| GPT-4.1 nano | $0.10 input / $0.40 output per 1M tokens (batch $0.05 / $0.20) | [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-4.1-nano); [Price Per Token](https://pricepertoken.com/pricing-page/model/openai-gpt-4.1-nano) | Medium-high |
| Gemini TTS (2.5 Flash TTS) | $10 per 1M audio output tokens (high case: $20 for Pro or 3.1 Flash TTS) | [Google Cloud Text-to-Speech pricing](https://cloud.google.com/text-to-speech/pricing/); [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) | Medium. Preview models; prices may change. Another listing names a "Flash-Lite TTS" price that could not be confirmed |
| Local STT/TTS (Whisper-class, Piper/Kokoro, LiteRT) | $0 API cost; electricity only | Open-source models | High for cost; hardware and effort are estimates |
| Electricity | $0.20 per kWh | Estimate for a residential rate in the owner's region. Replace with the actual tariff |  Estimate |
| Consumer GPU (RTX 4090-class) | About $1,800, amortized over 36 months | Estimate. Check current street price | Estimate |

## 3. Monthly cost per heavy user (2 h/day)

### 3.1 Option (a): current cloud stack

| Line | Calculation | Base | High |
|---|---|---|---|
| STT (AssemblyAI streaming) | 30 h x $0.15 (high: $0.45) | $4.50 | $13.50 |
| LLM (Gemini 3.1 Flash-Lite) | 2.7 M x $0.25 + 0.27 M x $1.50 | $1.08 | $1.08 |
| TTS (Gemini 2.5 Flash TTS) | 1.35 M x $10 (high: $20) | $13.50 | $27.00 |
| **Total** | | **$19.08** | **$41.58** |

Cost per conversation hour: about $0.32 (base).

### 3.2 Option (b): hybrid

| Line | Calculation | Cost |
|---|---|---|
| STT | Local faster-whisper or LiteRT ASR, CPU/GPU | $0 API |
| TTS | Local Piper or Kokoro | $0 API |
| LLM (GPT-4.1 nano) | 2.7 M x $0.10 + 0.27 M x $0.40 | $0.38 |
| Electricity | About 20 kWh/month (60 h active at extra 100 W, plus 20 W idle overhead, estimate) x $0.20 | About $4 |
| **Total** | | **About $4.4 (estimate)** |

Alternative LLM: Gemini 3.1 Flash-Lite at $1.08 instead of GPT-4.1 nano. Total about $5.1.
Cost per conversation hour: about $0.07-0.09 (estimate).

### 3.3 Option (c): fully self-hosted

| Line | Calculation | New GPU | Existing GPU |
|---|---|---|---|
| Hardware amortization | $1,800 / 36 months | $50.00 | $0 |
| Electricity | About 41 kWh/month (always-on idle at extra 40 W, plus 60 h active at 200 W, estimate) x $0.20 | About $8 | About $8 |
| SGLang LLM | Self-hosted on same GPU. Marginal API cost $0 | $0 | $0 |
| **Total** | | **About $58 (estimate)** | **About $8 (estimate)** |

Cost per conversation hour: about $0.97 with a new GPU, about $0.13 with an existing one (estimate).

Note on SGLang: this study did not measure throughput or model quality for SGLang. The LLM quality of a self-hosted model on Vietnamese and tool calling is the main unknown in option (c).

### 3.4 Summary

| Option | Monthly cost (est.) | Cost per hour (est.) | Notes |
|---|---|---|---|
| (a) Cloud | $19 (range $19-42) | $0.32 | TTS is about 70% of cost |
| (b) Hybrid | $4-5 | $0.08 | Needs a local ASR/TTS runtime on the PC |
| (c) Self-hosted, new GPU | $58 | $0.97 | Not justified on cost alone |
| (c) Self-hosted, existing GPU | $8 | $0.13 | Justified if the GPU is already there |

## 4. Latency and privacy trade-off matrix

Latency figures are estimates for planning. They must be replaced with measured p50/p95 values from the pilot.

| Criterion | (a) Cloud | (b) Hybrid | (c) Self-hosted | Weight |
|---|---|---|---|---|
| First-audio latency p50 (est.) | 0.8-1.5 s | 0.6-1.0 s (local STT, cloud LLM) | 0.4-0.8 s (all local) | High (latency is priority #1) |
| Network dependency | Required | Required for LLM only | None | Medium |
| Audio leaves the premises | Yes (to AssemblyAI, Gemini) | No for audio; text goes to LLM | No | High (privacy is priority #2) |
| Supplier and price text leaves premises | Yes | Yes (to LLM) unless LLM is local | No | High for KAM work |
| Vietnamese quality | Strong (cloud) | Depends on local ASR; test | Depends on local models; test | High |
| Operational effort | Low | Medium | High (GPU ops, updates) | Medium |
| Monthly cost | Highest | Low | Low if GPU exists | Low (not the driver) |
| Vendor lock-in | Medium | Low | Lowest | Medium |

Privacy note for the business: a cloud LLM sends business text (supplier names, pricing, promo terms) to a third party. Whether that is allowed depends on the employer's data policy and the vendor's data-use terms, which were not reviewed in this study. This needs a decision from the owner's IT or legal team, not only from the engineering side.

## 5. Recommendation

**Adopt the hybrid option (b) now, keep cloud STT as a fallback, and defer GPU purchase.**

Reasons:
1. Cost falls from about $19 to about $5 per month (estimate), but cost is not the main argument.
2. It keeps audio local, which is the biggest privacy gain for the least effort. Text still goes to the LLM.
3. The DaNa architecture already supports this. Each provider is a config switch. The local STT/TTS adapters are the main new work.
4. It keeps the self-hosted option open. If the pilot shows the local LLM is good enough, move the LLM to SGLang on an existing GPU (option c, about $8 per month, estimate).

Decision points:
- If the owner's employer forbids sending business text to a cloud LLM, skip (b) for the LLM and go to (c) with an existing GPU.
- If Vietnamese WER with local ASR is worse than the gate in Section 6, keep cloud STT for Vietnamese turns only.

## 6. KPIs and rollout plan

### 6.1 KPIs

| KPI | Definition | Target (estimate; set during baseline) |
|---|---|---|
| First-audio latency p50 / p95 | Time from end of user speech to first TTS audio sample | p50 at or below 1.0 s; p95 at or below 1.8 s |
| WER English | Word error rate on a fixed 200-utterance set of KAM phrases | At or below 10% |
| WER Vietnamese | Same set, Vietnamese | At or below 15% |
| Task success rate | Share of 30 scripted KAM tasks completed without manual fix | At or above 80% |
| Unsafe action count | Destructive actions executed without confirmation | Must be 0 |
| Cost per conversation hour | Provider + electricity / hours | At or below $0.10 (hybrid) |
| Reconnect rate (mobile, Phase 1) | Dropped audio sessions per 100 sessions | At or below 5 |

### 6.2 Rollout stages

| Stage | Weeks | Scope | Go criteria | No-go action |
|---|---|---|---|---|
| 0. Baseline | 1-2 | Current cloud stack on the PC. Measure all KPIs | Baseline recorded for 2 weeks; task set and WER set agreed | Fix measurement before anything else |
| 1. Hybrid pilot (PC) | 3-5 | Local STT + TTS, cloud LLM (GPT-4.1 nano or Gemini Flash-Lite) | p50 first-audio at or below 1.0 s; EN WER at or below 10%; VI WER at or below 15%; zero unsafe actions | Keep cloud STT; tune or change local model |
| 2. Phone PWA (Phase 1) | 6-9 | Push-to-talk from phone over Tailscale | Reconnect rate at or below 5%; task success at or above 80% on phone; device pairing and revocation tested | Stay on PC; fix transport |
| 3. Self-hosted decision | 10 | Measure SGLang LLM on existing GPU, if any | Task success within 5 points of cloud LLM; latency at or below 1.0 s p50; GPU available without new purchase | Stay on cloud LLM; no hardware purchase |

Each stage needs an owner sign-off. Stop and roll back to the previous stage if the unsafe-action count is above 0 at any point.

## 7. Value case and KAM use cases

### 7.1 Value (estimate)

- Time saved: 15 minutes per workday is assumed. That is about 5.5 hours per month (estimate).
- Value at an assumed loaded cost of $50/hour: about $275 per month (estimate).
- Provider cost of $5-19 per month is less than 10% of that value.
- The value only holds if task success is high. A 50% success rate halves the saving and adds rework. Task success is therefore the KPI that matters most.

### 7.2 Use cases for the computer-use layer (generic accessibility snapshot + MCP)

These are concrete tasks for a Key Account Manager in beauty and personal-care retail e-commerce. Each one fits a risk tier in the action gate.

| Use case | What DaNa does | Gate tier (proposed) |
|---|---|---|
| Voice browsing of brand portals | "Open the Brand X portal and go to the promo calendar for Q1." DaNa navigates and reads the page aloud | Read-only: ALLOW |
| Reading sales dashboards | "What was last week's sell-out for the top 5 SKUs in Shopee?" DaNa opens the dashboard and reads the numbers | Read-only: ALLOW. Credentials stay in the vault |
| Competitor price checks | "What is the shelf price for this serum on two competitor sites?" DaNa reads product pages | Read-only: ALLOW |
| Drafting supplier emails | "Draft a reply to the supplier about the delayed shipment, mention the PO number." DaNa writes a draft in the mail client | Reversible write (draft only): CONFIRM if sent; ALLOW for draft |
| Sending emails and submitting forms | "Send it." | Destructive or external: CONFIRM every time |
| Meeting prep summary | "Summarise the last three emails from this retailer buyer." | Read-only: ALLOW |
| Trade-spend and promo entry | Filling a promo request form in a retailer portal | Destructive or financial: CONFIRM; dry-run preview first |

Operational points:
- Screen reading through the accessibility snapshot is faster than screenshots for most portals. Test on the owner's top three portals.
- Retailer portals often have bot checks and session timeouts. The existing `bot-check/resolve` route handles the first; plan for the second.
- Use the owner's own work account and the company's approved browser profile. Do not store portal passwords in the agent until the vault is in place.

### 7.3 Risks specific to this business use

| Risk | Impact | Mitigation |
|---|---|---|
| Business data sent to a cloud LLM | Policy breach or confidentiality issue | Hybrid option; written approval from owner's IT/legal before cloud LLM use with supplier data |
| DaNa sends an email or submits a form by mistake | Commercial or contractual error | Confirmation required for any external send. Audit log |
| Wrong number read from a dashboard | Wrong decision | Read back the source and page title; show it in the transcript |
| Vietnamese speech misrecognised | Wrong product or supplier name | Confirm names aloud before any write action |

## 8. Open items

- Confirm the owner's actual conversation hours and electricity tariff. All costs above scale with these.
- Check current prices on the vendors' official pages before any purchase. Several sources disagree (AssemblyAI higher tier, Gemini Flash-Lite, Gemini TTS preview pricing).
- Confirm the data-use terms of each cloud vendor with the company's IT or legal team.
- Decide whether a phone version (see `mobile-app-plan.md`) is needed for the owner's workflow.

## 9. Sources

- [Activepieces: AssemblyAI pricing 2026](https://www.activepieces.com/blog/assemblyai-pricing-2026-rates-and-add-on-costs.md) - AssemblyAI streaming rates (aggregator)
- [Cekura: AssemblyAI pricing](https://www.cekura.ai/blogs/assemblyai-pricing) - AssemblyAI streaming rates (aggregator)
- [usagepricing: AssemblyAI](https://usagepricing.com/blueprint/assemblyai) - higher realtime tier (aggregator)
- [anotherwrapper: Gemini 3.1 Flash-Lite pricing](https://anotherwrapper.com/tools/llm-pricing/gemini-3.1-flash-lite-preview) - LLM rates (aggregator)
- [pricepertoken: Gemini 3.1 Flash-Lite](https://pricepertoken.com/pricing-page/model/google-gemini-3.1-flash-lite) - LLM rates (aggregator)
- [OpenAI: GPT-4.1 nano model page](https://developers.openai.com/api/docs/models/gpt-4.1-nano) - official model page; price confirmed only through aggregators
- [Price Per Token: GPT-4.1 nano](https://pricepertoken.com/pricing-page/model/openai-gpt-4.1-nano) - LLM rates (aggregator)
- [Google Cloud Text-to-Speech pricing](https://cloud.google.com/text-to-speech/pricing/) - TTS audio token rates and 25 tokens per second
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) - official pricing page; search returned it but its content was not read in full
- Local runtime facts: see `mobile-app-plan.md` Section 7.

All hardware, electricity, usage, value, and latency figures are estimates. Vendor prices are from third-party aggregators unless marked as official, and must be checked before use.
