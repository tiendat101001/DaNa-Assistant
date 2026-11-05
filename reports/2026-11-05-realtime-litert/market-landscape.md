# DaNa market landscape (R4) · 2026-11-05

Status: draft for owner review. Research by Research Agent R4 (market landscape).
Method: 13 web searches on 2026-11-05. No vendor pricing pages were fetched directly, so most prices come from secondary sources. Every claim tagged **(verified-secondary)** or **(unverified)** needs a check on the vendor's own page before it goes into a decision. Items marked **(estimate)** are my judgement.

## Executive summary

1. The big assistants (ChatGPT, Gemini Live, Copilot, Siri, Alexa+) are cloud-first. None is built to act on a Vietnamese seller's own tools with a confirmation gate.
2. Browser agents (ChatGPT agent, Claude for Chrome, Comet) can act on websites, but they run in the cloud or in a browser and are slow. They are not voice-first.
3. Open-source assistants (Home Assistant Assist, OVOS, Leon) are private and self-hostable, but their Vietnamese speech is weak or unverified.
4. DaNa's real differentiators are the combination of local speech (LiteRT, measured), Vietnamese plus English, act-on-any-app through a safety gate, and self-hosting with no per-hour cost.
5. DaNa is behind on polish, on measured end-to-end latency (about 1.0 s measured vs. vendors' sub-second feel, estimate), on the Vietnamese voice model (Piper only), and on a proven safe write path.
6. Five KAM use cases fit DaNa's read-only tier first (seller-center reads, competitor price scans, promo-calendar reading, dashboard Q&A, email drafts). External sends and promo submissions stay behind CONFIRM.
7. Three product bets for next quarter: (a) a KAM skill pack for seller centers and brand portals, (b) a Vietnamese voice quality gate, (c) a preview-then-confirm write path. KPIs are in Section 4.
8. Main risk: business data sent to a cloud LLM. The hybrid design and local-LLM option (SGLang) are the answer, and that needs an IT/legal decision.

## 1. Competitive landscape

Latency class and pricing are as found; the "est." tag marks my own grading. "Acts on apps" means the product can operate software beyond its own app.

| Product | Vietnamese support | Latency class | On-device / privacy | Can act on apps | Pricing (US list unless noted) |
|---|---|---|---|---|---|
| ChatGPT Advanced Voice | Unverified. No source confirms Vietnamese voice; OpenAI's "improved accents in select languages" wording does not name it | Low, realtime (est.) | Cloud only | Limited: plugins/tools in chat; no general desktop control | Plus $20/mo; free tier gets a limited preview (**verified-secondary**). Vietnam Plus price about 522,500 VND/mo with VAT (**unverified**: <https://blog.laozhang.ai/en/posts/chatgpt-plus-price-vietnam>) |
| Gemini Live | Not confirmed. 2024 launch list of 40+ languages does not name Vietnamese; **unverified for 2026** | Low, realtime (est.) | Cloud | Limited: connected Google apps only | Free tier includes Gemini Live (**verified-secondary**). Google AI Pro $19.99/mo US |
| Microsoft Copilot Voice (Windows) | Yes, listed on Microsoft's supported-languages page (**verified-secondary**; exact Windows requirements unverified) | Low to medium (est.) | Cloud; Windows voice trigger ("Hey, Copilot") | Some Windows integration; limited general action | Free for consumers (est.); Microsoft 365 bundles vary |
| Siri / Apple Intelligence | Siri Vietnamese live since iOS 18.4 (**verified-secondary**). Apple Intelligence Vietnamese reported for 2025-26 but sources conflict | Medium (est.); on-device parts are fast | Mixed: on-device where possible, Private Cloud Compute otherwise | Apple app intents only; limited third-party app control | Free with supported iPhones |
| Alexa+ | **No**. Vietnam is not in the launch list (**verified-secondary**) | Low, cloud (est.) | Cloud | Skills and smart-home; no general desktop control | Free with Prime; about $19.99/mo otherwise (US, **verified-secondary**: <https://subger.com/en/service/alexa-plus>) |
| Home Assistant Assist | Weak. Only a third-party cloud-based (FPT) Vietnamese TTS component found; VI pipeline reports of problems (**unverified**) | Local, depends on STT (est.) | Fully local option | Home-automation entities only | Free / open source |
| OpenVoiceOS (OVOS) | Not confirmed | Medium (est.); offline STT claims conflict (ToolMage says default STT is online) | Designed for privacy; offline options such as Vosk | Limited skills; no general app control | Free / open source |
| Leon | Not confirmed; 2.0 preview on `develop` may change offline behavior | Depends on model (est.) | Self-hosted, optional offline | Limited; custom skills | Free, MIT licence (**verified-secondary**) |
| Open Interpreter (desktop mode) | Via the chosen LLM only; no native Vietnamese voice | Depends on LLM (est.) | Local execution; LLM may be cloud | Yes: screen reading, mouse and keyboard (v0.4.0+, **verified-secondary**). Raw shell code execution is a risk for a safety gate | Free / open source; licence conflict (Apache 2.0 vs AGPL-3.0 across sources) |
| OpenAI ChatGPT agent (ex-Operator) | Text, not voice-first | Slow: multi-step browser runs (est.) | Runs in OpenAI's cloud browser | Yes, on websites; asks before sensitive steps | Pro $200/mo reported; Plus with limits reported; **sources conflict, unverified** |
| Claude for Chrome | Via Claude's language ability; no voice product | Slow: agentic browsing (est.) | Runs in the user's Chrome with their sign-ins | Yes, on websites | Included in paid Claude plans, no separate charge (**verified-secondary**; date of GA conflicts across sources) |
| Perplexity Comet | Not confirmed | Slow: agent tasks (est.) | Cloud AI; local browser | Yes, on websites (agent) | Browser free worldwide since Oct 2025; agent queries limited on free tier (**verified-secondary**) |
| **DaNa (this project)** | **Yes, EN and VI**; VI speech via Piper (TTS) and Whisper-tiny (ASR) | **Measured:** about 1.0 s voice to reply, about 220 ms with instant ack (see README) | **Local speech option (LiteRT), self-hosted LLM option**; cloud LLM optional | Yes: browser snapshot/act, Windows UI Automation, MCP, all through the safety gate | Software free/self-hosted; per-hour cost about $0.07-0.09 hybrid (estimate, business case) |

Caveat on the table: vendor voice products change every few months. Re-check the "unverified" cells before the owner relies on them.

## 2. Where DaNa is differentiated and where it is behind

### Differentiated (defensible)

- **Privacy and local speech.** Speech can run on the PC (measured LiteRT: Moonshine-tiny 250-480 ms, KittenTTS first speech 169-335 ms). Only the LLM text leaves, and that can also be local via SGLang. Consumer assistants cannot offer that.
- **Vietnamese plus English in one product.** Most open-source assistants have no working Vietnamese; Alexa+ and several others do not cover Vietnam at all.
- **Acting on any app, with a gate.** Browser snapshot/act and Windows UI Automation work on any site or desktop app. Read-only is allowed, destructive actions are confirmed, out-of-scope is refused, and everything is audited. Browser agents act on websites only, and not with a voice-first loop.
- **Self-hosting and cost.** Hybrid cost is about $4-6/mo per heavy user (estimate). Cloud stack is about $19/mo (estimate). Local-GPU option is about $8/mo if a GPU already exists (estimate).
- **Model registry.** Browse, pull and load from HuggingFace litert-community in one request. No consumer assistant lets users choose and swap speech models.

### Behind (must fix)

- **Latency.** Measured about 1.0-1.2 s to reply (estimate: 0.6 s target not yet met). Realtime vendor voice products are roughly 0.3-0.6 s to first audio (est.). The LLM round trip (about 370-450 ms on Together) is the floor.
- **Vietnamese voice quality.** Piper TTS and Whisper-tiny ASR are not evaluated on a fixed VI set yet. Vendor voice products are likely better on VI (est.).
- **Polish and onboarding.** No install-and-go experience. Setup needs a dev environment and API keys.
- **Mobile.** PWA remote is a plan only (about 20-25 person-days, estimate). Vendor apps are native.
- **Trust in the write path.** No measured task success rate yet. Vendors have larger safety teams and clearer consent UX.
- **Ecosystem.** No connectors to Shopee, Lazada, TikTok Shop, or Teams/Outlook. Vendors have Google Workspace, Microsoft 365, and Apple app integrations.

## 3. Five high-value KAM use cases (beauty e-commerce, Vietnam)

Gate tiers follow the business case: ALLOW (read-only), CONFIRM (write, external, or financial), REFUSE (out of scope).

| # | Use case | Example voice request | DaNa capability used | Gate tier |
|---|---|---|---|---|
| 1 | Seller-center status checks (Shopee, Lazada, TikTok Shop) | "Đơn hàng hôm nay của serum X, tồn kho bao nhiêu?" (How many orders and how much stock today for serum X?) | Browser snapshot on seller center; read order and stock tables; read back page title and time | ALLOW. Check each platform's terms on automated access before the pilot (**unverified**) |
| 2 | Competitor shelf-price scan | "Giá kem chống nắng của đối thủ A trên hai sàn là bao nhiêu?" | Browser snapshot on product pages; extract price, promo label, stock; write to a local table | ALLOW. Rate-limit requests; store source URL and timestamp |
| 3 | Promo-calendar reading from brand portal or email | "Lịch khuyến mãi tháng 12 của brand Y có gì?" (What is brand Y's December promo calendar?) | Browser or Outlook read; summarise dates, mechanics, and deadlines | ALLOW for read. Adding an event to a calendar = CONFIRM |
| 4 | Sell-out dashboard Q&A (Power BI, ERP export) | "Top 5 SKU bán chạy tuần trước ở Shopee?" | Browser or local file read of a dashboard or CSV export; numbers read back with source | ALLOW. Keep data local (local LLM preferred) |
| 5 | Supplier email drafting and promo-request entry | "Soạn email cho nhà cung cấp về đơn trễ, có số PO." (Draft an email to the supplier about the late order, include the PO number.) | Draft in mail client; for promo entry: dry-run preview then fill form | Draft = ALLOW. Send = CONFIRM every time. Promo submission = CONFIRM with preview |

Notes: Vietnamese names of products and suppliers must be read back and confirmed before any write (business-case risk table). Customer personal data should not go to a cloud LLM.

## 4. Three product bets for next quarter

Each bet has a KPI with a target. Targets are **(estimate)** unless tied to the business case.

Go / no-go rule (estimate): proceed to the next stage only if the KPI meets its target on the 30-task pilot. If it misses, keep the cloud fallback for that path and re-test once. Go/no-go per bet:
- Bet 1: GO if task success is at least 80% on 30 scripted tasks with zero unsafe actions. NO-GO if below 70%, or if any unsafe action occurs.
- Bet 2: GO if VI WER is at most 15% and p50 first audio is at most 1.0 s. NO-GO (keep cloud STT for VI) if VI WER is above 15%.
- Bet 3: GO if unsafe actions are 0 and rework is at most 10%. NO-GO if any external send or promo submission happens without CONFIRM.

| # | Bet | What we build | KPI | Target (est.) | Baseline |
|---|---|---|---|---|---|
| 1 | **KAM skill pack for seller centers and brand portals** | Scripted, tested flows for 5-6 KAM tasks on the top three portals; accessibility-snapshot selectors versioned per site | Task success on 30 scripted KAM tasks, no manual fix | At least 80% (business case) | Not measured |
| 2 | **Vietnamese voice quality gate** | 200-utterance VI and EN KAM phrase set; measure WER and first-audio latency for Moonshine/Whisper vs. cloud STT; keep cloud STT as VI fallback if the gate fails | VI WER and p50 first audio | VI WER at or below 15%; p50 first audio at or below 1.0 s | Measured latency about 1.0 s; VI WER not measured |
| 3 | **Preview-then-confirm write path** | Dry-run preview, read-back of names and numbers, and CONFIRM for every external or financial action; audit log surfaced in the UI | Unsafe action count; rework rate on write tasks | Unsafe actions 0; rework at or below 10% | No data |

Sequencing (estimate): Bet 2 first (it decides the local-vs-cloud STT path), then Bet 3 (safety must be proven before Bet 1 writes), then Bet 1 with the pilot.

## 5. Open items and what to verify

- Confirm Vietnamese support for ChatGPT Advanced Voice, Gemini Live, and Apple Intelligence on each vendor's official page (**unverified**).
- Confirm Shopee, Lazada, and TikTok Shop terms on automated reads by an agent (**unverified**).
- Confirm the owner's employer policy on sending business text to a cloud LLM (business case, Section 4).
- Replace all secondary-source prices with official pages before any purchase decision.
- Check Claude for Chrome GA date and Comet free-tier limits on the vendor pages.

## 6. Sources (secondary unless noted)

- Vietnamese ChatGPT Voice and Advanced Voice coverage: <https://phongvu.vn/cong-nghe/chatgpt-voice-advanced-song-dong-chan-thuc/>
- Gemini Live 40+ languages (2024 launch coverage): <https://blog.google/products/gemini/gemini-live-extensions-language-expansion/>
- Google AI plan pricing summaries: <https://pricepertoken.com/subscriptions/gemini>
- Microsoft Copilot supported regions and languages (official): <https://support.microsoft.com/en-us/topic/26de43a1-c176-4908-bef7-29c8c37ac7ce>
- Apple Siri / Apple Intelligence Vietnamese (VnExpress, MacRumors): <https://e.vnexpress.net/news/tech/tech-news/vietnamese-siri-goes-live-on-iphone-supports-ride-hailing-money-transfer-4868566.html>, <https://www.macrumors.com/2025/09/22/ios-26-1-apple-intelligence-languages/>
- Alexa+ country list (PPC Land): <https://ppc.land/amazon-prices-alexa-in-japan-at-y-3-000-a-month-five-times-the-prime-fee/>
- OpenAI Operator / ChatGPT agent (third-party reviews, conflicting): <https://ucstrategies.com/news/openai-operator-specs-pricing-real-world-performance-guide-2026/>
- Claude for Chrome availability (third-party, conflicting dates): <https://composite.com/blog/claude-chrome-reviews-pricing-alternatives>
- Perplexity Comet free worldwide (Oct 2025): <https://www.techrepublic.com/article/news-perplexity-comet-ai-browser-free/>
- Open Interpreter desktop mode and licence (aggregators, conflicting): <https://tooldirectory.ai/tools/open-interpreter>
- OpenVoiceOS offline status (Stork, ToolMage, conflicting): <https://openvoiceos.org/?p=380>
- Leon 2.0 preview notice and MIT licence: <https://awesome.ecosyste.ms/projects/github.com%2Fleon-ai%2Fleon>
- Home Assistant Vietnamese TTS component: <https://github.com/itdonline/hass_tts_homely>
- TikTok Shop Seller Assistant (global, Vietnam rollout not confirmed): <https://www.mediapost.com/publications/article/406094/tiktoks-new-ai-chatbot-assists-in-app-sellers.html>
- Vietnam marketplace share (Investor, Jan 2026; vendor blog conflicting): <https://theinvestor.vn/vietnam-e-commerce-rivalry-intensifies-as-tiktok-shop-surges-shopee-growth-slows-d18039.html>
- Internal: `README.md`, `business-case.md` in this folder.
