# Research R2: Latency budget, self-hosted LLMs, and generic computer-use standards

Date: 2026-11-05. Author: Research Agent R2.

**Research limits (read first).** The primary documentation hosts (docs.pipecat.ai, docs.langchain.com, docs.sglang.ai) did not resolve from this session, so every claim about them comes from search-result snippets, not from fetched pages. Items marked **UNVERIFIED** are single-source, vendor-reported, third-party, or from memory. Vendor latency numbers are marketing unless stated otherwise. Nothing here has been measured on DaNa hardware. The session's network allow-list needs `docs.pipecat.ai`, `docs.langchain.com` and `docs.sglang.ai` added before a follow-up pass can confirm the exact parameter names.

---

## 1. Executive summary

1. The 600 ms target (end of user speech to first audio byte) is tight. Sum the stages and it lands near 700-800 ms unless stages overlap. Overlap is the main lever.
2. Biggest single cost is usually end-of-speech detection. Pipecat's VAD `stop_secs` drives it. Current docs (snippet) say the default is 0.2 s; an older guide says 0.8 s. Check the pinned version's `VADParams` source.
3. Recommended endpointing: Silero VAD with `stop_secs=0.2`, plus Pipecat's local Smart Turn v3 analyzer (ONNX, no network) as the turn-stop strategy.
4. STT: AssemblyAI streaming (vendor claim: ~300 ms P50 to immutable final) is a reasonable choice. Deepgram Flux has built-in turn detection; Nova-3 relies on traditional endpointing. Both are vendor-reported.
5. LLM: disable reasoning/thinking. Gemini 2.5 Flash-family: `thinking_budget=0` (Gemini 2.5 Pro cannot disable thinking, and 0 returns 400). Gemini 3+: `thinking_level` (UNVERIFIED for LangChain). OpenAI GPT-5 family: `reasoning_effort="minimal"`. gpt-4.1-nano has the best short-prompt TTFT in one community report (~176 ms, UNVERIFIED).
6. TTS: Cartesia Sonic Turbo and ElevenLabs Flash v2.5 are the strongest vendor-reported streaming options (about 40-90 ms model-level; third-party measurements of 200-350 ms p50 time-to-first-audio). Gemini TTS has no verified TTFB figure. Piper or Kokoro locally is the privacy/cost option, with TTFB unmeasured here.
7. Realistic 600 ms combo: Silero stop 0.2 s + AssemblyAI streaming + Gemini 2.5 Flash-Lite (thinking off) or gpt-4.1-nano + Cartesia Sonic Turbo, with eager clause-level TTS flush and pre-warmed connections. Expect ~650-750 ms on a good network until measured. Hitting 600 ms reliably needs overlap, not just faster models.
8. SGLang runs on Linux (WSL2 likely, UNVERIFIED) with NVIDIA GPUs, and has a documented CPU backend for Intel AMX Xeons (4th gen and newer). Native Windows is not supported per community sources. It is an option for a self-hosted brain, not a 600 ms path on a laptop.
9. Computer use: standardize on an accessibility-tree snapshot with element refs (Playwright MCP style), exposed to the brain as typed MCP tools with annotations, behind DaNa's own safety gate. Use UI Automation (pywinauto or uiautomation) for desktop apps. Fall back to vision only when the tree is empty.
10. Annotations (`readOnlyHint`, `destructiveHint`, etc.) are advisory only. The DaNa gate must classify actions itself and never trust server-supplied hints.

---

## 2. Latency budget

Target: end of user speech to first audio byte <= 600 ms. "Current" figures are not measured for DaNa; the sources here are used for targets only, so the current column is left as "not measured".

| Stage | Current (DaNa) | Target | How to get there |
|---|---|---|---|
| (1) End-of-speech detection (VAD stop + turn decision) | not measured; default `stop_secs` may be 0.8 s on older pins | 150-200 ms | Silero `stop_secs=0.2`; Smart Turn v3 local analyzer to decide turn end earlier (model inference cost is small but not benchmarked here) |
| (2) STT finalization after end of speech | not measured | 200-300 ms | AssemblyAI streaming immutable final (vendor P50 ~300 ms); keep the WebSocket open (pre-warm) |
| (3) LLM time to first token (thinking off) | not measured | 100-200 ms | Gemini 2.5 Flash-Lite with `thinking_budget=0`, or gpt-4.1-nano / gpt-5-nano with minimal reasoning; short system prompt; prompt caching |
| (4) TTS time to first audio byte | not measured | 50-100 ms model, 100-150 ms end-to-end | Cartesia Sonic Turbo or ElevenLabs Flash v2.5; send first clause as soon as it exists |
| (5) Transport + audio output buffer | not measured | 30-50 ms | Co-locate region; small output buffer; WASAPI low-latency output (UNVERIFIED for DaNa's audio stack) |
| **Sum (no overlap)** | | **~530-800 ms** | |

Why the sum is not the answer: stages (1) and (2) overlap, because STT is already producing text while silence is being measured. Stages (3) and (4) overlap, because TTS starts on the first clause while the LLM keeps streaming. With that overlap, a ~600-700 ms result is realistic. 600 ms with zero slack is not, unless the LLM is very fast and the network is local.

Measure before optimizing: instrument each stage with timestamps (Pipecat's per-service TTFB/TTFS metrics, if enabled in the pinned version, are UNVERIFIED for this repo).

---

## 3. Stage details and sources

### 3.1 End-of-speech detection

- Pipecat `SileroVADAnalyzer` takes `VADParams` (`confidence`, `start_secs`, `stop_secs`, `min_volume`). The current API reference snippet says `stop_secs` defaults to **0.2** ("seconds of silence required before transitioning back to QUIET"). An older speech-input guide says **0.8**. Treat the installed version's source as authoritative. Source: https://docs.pipecat.ai/api-reference/server/services/vad/silero-vad-analyzer.md (via search snippet)
- Pipecat's docs say built-in STT P99 latency values assume `stop_secs=0.2`. If you change `stop_secs`, pass a measured TTFS P99 to the STT service via `ttfs_p99_latency`. The strategy warns if `stop_secs` >= the STT P99, which collapses the STT wait timeout to 0 s. Source: https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-turn-strategies (snippet)
- Smart Turn v3: `LocalSmartTurnAnalyzerV3` (`pipecat.audio.turn.smart_turn.local_smart_turn_v3`), ONNX, runs locally with no network. Usage: `TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())`. The reference says the bundled model is `smart-turn-v3.2-cpu` when no model path is set, so the default may now be v3.2. Source: https://reference-server.pipecat.ai/en/stable/api/pipecat.audio.turn.smart_turn.local_smart_turn_v3.html (snippet). Model size ~8.7 MB (third-party LiveKit adapter, smart-turn-livekit 0.3.0). No first-party inference-latency number was found. UNVERIFIED.

### 3.2 STT finalization

- AssemblyAI Universal-Streaming: vendor claims ~300 ms P50 to immutable final, with a neural end-of-turn model (`end_of_turn=true`). Vendor comparison: Nova-3 median 516 ms, P99 1,907 ms vs AssemblyAI 307 ms / 1,012 ms. All vendor-reported. Source: https://www.assemblyai.com/solutions/voice-agents (snippet). Note: AssemblyAI's pages now promote a newer "Universal-3 Pro Streaming" model. Confirm the exact model id in the current docs before setting config. UNVERIFIED.
- Deepgram: Nova-3 uses traditional endpointing. Flux has built-in turn detection. Source: https://dev.to/martschweiger/assemblyai-vs-deepgram-for-voice-agents-4e0b (secondary, cross-posted vendor content).
- No independent benchmark was found. Run your own test with the same audio.

### 3.3 LLM time to first token (thinking off)

- **Gemini 2.5 Flash / Flash-Lite, disable thinking.** The forum evidence is that `thinking_budget=0` disables thinking on `gemini-2.5-flash` (one HF project passes it through LangChain). Gemini 2.5 Pro rejects 0 with 400 INVALID_ARGUMENT. Some preview builds still think a little at 0. Sources: https://discuss.ai.google.dev/t/gemini-2-5-flash-preview-04-17-not-honoring-thinking-budget-0/80165 (snippet); https://discuss.google.dev/t/how-can-i-set-0-thinkingbudget-with-vertex-ai-gemini-2-5flash-i-wanna-disable-it/189244 (snippet).
- **LangChain parameter.** `ChatGoogleGenerativeAI(model="gemini-2.5-flash", thinking_budget=0)`. A Tessl mirror of the langchain-google-genai docs lists `thinking_budget: Optional[int] = None` and `include_thoughts`. The SAP AI SDK wrapper docs say `thinking_budget` is deprecated for Gemini 3+ in favor of `thinking_level`. UNVERIFIED against LangChain's own docs (fetch failed). Mirror: https://tessl.io/registry/tessl/pypi-langchain-google-genai/files/docs/chat-models.md
- **OpenAI minimal reasoning.** `reasoning_effort="minimal"` is a valid GPT-5 setting (added to speed up responses; default is medium). gpt-5-nano accepts it on Chat Completions; community reports show empty output unless `max_completion_tokens` is raised. Responses API uses `reasoning={"effort": "minimal"}`. In langchain-openai, pass `reasoning_effort="minimal"` to `ChatOpenAI(model="gpt-5-nano", ...)` or through `model_kwargs`. UNVERIFIED for the pinned langchain-openai version. Sources: https://community.openai.com/t/what-is-going-on-with-the-gpt-5-api/1338030?page=3 ; https://community.openai.com/t/gpt-5-nano-accepted-parameters/1355086 (snippets)
- **Model TTFT numbers.**
  - gpt-4.1-nano: ~176 ms to first token on short prompts, ~5 s at 600K+ tokens (community Reddit snapshot, unverified). https://reddit.sentinel-team.org/posts/1ti1zb0/snapshots/2026-05-22T18%3A40%3A12.467942Z
  - gemini-2.5-flash-lite: described as among the lowest-latency models mid-2026, but no first-token number found. One source's 5.7 tok/s figure looks wrong. UNVERIFIED.
  - gemini-2.5-flash-preview-tts: no TTFB found.
- Model ids to use: `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gpt-4.1-nano`, `gpt-5-nano`. Confirm each is available on the account before wiring.

### 3.4 TTS time to first audio

| Provider / model | Vendor claim | Third-party | Notes |
|---|---|---|---|
| Cartesia Sonic Turbo | ~40 ms model latency | — | Cartesia's own test: P90 TTFA 128-135 ms across regions. Vendor-run. https://cartesia.ai/vs/cartesia-vs-elevenlabs |
| Cartesia Sonic 2 | ~90 ms model latency | — | Codesota (Apr 2026) calls Sonic 2 latency leader. |
| ElevenLabs Flash v2.5 | ~75 ms model inference (32 languages) | p50 TTFA ~280-350 ms (DEV Community) | Excludes network and app time. Strong Vietnamese coverage claimed (UNVERIFIED). |
| Gemini 2.5 Flash preview TTS (`gemini-2.5-flash-preview-tts`) | none found | end-to-end p50 ~4 s on an aggregator for full response (not TTFB) | Not a streaming-latency candidate on current evidence. |
| OpenAI `gpt-4o-mini-tts` | not researched | | UNVERIFIED |
| Piper / Kokoro (local) | not researched | | Local, no network; latency depends on GPU/CPU. UNVERIFIED |

Conclusion for TTS: Cartesia Sonic Turbo is the best-supported low-latency option on current evidence. Gemini TTS is unproven for this use.

### 3.5 Pipeline overlap and techniques (top 8)

1. **Eager clause-level TTS flush.** Send the first clause (about 3-8 words, or at punctuation) to TTS as soon as the LLM emits it. Pipecat's TTS services stream, so the work is in the sentence aggregator. Biggest single win after endpointing.
2. **Pre-warm connections.** Open the STT WebSocket, LLM HTTP/2 session and TTS WebSocket at startup or at wake, so the first turn does not pay TLS/handshake cost (typically 100+ ms).
3. **Disable reasoning/thinking** on every turn (section 3.3). Use a separate reasoning model only for tool planning, if at all.
4. **Stable short system prompt** with prompt caching. OpenAI caches long shared prefixes automatically. Gemini has implicit caching on recent models (UNVERIFIED specifics). Keep the prefix byte-identical across turns.
5. **Endpointing tuned for speed**: Silero `stop_secs=0.2` plus Smart Turn v3. Re-measure STT P99 and set `ttfs_p99_latency` accordingly.
6. **Stream tokens into the TTS aggregator**, never wait for the full answer. This is already how Pipecat works; verify DaNa's brain does not buffer LangGraph output (the `TokenDelta` path).
7. **Keep tool calls off the fast path.** A tool round-trip adds a full LLM call. For simple turns, answer directly; route tool-heavy work to a second turn with an acknowledgement ("on it") spoken immediately.
8. **Co-locate and measure.** Pick the cloud region nearest the user for STT/LLM/TTS (Asia, US or EU, per vendor regions). Log p50/p95 per stage on real audio. Optimize only the measured bottleneck.

### 3.6 Which combination meets 600 ms

- Most plausible: Silero `stop_secs=0.2` + Smart Turn v3 + AssemblyAI streaming + Gemini 2.5 Flash-Lite (thinking off) or gpt-4.1-nano + Cartesia Sonic Turbo, with techniques 1-3 in place. Expect roughly 550-700 ms on a good connection. This is an estimate, not a measurement.
- Less likely: any combo using ElevenLabs third-party-measured TTFA (~280-350 ms) plus a non-streaming LLM. That path sits above 600 ms.
- Self-hosted SGLang on a consumer GPU is possible, but it adds a local model step. Its TTFT on an RTX 3060/4060 for a 1-3B model is UNVERIFIED (not researched in this pass).

---

## 4. SGLang for lightweight self-hosted LLMs

- **Install**: a third-party skill page shows a source install (`pip install -e "python[all]"` from a clone). The official install page did not resolve. The pip package `sglang[all]` is the likely route. UNVERIFIED. Source: https://skillselion.com/skills/ovachiever/droid-tings/sglang (secondary)
- **Launch command** (from docs snippet): `python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --tool-call-parser qwen25 --host 0.0.0.0`
  - A newer docs build recommends `sglang serve` as the entrypoint and deprecates `qwen25` in favor of `qwen`. Use `--tool-call-parser qwen` on current versions.
  - Other parsers listed include `qwen3_coder`, `step3`, a DeepSeek parser, a GPT-OSS parser, and `pythonic` for models that emit Python-style calls. Source: https://docs.sglang.io/advanced_features/tool_parser.html (snippet)
  - Flags named in the request (`--mem-fraction-static`, `--enable-torch-compile`, `--cuda-graph`, `--attention-backend`, `--port 30000`): the port default of 30000 and the flag names are UNVERIFIED in this pass. Check `python -m sglang.launch_server --help` on the installed version.
- **Endpoint**: OpenAI-compatible at `http://<host>:<port>/v1` (`/v1/chat/completions`). Use the OpenAI client with `base_url` pointed there, then call the models list to get the id. Tool calling works through the parser above.
- **Platforms**:
  - Linux with NVIDIA GPU: the primary path.
  - Native Windows: not supported (community sources).
  - WSL2 with NVIDIA GPU: reported working in a September 2026 Reddit post, with performance caveats. UNVERIFIED.
  - CPU: the official CPU backend supports Intel Xeon with AMX (4th gen Scalable and newer). Use the Docker `xeon` image (`docker/xeon.Dockerfile`) or build from source with `SGLANG_USE_CPU_ENGINE=1`, and `SGLANG_CPU_OMP_THREADS_BIND` for core placement. Dense models around 10B or fewer are the documented sizing guidance. Source: https://docs.sglang.ai/references/cpu.html (snippet). No AMD or consumer-CPU support is documented here.
- **Candidate small models (Vietnamese ability: UNVERIFIED for all; no benchmark run)**:
  - Qwen2.5-1.5B-Instruct / Qwen3-1.7B: Qwen family has broad multilingual training including Vietnamese; tool-call parser `qwen`.
  - Llama-3.2-1B / 3B-Instruct: good English, weaker Vietnamese; parser depends on version.
  - gemma-3-1b-it: small, multilingual claims, weaker tool use.
  - Expected TTFT on RTX 3060 / 4060 for these sizes: typically tens of ms for the first token on short prompts at fp16 or 4-bit. Not measured here. UNVERIFIED.
- **Recommendation**: keep the self-hosted model as an optional privacy path, not the default. The cloud fast path (Gemini Flash-Lite / gpt-4.1-nano) is more likely to meet 600 ms with less engineering.

---

## 5. Computer-use standards

### 5.1 Model Context Protocol (MCP)

- **Tool annotations** (MCP tools spec, via SDK docs snippets):
  - `readOnlyHint` default **false**
  - `destructiveHint` default **true** (only meaningful when not read-only)
  - `idempotentHint` default **false**
  - `openWorldHint` default **true** (true = touches external/unbounded world, e.g. web; false = closed, e.g. memory)
  - Annotations are advisory. Clients must not make security decisions from annotations supplied by an untrusted server. Sources: https://kotlin.sdk.modelcontextprotocol.io/kotlin-sdk-core/io.modelcontextprotocol.kotlin.sdk.types/-tool-annotations/index.html ; https://csharp.sdk.modelcontextprotocol.io/v1/api/ModelContextProtocol.Protocol.ToolAnnotations.html
- **Python SDK client** (from memory, not re-verified in this session): `mcp.ClientSession` over `mcp.client.stdio.stdio_client(StdioServerParameters(command=..., args=[...]))`; `await session.initialize()`, `await session.list_tools()`, `await session.call_tool(name, arguments)`. UNVERIFIED for current SDK version.
- **langchain-mcp-adapters**: converts MCP tools to LangChain tools for a LangGraph agent (`MultiServerMCPClient`). Plausible fit for DaNa's brain. UNVERIFIED; not fetched.

### 5.2 Playwright MCP (accessibility-snapshot approach)

- Tools return an accessibility-tree snapshot, not a screenshot, and every interacting tool returns refs like `e10` (frame-prefixed `f2e7` where relevant). A ref is valid until the page changes; stale refs fail with "Ref ... not found in the current page snapshot", so re-snapshot.
- `browser_click` takes the ref. The parameter name changed across versions (`ref` in older READMEs, `target` in current docs). Check the installed schema. Source: https://playwright.dev/mcp/snapshots (snippet)
- `browser_find` returns only matching snapshot nodes plus context, which is cheaper than a full snapshot on large pages.
- `--vision` switches to screenshot mode. Default is the aria snapshot.
- Known issue (Cursor forum): some clients receive metadata without refs. Test the snapshot round-trip in DaNa's MCP bridge.

### 5.3 browser-use DOM indexing

Not researched in this pass (out of time budget). UNVERIFIED. Expected: indexes interactive DOM elements and gives the LLM numbered targets. It is an alternative to the Playwright ref model with a similar idea.

### 5.4 Windows desktop (UI Automation)

- **pywinauto**: UIA backend (preferred), with win32 fallback. Best-known Python option.
- **uiautomation** (Python package): direct UIA wrapper. Not separately verified here.
- Windows MCP servers found (all need Python 3.12+ and must run in the interactive desktop session):
  - `windows-gui-mcp` (dcl632): UIA first, then pywinauto win32, pyautogui coordinates last. https://pypi.org/project/windows-gui-mcp/
  - `uiautomation-mcp`: UIA inspection, element find, actions, screenshots. https://pypi.org/project/uiautomation-mcp/
  - `pywinauto-mcp` (sandraschi): consolidated tools, OCR optional. Its listing is marked "Unvalidated" and mentions an opt-in keylogger and face-recognition tools. Do not adopt without reviewing its code and safety docs.
- Recommendation: write DaNa's own thin UIA tool layer (snapshot = UIA tree with stable IDs, act-by-ref) using pywinauto. Use third-party MCP servers as references, not as dependencies.

### 5.5 Vision-based computer-use models

- **Gemini 2.5 Computer Use** (`gemini-2.5-computer-use-preview-10-2025`): preview from October 2025, browser-focused (no desktop or local-file control). One source says it was retired July 28, 2026 in favor of a newer Flash model. Single source; verify on Google's deprecation page. Vendor benchmarks only. UNVERIFIED.
- **Anthropic computer use**: full-desktop environment. One secondary source lists Claude Sonnet 4.6 at 72.7% OSWorld (Feb 2026). UNVERIFIED.
- **OpenAI `computer-use-preview`**: Responses API only. One source reports 38.1% OSWorld (2025, probably superseded). UNVERIFIED.
- Pricing claims from an aggregator are unverified; do not budget on them.
- **When to fall back to vision**: only when the accessibility/UIA tree is empty or unusable (canvas apps, games, custom-rendered UI, images, unlabeled controls). Vision is slower and more expensive per step and less deterministic. Keep it as a last-resort tool (`screenshot` + coordinate click) behind the same gate.

---

## 6. Recommended architecture

```
voice turn -> brain (LangGraph, LLM with thinking off)
                 |
                 v  typed tool calls only (no raw shell/eval)
          MCP bridge (stdio clients, one per server)
                 |  tools: snapshot(), click(ref), type(ref,text),
                 |         select(ref,value), navigate(url), press(key), screenshot()  [fallback]
                 v
          DaNa safety gate (non-LLM)
             - classify each action itself (read / write / destructive / out-of-scope)
             - ignore server annotations as authority (advisory only)
             - read-only -> ALLOW; destructive -> CONFIRM (voice); out-of-scope -> REFUSE
                 |
                 v
          executor: Playwright (browser) or pywinauto/UIA (desktop apps)
                 |
                 v
          append-only audit log (every decision + result)
```

Key points:
- **Generic snapshot + ref** replaces hard-coded per-site CSS selectors. Playwright ARIA snapshot for web, UIA tree for desktop; both give stable refs for act-by-ref.
- **Stale-ref handling**: on "ref not found", the executor re-snapshots and returns a structured error so the brain retries once.
- **Speak while acting**: the brain should emit a short acknowledgement before a multi-step action, so latency is hidden.
- **Safety**: all confirmations go through the voice layer. Hard refusals for out-of-scope apps (banking, password managers, admin consoles unless allow-listed).
- **Browser-first, desktop second**: Playwright covers most needs; UIA for native apps once the bridge is stable.

---

## 7. Open items and next steps

1. Allow `docs.pipecat.ai`, `docs.langchain.com`, `docs.sglang.ai` in the session network settings, then re-verify: VADParams defaults in the pinned Pipecat version, `ChatGoogleGenerativeAI` thinking params, `ChatOpenAI` reasoning param, SGLang flags and port.
2. Measure TTFB per provider from the target region on real DaNa audio (STT final, LLM TTFT, TTS TTFA). Decide the combo from measurements, not from vendor numbers.
3. Confirm current availability and deprecations of `gemini-2.5-flash-lite`, `gpt-4.1-nano`, `gpt-5-nano`, and Gemini TTS / Computer Use model ids.
4. Prototype the snapshot-and-ref bridge on one browser task and one Windows app task before choosing an MCP server dependency.
5. Review `windows-gui-mcp` and `uiautomation-mcp` source before reuse.
