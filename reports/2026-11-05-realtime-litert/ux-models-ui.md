# UX report: Models tab (dana-ui)

Audience: a non-engineer power user (KAM, beauty retail e-commerce) who wants to
pull, configure and self-host models with a few clicks. Plain-language labels,
no JSON, every failure explained with a next step.

## Screens (tab "🧩 Models", single scrolling page, top to bottom)
1. **Latency budget** - 600 ms bar split into VAD stop / STT / LLM first token /
   TTS first byte. Estimates editable (persisted in localStorage), red line at
   600 ms, "over budget by N ms" text, latest live `voice_to_first_audio`
   (falls back to `total`) from WebSocket `latency_span` events, per-stage live values.
2. **Downloads** - PullJobs: progress bar per job, bytes, files n/m, current file,
   Cancel. Polls `GET /models/jobs` every 1 s while queued/running, else 10 s.
3. **Browse litert-community** - HubBrowser: search (250 ms debounce), task chips
   (All/ASR/TTS/LLM/Vision/Embedding/Other), sort (downloads/likes/recent),
   cards (task badge, format badges, downloads, likes, 🔒 gated, "Pulled").
   Click opens a right-hand detail drawer (full-screen on phones).
4. **My models** - LocalModels table (stacked cards on phones): status, size,
   file count, loaded backend/threads/load_ms, per-row backend+threads inputs,
   Load / Unload / Use as ASR / Use as TTS / Run in browser (LLM) / Delete (confirm).
5. **LLM hosting** - LlmHosting: presets grouped Gemini / OpenAI / SGLang / Other.
6. **Run in browser (WebGPU)** - WebGpuPanel.

## Key flows
- Pull: card -> drawer (recommended files pre-checked, total size updates) ->
  backend, threads 1-16, "Load after pull" -> **Pull** or **Pull & Load** ->
  drawer closes, job appears in Downloads, "My models" refreshes when job ends.
- Assign: My models -> "Use as ASR/TTS" -> green banner lists returned `changes`
  with `restart_required` badges.
- LLM: pick preset -> (SGLang: edit base_url/model, "Check server" lists served
  models as clickable chips) -> Apply (PUT /config with `preset.patch`; for
  SGLang `llm.base_url`/`llm.model` are merged into the patch) -> Measure latency
  (POST /llm/probe) shows TTFT (budget 300 ms) and total (1500 ms): green within
  budget, amber up to 1.5x, red beyond.
- Browser LLM: needs a pulled LLM with `.task`/`.litertlm`; runtime and weights
  fetched on demand, prompt box streams output, TTFT + total shown.

## Component tree
```
App (tab state, spans) -> ModelsView (ModelsApi, refresh counters)
  LatencyBudget          (spans)
  PullJobs               (kick, onJobFinished)
  HubBrowser > DetailDrawer
  LocalModels            (onList, onRunInBrowser)
  LlmHosting
  WebGpuPanel            -> webgpu.ts (detect + lazy @mediapipe/tasks-genai)
api/models.ts (ModelsApi over ControlApi.call; fmtBytes, describeError)
```

## States and errors
- `describeError`: 403 -> "gated: set HF_TOKEN and accept the license";
  502 -> "Cannot reach Hugging Face. Allow huggingface.co / check the network";
  others show status + server `detail`.
- Banners: `role="alert"` for errors, `role="status"` for success/progress.
- Empty states for hub results, jobs, local models, no browser-capable LLM.
- WebGPU unavailable: amber banner with reason, run button disabled.
- All server calls are caught; a failed poll shows a banner and keeps retrying.

## Accessibility
- Chips use `aria-pressed`; preset cards are a `radiogroup` with `aria-checked`.
- Drawer is `role="dialog"`, closes on Esc and backdrop click; delete confirm is
  `alertdialog` with focus on Cancel (no focus trap yet).
- Progress bars have `role="progressbar"` + valuenow; gated lock has aria-label.
- Visible `:focus-visible` rings; 40 px min touch targets under 760 px; colors
  use existing light/dark tokens; status is never color-only (text badges).

## Mobile / PWA
`public/manifest.webmanifest` (standalone, theme #2f6feb), `public/icon.svg`,
linked in `index.html` with theme-color. Tabs scroll horizontally at 360 px,
tables collapse to cards, drawer goes full width, grids use auto-fit.
No service worker yet, so it is installable but not offline.

## Notes / assumptions
- Contract does not define `/models/file` auth beyond header or `?token=`; the
  WebGPU panel uses the header via fetch. The whole model is buffered in memory
  (multi-GB models may fail on low-RAM devices).
- `@mediapipe/tasks-genai` pinned 0.10.30; wasm loaded from jsDelivr (needs
  internet on the client; self-host under `public/` for air-gapped use).
- Preset grouping is by keyword in id/label/provider.
- Fixed a pre-existing unused `BotCheckData` import in App.tsx that broke `tsc`.

## Next steps
- Service worker for offline shell; focus trap in dialogs.
- Cache model bytes in OPFS/IndexedDB; stream download progress in browser panel.
- Show hub `last_modified` and license; per-model "recommended for your PC".
- Unit tests (vitest) for ModelsApi and polling cadence; live e2e vs backend.
