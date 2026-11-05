# Team contract — realtime LiteRT / model registry / generic computer use

Owner: project lead (integration, `config.py`, `app.py`, `control/app.py`,
`pyproject.toml`, `.env.example`, README, final reports). Everyone else edits
**only the files they own** (below). Need a change elsewhere? Put it in your
final report under "Integration notes for owner" — do not edit.

## Ground rules (all coders)

- Read `/home/user/DaNa-Assistant/CLAUDE.md` first. Package root: `dana-agent/`.
- **Never run `uv sync`** (it prunes extras for the whole team). Use `uv run …`.
  If you must try a package, `uv pip install <pkg>` only if your brief allows it,
  and list it under "deps to add to pyproject".
- No state-changing git (no commit/stash/checkout/reset). `git diff`/`status` OK.
- Format/lint/type only your files: `uv run ruff format <files>`,
  `uv run ruff check <files>`, `uv run mypy <files>` (strict; Python 3.11+ code).
- Tests are offline: no network, no keys, no mic. HuggingFace is unreachable
  from this container — use `httpx.MockTransport` / fakes. Run
  `uv run pytest <your test files> --no-cov -q -p no:cacheprovider`.
- Never hardcode or log secrets. HF token = `cfg.models.hf_token` (SecretStr,
  filled from `HF_TOKEN`).
- Structured logging: `log = get_logger(__name__)`; `log.info("event", k=v)`.
- Lazy-import heavy deps (ai_edge_litert, numpy-heavy stuff, piper, openai,
  mcp, pywinauto) inside functions so the offline suite never needs them.
- Working code over breadth. Report: files, what works (test evidence), what
  is stubbed, integration notes, deps.

## Shared config (already in `src/dana_agent/config.py` — read it)

- `cfg.models: ModelsConfig` — hub_endpoint, hub_author="litert-community",
  hf_token, models_dir=".dana/models", catalog_ttl_s, default_include,
  max_concurrent_pulls, request_timeout_s.
- `cfg.conversation.latency: LatencyConfig` — target_first_audio_ms=600,
  vad_stop_secs=0.2, vad_start_secs, vad_confidence,
  tts_first_chunk_min_chars=20, warmup_on_start.
- `STTConfig`/`TTSConfig` gained `backend: "cpu"|"webgpu"`, `num_threads`,
  `model_path` (via `LocalModelFields`). For `provider="litert"`, `model` is a
  registry repo id (e.g. `litert-community/whisper-tiny`).
- `LLMConfig` gained `max_tokens`, `thinking_budget`, `reasoning_effort`,
  `timeout_s`, `extra`.
- `ActionsConfig` gained `search_engines: dict[name, url_template_with_{q}]`
  and `mcp_servers: list[MCPServerConfig]`.

## Cross-module contracts

### Model directory layout (C1 owns; C2/C4 consume)

```python
# dana_agent/models/store.py
def model_dir(models_dir: str | Path, repo_id: str) -> Path:
    """<models_dir>/<org>__<name>  e.g. .dana/models/litert-community__whisper-tiny"""
def resolve_model_path(models_dir, repo_id, *, suffixes=(".tflite",)) -> Path | None:
    """First file under model_dir(...) with one of the suffixes (largest first), or None."""
```
Manifest: `<models_dir>/registry.json` (atomic writes).

### LiteRT runtime (C2 owns `dana_agent/runtime/litert.py`; C1/C4 consume)

```python
def litert_available() -> bool
class LiteRTModel:
    path: Path; backend: str; num_threads: int; load_ms: float
    def signatures(self) -> dict[str, dict[str, list[str]]]   # {sig: {"inputs": [...], "outputs": [...]}}
    def runner(self, signature: str | None = None) -> Any      # ai_edge_litert SignatureRunner
    def info(self) -> dict[str, Any]                            # JSON-safe: path, size, tensors, load_ms
    def close(self) -> None
class ModelCache:   # process-wide; thread-safe
    def get_or_load(self, path: Path, *, backend: str = "cpu", num_threads: int = 4) -> LiteRTModel
    def unload(self, path: Path) -> bool
    def loaded(self) -> list[dict[str, Any]]
MODEL_CACHE: ModelCache
```
`backend="webgpu"` is a browser-side runtime: server-side load raises
`ValueError` — the registry instead serves the files to the UI.

### Chat-model warmup (C5 owns, C3 consumes)

```python
# dana_agent/brain/model.py
async def warmup_chat_model(model: BaseChatModel) -> float   # ms; never raises (logs + returns -1)
```

### Optional provider warmup (C2/C4 implement, C3 calls)

Providers MAY define `async def warmup(self) -> None` (load model, open
client). C3's `conversation/warmup.py` calls it duck-typed if present.

### Control API (C1 + C5 own routers; owner mounts them; UX consumes)

All under `/api/v1`, bearer auth (reuse `require_auth` from
`control/routes.py`). Routers are mounted by the owner in `control/app.py`:
`app.state.models` = `ModelManager(cfg.models)` (C1 class).

**Models (`control/models_routes.py`, router `models_api`, prefix `/api/v1/models`)**

| Method & path | Body / query | Response |
|---|---|---|
| GET `/hub` | `search?`, `task?` (asr/tts/llm/vision/embedding/other), `refresh?`, `limit?` | `{author, fetched_at, cached, models: HubModel[]}` |
| GET `/hub/{repo_id:path}` | — | `HubModel & {files: HubFile[], total_size, recommended_files: string[]}` |
| GET `/local` | — | `{models_dir, models: LocalModel[]}` |
| POST `/pull` | `{repo_id, revision?="main", files?: string[], include?: string[], load?: bool, backend?: "cpu"\|"webgpu", num_threads?: int}` | 202 `{job: PullJob}` |
| GET `/jobs` | — | `{jobs: PullJob[]}` |
| GET `/jobs/{job_id}` | — | `PullJob` |
| DELETE `/jobs/{job_id}` | — | `PullJob` (cancelled) |
| POST `/load` | `{repo_id, backend?="cpu", num_threads?=4, file?}` | `LocalModel` |
| POST `/unload` | `{repo_id}` | `LocalModel` |
| DELETE `/local/{repo_id:path}` | — | `{deleted: true, repo_id}` |
| POST `/assign` | `{role: "stt"\|"tts", repo_id, backend?, num_threads?}` | `{changes: [...]}` (applies config via controller) |
| GET `/file` | `repo_id`, `path`, auth via header **or** `?token=` | file bytes (for in-browser WebGPU runtimes) |

```ts
HubModel  = { id, author, name, task, pipeline_tag, tags: string[], downloads, likes,
              last_modified, gated: boolean, formats: string[], pulled: boolean }
HubFile   = { path, size, format }            // format: tflite|litertlm|task|json|other
PullJob   = { id, repo_id, revision, status: "queued"|"running"|"done"|"failed"|"cancelled",
              bytes_done, bytes_total, files_done, files_total, current_file, error,
              started_at, finished_at, load_after: boolean }
LocalModel= { id, revision, task, status: "pulling"|"ready"|"failed"|"loaded",
              files: {path,size}[], size_bytes, pulled_at,
              loaded: {backend, num_threads, load_ms, info} | null, error: string|null }
```

**LLM (`control/llm_routes.py`, router `llm_api`, prefix `/api/v1/llm`)** — C5

| Method & path | Body | Response |
|---|---|---|
| GET `/presets` | — | `{presets: [{id, label, provider, model, base_url, notes, expected_ttft_ms, patch}]}` (`patch` = partial AppConfig for PUT /config) |
| POST `/probe` | `{provider?, model?, base_url?, ...}` (overlays current llm cfg) | `{ok, ttft_ms, total_ms, text, error}` |
| GET `/health` | `base_url` query | `{ok, models: string[], error}` (GET `<base_url>/models`) |

## Ownership map

| Agent | Owns |
|---|---|
| C1 registry | `src/dana_agent/models/**`, `src/dana_agent/control/models_routes.py`, `tests/test_models_*.py` |
| C2 runtime+ASR | `src/dana_agent/runtime/**`, `src/dana_agent/providers/stt/litert.py`, `providers/stt/faster_whisper.py`, `providers/stt/__init__.py`, `tests/test_runtime_*.py`, `tests/test_stt_litert*.py` |
| C3 latency pipeline | `src/dana_agent/conversation/**`, `tests/test_latency_*.py`, `tests/test_conversation_processors.py` |
| C4 TTS | `src/dana_agent/providers/tts/**`, `tests/test_tts_*.py` |
| C5 LLM + MCP | `src/dana_agent/brain/providers.py`, `brain/model.py`, `brain/probe.py`, `src/dana_agent/capabilities.py`, `src/dana_agent/control/llm_routes.py`, `src/dana_agent/providers/actions/mcp.py`, `scripts/serve_sglang.*` (repo root `scripts/`), `tests/test_llm_*.py`, `tests/test_mcp_*.py` |
| C6 computer use | `src/dana_agent/providers/actions/{browser,adapters,os_win,composite}.py`, `src/dana_agent/brain/tools.py`, `tests/test_action_generic*.py`, `tests/fixtures/browser.py` |
| UX | `dana-ui/**` |
