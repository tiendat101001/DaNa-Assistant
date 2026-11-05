// Typed client for /api/v1/models/* and /api/v1/llm/* (team contract).
import { ControlApi } from "./client";
import type { ConfigPutResult } from "./types";

export type Task = "asr" | "tts" | "llm" | "vision" | "embedding" | "other";
export type Backend = "cpu" | "webgpu";

export interface HubModel {
  id: string;
  author: string;
  name: string;
  task: Task;
  pipeline_tag: string | null;
  tags: string[];
  downloads: number;
  likes: number;
  last_modified: string;
  gated: boolean;
  formats: string[];
  pulled: boolean;
}
export interface HubFile {
  path: string;
  size: number;
  format: "tflite" | "litertlm" | "task" | "json" | "other";
}
export type HubModelDetail = HubModel & {
  files: HubFile[];
  total_size: number;
  recommended_files: string[];
};
export interface HubList {
  author: string;
  fetched_at: number;
  cached: boolean;
  models: HubModel[];
}
export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";
export interface PullJob {
  id: string;
  repo_id: string;
  revision: string;
  status: JobStatus;
  bytes_done: number;
  bytes_total: number;
  files_done: number;
  files_total: number;
  current_file: string | null;
  error: string | null;
  started_at: number | null;
  finished_at: number | null;
  load_after: boolean;
}
export interface LocalModel {
  id: string;
  revision: string;
  task: Task;
  status: "pulling" | "ready" | "failed" | "loaded";
  files: { path: string; size: number }[];
  size_bytes: number;
  pulled_at: number;
  loaded: { backend: Backend; num_threads: number; load_ms: number; info: unknown } | null;
  error: string | null;
}
export interface PullRequest {
  repo_id: string;
  revision?: string;
  files?: string[];
  include?: string[];
  load?: boolean;
  backend?: Backend;
  num_threads?: number;
}
export interface AssignChange {
  path: string;
  old?: unknown;
  new?: unknown;
  status?: string;
}
export interface LlmPreset {
  id: string;
  label: string;
  provider: string;
  model: string;
  base_url: string | null;
  notes: string;
  expected_ttft_ms: number | null;
  patch: Record<string, unknown>;
}
export interface ProbeResult {
  ok: boolean;
  ttft_ms: number | null;
  total_ms: number | null;
  text: string;
  error: string | null;
}
export interface HealthResult {
  ok: boolean;
  models: string[];
  error: string | null;
}

export class ModelsApi {
  constructor(private api: ControlApi) {}

  get token(): string {
    return this.api.bearer;
  }

  hub(p: { search?: string; task?: string; refresh?: boolean; limit?: number }): Promise<HubList> {
    const q = new URLSearchParams();
    if (p.search) q.set("search", p.search);
    if (p.task && p.task !== "all") q.set("task", p.task);
    if (p.refresh) q.set("refresh", "true");
    if (p.limit) q.set("limit", String(p.limit));
    const qs = q.toString();
    return this.api.call("GET", `/models/hub${qs ? `?${qs}` : ""}`);
  }
  hubDetail(repoId: string): Promise<HubModelDetail> {
    return this.api.call("GET", `/models/hub/${repoId}`);
  }
  local(): Promise<{ models_dir: string; models: LocalModel[] }> {
    return this.api.call("GET", "/models/local");
  }
  pull(body: PullRequest): Promise<{ job: PullJob }> {
    return this.api.call("POST", "/models/pull", body);
  }
  jobs(): Promise<{ jobs: PullJob[] }> {
    return this.api.call("GET", "/models/jobs");
  }
  job(id: string): Promise<PullJob> {
    return this.api.call("GET", `/models/jobs/${encodeURIComponent(id)}`);
  }
  cancelJob(id: string): Promise<PullJob> {
    return this.api.call("DELETE", `/models/jobs/${encodeURIComponent(id)}`);
  }
  load(body: { repo_id: string; backend?: Backend; num_threads?: number; file?: string }): Promise<LocalModel> {
    return this.api.call("POST", "/models/load", body);
  }
  unload(repo_id: string): Promise<LocalModel> {
    return this.api.call("POST", "/models/unload", { repo_id });
  }
  remove(repoId: string): Promise<{ deleted: boolean; repo_id: string }> {
    return this.api.call("DELETE", `/models/local/${repoId}`);
  }
  assign(body: {
    role: "stt" | "tts";
    repo_id: string;
    backend?: Backend;
    num_threads?: number;
  }): Promise<{ changes: AssignChange[] }> {
    return this.api.call("POST", "/models/assign", body);
  }
  fileUrl(repoId: string, path: string): string {
    return `/api/v1/models/file?repo_id=${encodeURIComponent(repoId)}&path=${encodeURIComponent(path)}`;
  }

  presets(): Promise<{ presets: LlmPreset[] }> {
    return this.api.call("GET", "/llm/presets");
  }
  probe(body: Record<string, unknown> = {}): Promise<ProbeResult> {
    return this.api.call("POST", "/llm/probe", body);
  }
  health(baseUrl: string): Promise<HealthResult> {
    return this.api.call("GET", `/llm/health?base_url=${encodeURIComponent(baseUrl)}`);
  }
  applyPatch(patch: Record<string, unknown>): Promise<ConfigPutResult> {
    return this.api.putConfig(patch);
  }
}

export function fmtBytes(n: number): string {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i >= 2 ? 1 : 0)} ${u[i]}`;
}
export function fmtNum(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
}
// Friendly text for common failures (403 gated, 502 hub unreachable).
export function describeError(e: unknown): string {
  const status = (e as { status?: number }).status;
  const detail = (e as { detail?: unknown }).detail;
  const d = typeof detail === "string" ? ` ${detail}` : "";
  if (status === 403)
    return `This model is gated. Set HF_TOKEN on the server and accept the license on huggingface.co first.${d}`;
  if (status === 502)
    return `Cannot reach Hugging Face. Allow huggingface.co in your firewall / check the network.${d}`;
  if (status === 404) return `Not found.${d}`;
  return e instanceof Error ? `${e.message}${d}` : String(e);
}
