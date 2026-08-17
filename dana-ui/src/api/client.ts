// Typed REST client for the control API. Every call carries the bearer token
// (except /health). Same-origin in production (agent serves the bundle); Vite
// proxies to the agent in dev — so paths are always relative.

import type {
  AuditEntry,
  ConfigPutResult,
  JsonSchema,
  ProviderCatalog,
  Status,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: unknown,
  ) {
    super(message);
  }
}

export class ControlApi {
  constructor(private token: string) {}

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const res = await fetch(`/api/v1${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${this.token}`,
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const text = await res.text();
    const parsed = text ? JSON.parse(text) : null;
    if (!res.ok) {
      const detail = parsed && typeof parsed === "object" ? parsed.detail : undefined;
      throw new ApiError(res.status, `${method} ${path} → ${res.status}`, detail);
    }
    return parsed as T;
  }

  // Auth probe: a cheap authenticated GET to validate the token.
  async checkToken(): Promise<Status> {
    return this.request<Status>("GET", "/status");
  }

  getConfig(): Promise<Record<string, unknown>> {
    return this.request("GET", "/config");
  }

  putConfig(patch: Record<string, unknown>): Promise<ConfigPutResult> {
    return this.request("PUT", "/config", patch);
  }

  getSchema(): Promise<JsonSchema> {
    return this.request("GET", "/config/schema");
  }

  getProviders(): Promise<ProviderCatalog> {
    return this.request("GET", "/providers");
  }

  getStatus(): Promise<Status> {
    return this.request("GET", "/status");
  }

  startSession(): Promise<Status & { status: string }> {
    return this.request("POST", "/session/start");
  }

  stopSession(): Promise<Status & { status: string }> {
    return this.request("POST", "/session/stop");
  }

  mute(): Promise<{ muted: boolean }> {
    return this.request("POST", "/mic/mute");
  }

  unmute(): Promise<{ muted: boolean }> {
    return this.request("POST", "/mic/unmute");
  }

  resolveConfirmation(id: string, approved: boolean): Promise<unknown> {
    return this.request("POST", `/confirmations/${encodeURIComponent(id)}`, { approved });
  }

  getAudit(since?: number): Promise<{ entries: AuditEntry[] }> {
    const q = since !== undefined ? `?since=${since}` : "";
    return this.request("GET", `/audit${q}`);
  }
}
