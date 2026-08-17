// Types mirroring the control-API contract (docs/PLAN.md §5). Kept hand-written
// and small so the UI stays dependency-light; the JSON Schema from
// GET /config/schema drives the *config form* structure at runtime.

export type EventType =
  | "state_transition"
  | "transcript_partial"
  | "transcript_final"
  | "assistant_turn"
  | "latency_span"
  | "confirmation_request"
  | "audit_entry"
  | "config_changed"
  | "mic_state"
  | "error";

// The server-push envelope. `data` is event-specific (see the payload types).
export interface Envelope<T = unknown> {
  type: EventType;
  ts: number;
  session_id: string | null;
  seq: number;
  data: T;
}

export interface StateTransition {
  from: string;
  to: string;
  cause: string;
}

export interface TranscriptData {
  role: string;
  text: string;
  lang?: string;
}

export interface AssistantTurn {
  text: string;
  interrupted: boolean;
}

export type LatencyStage = "vad" | "stt" | "llm_ttft" | "tts_ttfb" | "playback" | "total";

export interface LatencySpan {
  turn_id: string | null;
  stage: LatencyStage;
  ms: number | null;
}

export interface ConfirmationRequest {
  id: string;
  tool: string;
  args_resolved: Record<string, unknown>;
  reversibility: string;
  preview: string;
}

export interface AuditEntry {
  at?: number;
  tool: string;
  reversibility?: string;
  decision?: string;
  executed?: boolean;
  ok?: boolean | null;
  arguments?: Record<string, unknown>;
}

export interface MicState {
  muted: boolean;
}

export interface Status {
  state: string;
  session_running: boolean;
  muted: boolean;
  uptime_s: number;
  session_id: string | null;
  language: string;
  mode: string;
  providers: Record<string, string | null>;
}

export interface Capabilities {
  streaming: boolean;
  languages: string[];
  self_hosted: boolean;
  tool_calling: boolean;
  latency_class: string;
}

export interface ProviderEntry {
  name: string;
  capabilities: Capabilities | null;
}

export interface ProviderCatalog {
  [capability: string]: ProviderEntry[] | string[];
  brain_llm: string[];
}

export type ChangeStatus = "applied" | "restart_required";

export interface FieldChange {
  path: string;
  old: unknown;
  new: unknown;
  status: ChangeStatus;
}

export interface ConfigPutResult {
  changes: FieldChange[];
}

// A minimal JSON-Schema view — enough to render a form from GET /config/schema.
export interface JsonSchema {
  type?: string;
  title?: string;
  description?: string;
  properties?: Record<string, JsonSchema>;
  $defs?: Record<string, JsonSchema>;
  $ref?: string;
  enum?: unknown[];
  default?: unknown;
  minimum?: number;
  maximum?: number;
  anyOf?: JsonSchema[];
  items?: JsonSchema;
  format?: string;
  writeOnly?: boolean;
}
