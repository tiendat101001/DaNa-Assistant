import { useEffect, useState } from "react";
import {
  type HealthResult,
  type LlmPreset,
  type ModelsApi,
  type ProbeResult,
  describeError,
} from "../../api/models";
import type { FieldChange } from "../../api/types";

const GROUPS = ["Gemini", "OpenAI", "SGLang", "Other"] as const;
type Group = (typeof GROUPS)[number];
const TTFT_BUDGET_MS = 300;
const TOTAL_BUDGET_MS = 1500;

function groupOf(p: LlmPreset): Group {
  const k = `${p.id} ${p.label} ${p.provider}`.toLowerCase();
  if (k.includes("sglang")) return "SGLang";
  if (k.includes("gemini") || k.includes("google")) return "Gemini";
  if (k.includes("openai") || k.includes("gpt")) return "OpenAI";
  return "Other";
}
function tone(ms: number | null, budget: number): string {
  if (ms == null) return "";
  return ms <= budget ? "ok-text" : ms <= budget * 1.5 ? "warn-text" : "error";
}

export function LlmHosting({ api }: { api: ModelsApi }) {
  const [presets, setPresets] = useState<LlmPreset[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [baseUrl, setBaseUrl] = useState("http://localhost:30000/v1");
  const [model, setModel] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [changes, setChanges] = useState<FieldChange[] | null>(null);
  const [health, setHealth] = useState<HealthResult | null>(null);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    api
      .presets()
      .then((r) => setPresets(r.presets))
      .catch((e) => setError(describeError(e)));
  }, [api]);

  const preset = presets.find((p) => p.id === sel) ?? null;
  const isSglang = preset ? groupOf(preset) === "SGLang" : false;

  const choose = (p: LlmPreset) => {
    setSel(p.id);
    setChanges(null);
    setProbe(null);
    setHealth(null);
    if (p.base_url) setBaseUrl(p.base_url);
    if (p.model) setModel(p.model);
  };

  // Patch to apply: the preset's own patch, with SGLang's editable fields merged in.
  const buildPatch = (): Record<string, unknown> => {
    if (!preset) return {};
    const patch = structuredClone(preset.patch ?? {}) as Record<string, unknown>;
    if (isSglang) {
      // Presets are partial AppConfig overlays: {conversation: {llm: {...}}}.
      const conversation = { ...((patch.conversation as Record<string, unknown>) ?? {}) };
      const llm = { ...((conversation.llm as Record<string, unknown>) ?? {}) };
      llm.base_url = baseUrl;
      if (model) llm.model = model;
      conversation.llm = llm;
      patch.conversation = conversation;
    }
    return patch;
  };

  const wrap = async (name: string, fn: () => Promise<void>) => {
    setBusy(name);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(null);
    }
  };

  const apply = () =>
    wrap("apply", async () => {
      const r = await api.applyPatch(buildPatch());
      setChanges(r.changes);
    });
  const check = () =>
    wrap("health", async () => {
      setHealth(await api.health(baseUrl));
    });
  const measure = () =>
    wrap("probe", async () => {
      const body: Record<string, unknown> = preset
        ? { provider: preset.provider, model: model || preset.model }
        : {};
      if (preset && (isSglang || preset.base_url)) body.base_url = baseUrl || preset.base_url;
      setProbe(await api.probe(body));
    });

  return (
    <div>
      {error && (
        <div className="banner bad" role="alert">
          {error}
        </div>
      )}
      {presets.length === 0 && !error && <p className="muted">Loading presets…</p>}
      {GROUPS.map((g) => {
        const items = presets.filter((p) => groupOf(p) === g);
        if (!items.length) return null;
        return (
          <section key={g}>
            <h3>{g}</h3>
            <div className="model-grid" role="radiogroup" aria-label={`${g} presets`}>
              {items.map((p) => (
                <button
                  key={p.id}
                  role="radio"
                  aria-checked={sel === p.id}
                  className={sel === p.id ? "model-card selected" : "model-card"}
                  onClick={() => choose(p)}
                >
                  <strong>{p.label}</strong>
                  <div className="muted mono">{p.model}</div>
                  <div className="muted">{p.notes}</div>
                  {p.expected_ttft_ms != null && (
                    <span className="badge fmt">~{p.expected_ttft_ms} ms first token</span>
                  )}
                </button>
              ))}
            </div>
          </section>
        );
      })}

      {preset && (
        <div className="llm-actions">
          {isSglang && (
            <div className="form-grid">
              <label>
                Server address (base_url)
                <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
              </label>
              <label>
                Model name
                <input value={model} onChange={(e) => setModel(e.target.value)} />
              </label>
            </div>
          )}
          <div className="row">
            <button disabled={busy !== null} onClick={() => void apply()}>
              Apply “{preset.label}”
            </button>
            {isSglang && (
              <button className="secondary" disabled={busy !== null} onClick={() => void check()}>
                {busy === "health" ? "Checking…" : "Check server"}
              </button>
            )}
            <button className="secondary" disabled={busy !== null} onClick={() => void measure()}>
              {busy === "probe" ? "Measuring…" : "Measure latency"}
            </button>
          </div>
          {health && (
            <div className={health.ok ? "banner ok" : "banner bad"} role="status">
              {health.ok
                ? `Server reachable. Models: ${health.models.join(", ") || "none listed"}`
                : `Server not reachable: ${health.error ?? "unknown error"}`}
              {health.ok && health.models.length > 0 && (
                <div className="row">
                  {health.models.slice(0, 6).map((m) => (
                    <button key={m} className="chip" onClick={() => setModel(m)}>
                      {m}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {probe && (
            <div className={probe.ok ? "banner ok" : "banner bad"} role="status">
              {probe.ok ? (
                <>
                  First token:{" "}
                  <strong className={tone(probe.ttft_ms, TTFT_BUDGET_MS)}>
                    {Math.round(probe.ttft_ms ?? 0)} ms
                  </strong>{" "}
                  (budget {TTFT_BUDGET_MS}) · Total:{" "}
                  <strong className={tone(probe.total_ms, TOTAL_BUDGET_MS)}>
                    {Math.round(probe.total_ms ?? 0)} ms
                  </strong>
                  {probe.text && <div className="muted">“{probe.text}”</div>}
                </>
              ) : (
                `Probe failed: ${probe.error ?? "unknown error"}`
              )}
            </div>
          )}
          {changes && (
            <div className="banner ok" role="status">
              <strong>Applied.</strong>
              <ul>
                {changes.map((c) => (
                  <li key={c.path}>
                    <span className="mono">{c.path}</span> → {String(c.new)}{" "}
                    {c.status === "restart_required" && (
                      <span className="badge warn">restart required</span>
                    )}
                  </li>
                ))}
                {changes.length === 0 && <li>No changes (already set).</li>}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
