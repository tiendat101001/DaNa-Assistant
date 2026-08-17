import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, ControlApi } from "../api/client";
import type { FieldChange, JsonSchema, ProviderCatalog, ProviderEntry } from "../api/types";

// A schema-driven config form: the structure comes from GET /config/schema and
// the values from GET /config, so a new provider or field never needs a UI edit.
// `provider` fields become capability-aware dropdowns fed by GET /providers.

const REDACTED = "**********";

// Map a dotted config path to the provider-catalog key for its dropdown.
function catalogKeyFor(path: string): string | null {
  const p = path.replace(/\.provider$/, "");
  if (p === "vad") return "vad";
  if (p === "wake") return "wake";
  if (p === "actions") return "actions";
  if (p === "conversation.stt") return "stt";
  if (p === "conversation.tts") return "tts";
  if (p === "conversation.s2s") return "s2s";
  if (p === "conversation.llm") return "brain_llm";
  return null;
}

export function ConfigEditor({ api }: { api: ControlApi }) {
  const [schema, setSchema] = useState<JsonSchema | null>(null);
  const [original, setOriginal] = useState<Record<string, unknown> | null>(null);
  const [draft, setDraft] = useState<Record<string, unknown> | null>(null);
  const [catalog, setCatalog] = useState<ProviderCatalog | null>(null);
  const [changes, setChanges] = useState<FieldChange[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    const [sc, cfg, cat] = await Promise.all([
      api.getSchema(),
      api.getConfig(),
      api.getProviders(),
    ]);
    setSchema(sc);
    setOriginal(cfg);
    setDraft(structuredClone(cfg));
    setCatalog(cat);
  }, [api]);

  useEffect(() => {
    load().catch((e) => setError(String(e)));
  }, [load]);

  const providerOptions = useCallback(
    (path: string): string[] | null => {
      const key = catalogKeyFor(path);
      if (!key || !catalog) return null;
      const entry = catalog[key];
      if (!entry) return null;
      if (Array.isArray(entry) && typeof entry[0] === "string") return entry as string[];
      return (entry as ProviderEntry[]).map((e) => e.name);
    },
    [catalog],
  );

  const setPath = useCallback((path: string, value: unknown) => {
    setDraft((prev) => {
      if (!prev) return prev;
      const next = structuredClone(prev);
      const parts = path.split(".");
      let obj = next as Record<string, unknown>;
      for (const part of parts.slice(0, -1)) obj = obj[part] as Record<string, unknown>;
      obj[parts[parts.length - 1]] = value;
      return next;
    });
  }, []);

  const patch = useMemo(() => {
    if (!original || !draft) return {};
    return diff(original, draft);
  }, [original, draft]);

  const dirty = Object.keys(patch).length > 0;

  const save = async () => {
    setBusy(true);
    setError(null);
    setChanges(null);
    try {
      const res = await api.putConfig(patch);
      setChanges(res.changes);
      await load(); // reflect applied fields
    } catch (e) {
      if (e instanceof ApiError && e.status === 422) {
        setError(`Validation failed — ${formatValidationErrors(e.detail)}`);
      } else if (e instanceof ApiError) {
        setError(`Save failed (${e.status}). Is the agent still running?`);
      } else {
        setError(String(e));
      }
    } finally {
      setBusy(false);
    }
  };

  if (error && !schema) return <p className="error">{error}</p>;
  if (!schema || !draft) return <p className="muted">Loading config…</p>;

  return (
    <div className="config">
      <SchemaObject
        schema={schema}
        root={schema}
        value={draft}
        path=""
        setPath={setPath}
        providerOptions={providerOptions}
      />
      <div className="config-actions">
        <button disabled={!dirty || busy} onClick={save}>
          {busy ? "Saving…" : "Save changes"}
        </button>
        <button
          className="secondary"
          disabled={!dirty || busy}
          onClick={() => original && setDraft(structuredClone(original))}
        >
          Reset
        </button>
        {dirty && <span className="muted">{Object.keys(patch).length} field(s) changed</span>}
      </div>
      {error && <p className="error">{error}</p>}
      {changes && <ChangeSummary changes={changes} />}
    </div>
  );
}

function ChangeSummary({ changes }: { changes: FieldChange[] }) {
  if (changes.length === 0) return <p className="muted">No changes.</p>;
  return (
    <ul className="changes">
      {changes.map((c) => (
        <li key={c.path}>
          <span className="mono">{c.path}</span>{" "}
          <span className={`tag ${c.status}`}>{c.status.replace("_", " ")}</span>
        </li>
      ))}
    </ul>
  );
}

// --- schema-driven rendering ---------------------------------------------

function resolveRef(schema: JsonSchema, root: JsonSchema): JsonSchema {
  if (!schema.$ref) return schema;
  const name = schema.$ref.split("/").pop()!;
  return root.$defs?.[name] ?? schema;
}

function SchemaObject({
  schema,
  root,
  value,
  path,
  setPath,
  providerOptions,
}: {
  schema: JsonSchema;
  root: JsonSchema;
  value: Record<string, unknown>;
  path: string;
  setPath: (path: string, value: unknown) => void;
  providerOptions: (path: string) => string[] | null;
}) {
  const resolved = resolveRef(schema, root);
  const props = resolved.properties ?? {};
  return (
    <div className={path ? "config-group" : ""}>
      {path && <h3>{path.split(".").pop()}</h3>}
      {Object.entries(props).map(([key, propSchema]) => {
        const childPath = path ? `${path}.${key}` : key;
        const child = resolveRef(propSchema, root);
        const current = (value?.[key] ?? undefined) as unknown;
        if (child.properties) {
          return (
            <SchemaObject
              key={childPath}
              schema={child}
              root={root}
              value={(current ?? {}) as Record<string, unknown>}
              path={childPath}
              setPath={setPath}
              providerOptions={providerOptions}
            />
          );
        }
        return (
          <Leaf
            key={childPath}
            name={key}
            path={childPath}
            schema={propSchema}
            root={root}
            value={current}
            setPath={setPath}
            options={key === "provider" ? providerOptions(childPath) : null}
          />
        );
      })}
    </div>
  );
}

function Leaf({
  name,
  path,
  schema,
  root,
  value,
  setPath,
  options,
}: {
  name: string;
  path: string;
  schema: JsonSchema;
  root: JsonSchema;
  value: unknown;
  setPath: (path: string, value: unknown) => void;
  options: string[] | null;
}) {
  const s = resolveRef(schema, root);
  const enumVals = s.enum ?? collectEnum(schema, root);
  const isSecret = value === REDACTED;
  const type = inferType(s, schema);

  return (
    <label className="field">
      <span className="field-name">{name}</span>
      {options ? (
        <select value={String(value ?? "")} onChange={(e) => setPath(path, e.target.value)}>
          {options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      ) : enumVals ? (
        <select value={String(value ?? "")} onChange={(e) => setPath(path, e.target.value)}>
          {enumVals.map((o) => (
            <option key={String(o)} value={String(o)}>
              {String(o)}
            </option>
          ))}
        </select>
      ) : type === "boolean" ? (
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => setPath(path, e.target.checked)}
        />
      ) : isSecret ? (
        <input
          type="password"
          placeholder="•••••••• (unchanged)"
          onChange={(e) => setPath(path, e.target.value || REDACTED)}
        />
      ) : type === "array" ? (
        <input
          type="text"
          value={Array.isArray(value) ? value.join(", ") : ""}
          placeholder="comma,separated"
          onChange={(e) =>
            setPath(
              path,
              e.target.value
                .split(",")
                .map((x) => x.trim())
                .filter(Boolean),
            )
          }
        />
      ) : type === "number" || type === "integer" ? (
        <input
          type="number"
          value={value == null ? "" : Number(value)}
          onChange={(e) => setPath(path, e.target.value === "" ? null : Number(e.target.value))}
        />
      ) : (
        <input
          type="text"
          value={value == null ? "" : String(value)}
          onChange={(e) => setPath(path, e.target.value)}
        />
      )}
    </label>
  );
}

function inferType(s: JsonSchema, raw: JsonSchema): string {
  if (s.type) return s.type;
  if (raw.type) return raw.type;
  // anyOf: [ {type}, {type: null} ] — take the first concrete type.
  const any = s.anyOf ?? raw.anyOf;
  if (any) {
    const concrete = any.find((x) => x.type && x.type !== "null");
    if (concrete?.type) return concrete.type;
  }
  return "string";
}

function collectEnum(schema: JsonSchema, root: JsonSchema): unknown[] | null {
  const s = resolveRef(schema, root);
  if (s.enum) return s.enum;
  const any = s.anyOf ?? schema.anyOf;
  if (any) {
    for (const branch of any) {
      const r = resolveRef(branch, root);
      if (r.enum) return r.enum;
    }
  }
  return null;
}

// Minimal recursive diff → the smallest patch object of changed leaves.
function diff(
  original: Record<string, unknown>,
  draft: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(draft)) {
    const a = original[key];
    const b = draft[key];
    if (isObject(a) && isObject(b)) {
      const sub = diff(a, b);
      if (Object.keys(sub).length > 0) out[key] = sub;
    } else if (JSON.stringify(a) !== JSON.stringify(b)) {
      // Never send back a still-redacted secret (nothing actually changed).
      if (b === REDACTED) continue;
      out[key] = b;
    }
  }
  return out;
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

// Render pydantic's 422 error list as readable "path: message" lines instead of
// raw JSON, so a bad value points the user at the offending field.
function formatValidationErrors(detail: unknown): string {
  if (Array.isArray(detail)) {
    return detail
      .map((err) => {
        const e = err as { loc?: unknown[]; msg?: string };
        const loc = Array.isArray(e.loc) ? e.loc.filter((p) => p !== "body").join(".") : "";
        return loc ? `${loc}: ${e.msg ?? "invalid"}` : (e.msg ?? "invalid");
      })
      .join("; ");
  }
  return typeof detail === "string" ? detail : JSON.stringify(detail);
}
