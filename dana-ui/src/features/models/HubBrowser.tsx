import { useCallback, useEffect, useState } from "react";
import {
  type Backend,
  type HubModel,
  type HubModelDetail,
  type ModelsApi,
  describeError,
  fmtBytes,
  fmtNum,
} from "../../api/models";

const TASKS = ["all", "asr", "tts", "llm", "vision", "embedding", "other"] as const;
const SORTS = [
  { id: "downloads", label: "Most downloaded" },
  { id: "likes", label: "Most liked" },
  { id: "recent", label: "Recently updated" },
] as const;
type Sort = (typeof SORTS)[number]["id"];

export function HubBrowser({
  api,
  onPulled,
  refreshKey,
}: {
  api: ModelsApi;
  onPulled: () => void;
  refreshKey: number;
}) {
  const [search, setSearch] = useState("");
  const [task, setTask] = useState<string>("all");
  const [sort, setSort] = useState<Sort>("downloads");
  const [models, setModels] = useState<HubModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(
    async (refresh = false) => {
      setLoading(true);
      setError(null);
      try {
        const r = await api.hub({ search: search.trim() || undefined, task, refresh, limit: 100 });
        setModels(r.models);
      } catch (e) {
        setError(describeError(e));
      } finally {
        setLoading(false);
      }
    },
    [api, search, task],
  );

  useEffect(() => {
    const t = setTimeout(() => void load(), 250);
    return () => clearTimeout(t);
  }, [load, refreshKey]);

  const sorted = [...models].sort((a, b) =>
    sort === "downloads"
      ? b.downloads - a.downloads
      : sort === "likes"
        ? b.likes - a.likes
        : b.last_modified.localeCompare(a.last_modified),
  );

  return (
    <div className="hub">
      <div className="row hub-controls">
        <input
          type="search"
          className="grow"
          placeholder="Search litert-community models…"
          aria-label="Search models"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select aria-label="Sort models" value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
          {SORTS.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
        <button className="secondary" onClick={() => void load(true)} disabled={loading}>
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>
      <div className="chips" role="group" aria-label="Filter by task">
        {TASKS.map((t) => (
          <button
            key={t}
            className={task === t ? "chip active" : "chip"}
            aria-pressed={task === t}
            onClick={() => setTask(t)}
          >
            {t === "all" ? "All" : t === "asr" || t === "tts" || t === "llm" ? t.toUpperCase() : cap(t)}
          </button>
        ))}
      </div>
      {error && (
        <div className="banner bad" role="alert">
          {error}
        </div>
      )}
      {loading && sorted.length === 0 && !error && (
        <p className="muted" role="status">
          Loading catalog from huggingface.co (first load can take ~15 s)…
        </p>
      )}
      {!error && !loading && sorted.length === 0 && <p className="muted">No models match.</p>}
      <div className="model-grid">
        {sorted.map((m) => (
          <button key={m.id} className="model-card" onClick={() => setSelected(m.id)}>
            <div className="mc-head">
              <strong className="mc-name">{m.name}</strong>
              {m.gated && (
                <span
                  className="lock"
                  title="Gated: needs HF_TOKEN + accepted license"
                  aria-label="Gated model: needs HF_TOKEN and accepted license"
                >
                  🔒
                </span>
              )}
            </div>
            <div className="badges">
              <span className={`badge task-${m.task}`}>{m.task.toUpperCase()}</span>
              {m.formats.map((f) => (
                <span key={f} className="badge fmt">
                  {f}
                </span>
              ))}
              {m.pulled && <span className="badge ok">Pulled</span>}
            </div>
            <div className="muted mc-stats">
              ⬇ {fmtNum(m.downloads)} · ♥ {fmtNum(m.likes)}
            </div>
          </button>
        ))}
      </div>
      {selected && (
        <DetailDrawer
          api={api}
          repoId={selected}
          onClose={() => setSelected(null)}
          onPulled={() => {
            onPulled();
            setSelected(null);
          }}
        />
      )}
    </div>
  );
}

function DetailDrawer({
  api,
  repoId,
  onClose,
  onPulled,
}: {
  api: ModelsApi;
  repoId: string;
  onClose: () => void;
  onPulled: () => void;
}) {
  const [detail, setDetail] = useState<HubModelDetail | null>(null);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [backend, setBackend] = useState<Backend>("cpu");
  const [threads, setThreads] = useState(4);
  const [loadAfter, setLoadAfter] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let live = true;
    api
      .hubDetail(repoId)
      .then((d) => {
        if (!live) return;
        setDetail(d);
        setChecked(new Set(d.recommended_files));
      })
      .catch((e) => live && setError(describeError(e)));
    return () => {
      live = false;
    };
  }, [api, repoId]);

  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);

  const total = detail
    ? detail.files.filter((f) => checked.has(f.path)).reduce((a, f) => a + f.size, 0)
    : 0;

  const toggle = (p: string) =>
    setChecked((prev) => {
      const n = new Set(prev);
      if (n.has(p)) n.delete(p);
      else n.add(p);
      return n;
    });

  const doPull = async (load: boolean) => {
    setBusy(true);
    setError(null);
    try {
      await api.pull({
        repo_id: repoId,
        files: [...checked],
        load,
        backend,
        num_threads: threads,
      });
      onPulled();
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label={`Model ${repoId}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="row between">
          <h3 className="drawer-title">{repoId}</h3>
          <button className="secondary" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        {error && (
          <div className="banner bad" role="alert">
            {error}
          </div>
        )}
        {detail?.gated && (
          <div className="banner warn">
            🔒 Gated model — needs HF_TOKEN on the server and an accepted license.
          </div>
        )}
        {!detail && !error && <p className="muted">Loading files…</p>}
        {detail && (
          <>
            <div className="badges">
              <span className={`badge task-${detail.task}`}>{detail.task.toUpperCase()}</span>
              {detail.pulled && <span className="badge ok">Pulled</span>}
              <span className="muted">
                ⬇ {fmtNum(detail.downloads)} · ♥ {fmtNum(detail.likes)}
              </span>
            </div>
            <ul className="filelist">
              {detail.files.map((f) => (
                <li key={f.path}>
                  <label>
                    <input
                      type="checkbox"
                      checked={checked.has(f.path)}
                      onChange={() => toggle(f.path)}
                    />
                    <span className="mono fname">{f.path}</span>
                    <span className="badge fmt">{f.format}</span>
                    <span className="muted">{fmtBytes(f.size)}</span>
                  </label>
                </li>
              ))}
            </ul>
            <p>
              Selected: <strong>{checked.size}</strong> files · <strong>{fmtBytes(total)}</strong>{" "}
              <span className="muted">(all files: {fmtBytes(detail.total_size)})</span>
            </p>
            <div className="form-grid">
              <label>
                Backend
                <select value={backend} onChange={(e) => setBackend(e.target.value as Backend)}>
                  <option value="cpu">CPU (this PC)</option>
                  <option value="webgpu">WebGPU (browser)</option>
                </select>
              </label>
              <label>
                Threads: {threads}
                <input
                  type="range"
                  min={1}
                  max={16}
                  value={threads}
                  onChange={(e) => setThreads(Number(e.target.value))}
                />
              </label>
            </div>
            <label className="check">
              <input
                type="checkbox"
                checked={loadAfter}
                onChange={(e) => setLoadAfter(e.target.checked)}
              />
              Load after pull
            </label>
            <div className="row">
              <button disabled={busy || checked.size === 0} onClick={() => void doPull(loadAfter)}>
                Pull
              </button>
              <button
                className="secondary"
                disabled={busy || checked.size === 0}
                onClick={() => void doPull(true)}
              >
                Pull &amp; Load
              </button>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
