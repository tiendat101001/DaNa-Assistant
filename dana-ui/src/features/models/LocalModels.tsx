import { useEffect, useState } from "react";
import {
  type AssignChange,
  type Backend,
  type LocalModel,
  type ModelsApi,
  describeError,
  fmtBytes,
} from "../../api/models";

export function LocalModels({
  api,
  refreshKey,
  onChanged,
  onRunInBrowser,
  onList,
}: {
  api: ModelsApi;
  refreshKey: number;
  onChanged: () => void;
  onRunInBrowser?: (m: LocalModel) => void;
  onList?: (m: LocalModel[]) => void;
}) {
  const [models, setModels] = useState<LocalModel[]>([]);
  const [dir, setDir] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ title: string; changes: AssignChange[] } | null>(null);
  const [opts, setOpts] = useState<Record<string, { backend: Backend; threads: number }>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmDel, setConfirmDel] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const r = await api.local();
      setModels(r.models);
      onList?.(r.models);
      setDir(r.models_dir);
      setError(null);
    } catch (e) {
      setError(describeError(e));
    }
  };
  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, refreshKey]);

  const opt = (id: string) => opts[id] ?? { backend: "cpu" as Backend, threads: 4 };
  const setOpt = (id: string, v: Partial<{ backend: Backend; threads: number }>) =>
    setOpts((p) => ({ ...p, [id]: { ...opt(id), ...v } }));

  const run = async (id: string, fn: () => Promise<void>) => {
    setBusy(id);
    setError(null);
    try {
      await fn();
      await refresh();
      onChanged();
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div>
      {dir && <p className="muted mono">{dir}</p>}
      {error && (
        <div className="banner bad" role="alert">
          {error}
        </div>
      )}
      {notice && (
        <div className="banner ok" role="status">
          <strong>{notice.title}</strong>
          <ul>
            {notice.changes.map((c, i) => (
              <li key={i}>
                <span className="mono">{c.path}</span>: {String(c.old ?? "—")} → {String(c.new ?? "—")}{" "}
                {c.status === "restart_required" ? (
                  <span className="badge warn">restart required</span>
                ) : (
                  <span className="badge ok">{c.status ?? "applied"}</span>
                )}
              </li>
            ))}
          </ul>
          <button className="secondary" onClick={() => setNotice(null)}>
            Dismiss
          </button>
        </div>
      )}
      {models.length === 0 ? (
        <p className="muted">Nothing pulled yet. Browse the Hub and click Pull.</p>
      ) : (
        <div className="table-wrap">
          <table className="tbl">
            <thead>
              <tr>
                <th>Model</th>
                <th>Status</th>
                <th>Size</th>
                <th>Files</th>
                <th>Loaded</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const o = opt(m.id);
                const loaded = m.status === "loaded" && m.loaded;
                const canAssign = m.status === "ready" || m.status === "loaded";
                return (
                  <tr key={m.id}>
                    <td data-label="Model">
                      <strong>{m.id}</strong>
                      <div>
                        <span className={`badge task-${m.task}`}>{m.task.toUpperCase()}</span>
                      </div>
                      {m.error && <div className="error">{m.error}</div>}
                    </td>
                    <td data-label="Status">
                      <span className={`badge st-${m.status}`}>{m.status}</span>
                    </td>
                    <td data-label="Size">{fmtBytes(m.size_bytes)}</td>
                    <td data-label="Files" title={m.files.map((f) => f.path).join("\n")}>
                      {m.files.length}
                    </td>
                    <td data-label="Loaded" className="mono">
                      {loaded && m.loaded
                        ? `${m.loaded.backend} · ${m.loaded.num_threads}t · ${Math.round(m.loaded.load_ms)} ms`
                        : "—"}
                    </td>
                    <td data-label="Actions">
                      <div className="row">
                        <select
                          aria-label="Backend"
                          value={o.backend}
                          onChange={(e) => setOpt(m.id, { backend: e.target.value as Backend })}
                        >
                          <option value="cpu">CPU</option>
                          <option value="webgpu">WebGPU</option>
                        </select>
                        <input
                          type="number"
                          aria-label="Threads"
                          min={1}
                          max={16}
                          value={o.threads}
                          className="num"
                          onChange={(e) =>
                            setOpt(m.id, { threads: Math.max(1, Math.min(16, Number(e.target.value) || 1)) })
                          }
                        />
                      </div>
                      <div className="row">
                        <button
                          disabled={busy === m.id || !canAssign}
                          onClick={() =>
                            void run(m.id, async () => {
                              await api.load({ repo_id: m.id, backend: o.backend, num_threads: o.threads });
                            })
                          }
                        >
                          Load
                        </button>
                        <button
                          className="secondary"
                          disabled={busy === m.id || !loaded}
                          onClick={() => void run(m.id, async () => void (await api.unload(m.id)))}
                        >
                          Unload
                        </button>
                        {(["stt", "tts"] as const).map((role) => (
                          <button
                            key={role}
                            className="secondary"
                            disabled={busy === m.id || !canAssign}
                            onClick={() =>
                              void run(m.id, async () => {
                                const r = await api.assign({
                                  role,
                                  repo_id: m.id,
                                  backend: o.backend,
                                  num_threads: o.threads,
                                });
                                setNotice({
                                  title: `${m.id} is now the ${role === "stt" ? "ASR" : "TTS"} model`,
                                  changes: r.changes,
                                });
                              })
                            }
                          >
                            Use as {role === "stt" ? "ASR" : "TTS"}
                          </button>
                        ))}
                        {onRunInBrowser && m.task === "llm" && canAssign && (
                          <button className="secondary" onClick={() => onRunInBrowser(m)}>
                            Run in browser
                          </button>
                        )}
                        <button
                          className="danger"
                          disabled={busy === m.id}
                          onClick={() => setConfirmDel(m.id)}
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {confirmDel && (
        <div className="drawer-backdrop" onClick={() => setConfirmDel(null)}>
          <div
            className="dialog"
            role="alertdialog"
            aria-modal="true"
            aria-label="Confirm delete"
            onClick={(e) => e.stopPropagation()}
          >
            <p>
              Delete <strong>{confirmDel}</strong> and all its files from disk? You can pull it again
              later.
            </p>
            <div className="row">
              <button
                className="danger"
                onClick={() => {
                  const id = confirmDel;
                  setConfirmDel(null);
                  void run(id, async () => void (await api.remove(id)));
                }}
              >
                Delete
              </button>
              <button className="secondary" autoFocus onClick={() => setConfirmDel(null)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
