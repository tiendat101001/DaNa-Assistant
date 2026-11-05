import { useEffect, useRef, useState } from "react";
import { type LocalModel, type ModelsApi, describeError } from "../../api/models";
import { type BrowserLlm, type GpuInfo, detectWebGpu, loadBrowserLlm } from "./webgpu";

export function WebGpuPanel({
  api,
  models,
  preselect,
}: {
  api: ModelsApi;
  models: LocalModel[];
  preselect?: string | null;
}) {
  const [gpu, setGpu] = useState<GpuInfo | null>(null);
  const [repo, setRepo] = useState<string>("");
  const [file, setFile] = useState<string>("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [llm, setLlm] = useState<BrowserLlm | null>(null);
  const [prompt, setPrompt] = useState("Say hello in one short sentence.");
  const [out, setOut] = useState("");
  const [stats, setStats] = useState<{ ttftMs: number; totalMs: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const llmRef = useRef<BrowserLlm | null>(null);

  const candidates = models.filter(
    (m) =>
      m.task === "llm" &&
      (m.status === "ready" || m.status === "loaded") &&
      m.files.some((f) => /\.(task|litertlm)$/i.test(f.path)),
  );

  useEffect(() => {
    void detectWebGpu().then(setGpu);
    return () => llmRef.current?.close();
  }, []);

  useEffect(() => {
    if (preselect) setRepo(preselect);
  }, [preselect]);

  const current = candidates.find((m) => m.id === repo) ?? candidates[0];
  const files = current?.files.filter((f) => /\.(task|litertlm)$/i.test(f.path)) ?? [];
  const chosenFile = file && files.some((f) => f.path === file) ? file : files[0]?.path;

  const start = async () => {
    if (!current || !chosenFile) return;
    setBusy(true);
    setError(null);
    llmRef.current?.close();
    setLlm(null);
    try {
      const l = await loadBrowserLlm({
        url: api.fileUrl(current.id, chosenFile),
        token: api.token,
        onStatus: setStatus,
      });
      llmRef.current = l;
      setLlm(l);
    } catch (e) {
      setStatus("");
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  const run = async () => {
    if (!llm) return;
    setBusy(true);
    setOut("");
    setStats(null);
    setError(null);
    try {
      setStats(await llm.generate(prompt, setOut));
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      {gpu === null ? (
        <p className="muted">Checking WebGPU…</p>
      ) : gpu.available ? (
        <p className="ok-text">
          ✅ WebGPU available
          {gpu.description || gpu.vendor ? ` — ${gpu.description || gpu.vendor}` : ""}
          {gpu.architecture ? ` (${gpu.architecture})` : ""}
        </p>
      ) : (
        <div className="banner warn">⚠ {gpu.reason}</div>
      )}
      {candidates.length === 0 ? (
        <p className="muted">
          Pull an LLM model with a <span className="mono">.task</span> or{" "}
          <span className="mono">.litertlm</span> file (e.g. Gemma) to run it in your browser.
        </p>
      ) : (
        <>
          <div className="form-grid">
            <label>
              Model
              <select value={current?.id} onChange={(e) => setRepo(e.target.value)}>
                {candidates.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              File
              <select value={chosenFile} onChange={(e) => setFile(e.target.value)}>
                {files.map((f) => (
                  <option key={f.path} value={f.path}>
                    {f.path}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="row">
            <button disabled={busy || !gpu?.available} onClick={() => void start()}>
              Run in browser (WebGPU)
            </button>
            {status && <span className="muted">{status}</span>}
          </div>
          {llm && (
            <div className="webgpu-chat">
              <textarea
                aria-label="Prompt"
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
              <div className="row">
                <button disabled={busy || !prompt.trim()} onClick={() => void run()}>
                  Generate
                </button>
                {stats && (
                  <span className="mono muted">
                    first token {Math.round(stats.ttftMs)} ms · total {Math.round(stats.totalMs)} ms
                  </span>
                )}
              </div>
              <pre className="output" aria-live="polite">
                {out || " "}
              </pre>
            </div>
          )}
        </>
      )}
      {error && (
        <div className="banner bad" role="alert">
          {error}
        </div>
      )}
    </div>
  );
}
