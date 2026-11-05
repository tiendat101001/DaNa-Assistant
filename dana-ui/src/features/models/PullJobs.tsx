import { useEffect, useRef, useState } from "react";
import { type ModelsApi, type PullJob, describeError, fmtBytes } from "../../api/models";

const ACTIVE = new Set(["queued", "running"]);

export function PullJobs({
  api,
  kick,
  onJobFinished,
}: {
  api: ModelsApi;
  kick: number;
  onJobFinished: () => void;
}) {
  const [jobs, setJobs] = useState<PullJob[]>([]);
  const [error, setError] = useState<string | null>(null);
  const prevActive = useRef<Set<string>>(new Set());
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let live = true;
    const tick = async () => {
      let delay = 10000;
      try {
        const { jobs: js } = await api.jobs();
        if (!live) return;
        setJobs(js);
        setError(null);
        const active = new Set(js.filter((j) => ACTIVE.has(j.status)).map((j) => j.id));
        for (const id of prevActive.current) if (!active.has(id)) onJobFinished();
        prevActive.current = active;
        if (active.size > 0) delay = 1000;
      } catch (e) {
        if (live) setError(describeError(e));
      }
      if (live) timer.current = setTimeout(() => void tick(), delay);
    };
    void tick();
    return () => {
      live = false;
      if (timer.current) clearTimeout(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, kick]);

  const cancel = async (id: string) => {
    try {
      const j = await api.cancelJob(id);
      setJobs((p) => p.map((x) => (x.id === id ? j : x)));
    } catch (e) {
      setError(describeError(e));
    }
  };

  if (jobs.length === 0 && !error) return <p className="muted">No downloads yet.</p>;

  return (
    <div className="jobs">
      {error && (
        <div className="banner bad" role="alert">
          {error}
        </div>
      )}
      {jobs.map((j) => {
        const pct = j.bytes_total > 0 ? Math.min(100, (j.bytes_done / j.bytes_total) * 100) : 0;
        return (
          <div key={j.id} className="job">
            <div className="row between">
              <strong>{j.repo_id}</strong>
              <span className={`badge st-${j.status}`}>{j.status}</span>
            </div>
            <div
              className="progress"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(pct)}
              aria-label={`Download progress for ${j.repo_id}`}
            >
              <div className="progress-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="row between muted">
              <span>
                {fmtBytes(j.bytes_done)} / {fmtBytes(j.bytes_total)} · file {j.files_done}/
                {j.files_total}
                {j.current_file && ACTIVE.has(j.status) ? ` · ${j.current_file}` : ""}
              </span>
              {ACTIVE.has(j.status) && (
                <button className="secondary" onClick={() => void cancel(j.id)}>
                  Cancel
                </button>
              )}
            </div>
            {j.error && <div className="error">{j.error}</div>}
          </div>
        );
      })}
    </div>
  );
}
