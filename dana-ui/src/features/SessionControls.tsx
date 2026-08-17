import { useState } from "react";
import { ControlApi } from "../api/client";

export function SessionControls({
  api,
  running,
  onChange,
}: {
  api: ControlApi;
  running: boolean | undefined;
  onChange: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChange();
    } catch {
      setError("Request failed. Check the agent is running.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="row">
        <button disabled={busy || running} onClick={() => act(() => api.startSession())}>
          {busy ? "…" : "Start"}
        </button>
        <button
          className="secondary"
          disabled={busy || !running}
          onClick={() => act(() => api.stopSession())}
        >
          Stop
        </button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}
