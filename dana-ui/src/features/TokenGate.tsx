import { useState } from "react";
import { ControlApi } from "../api/client";

// First screen: the user pastes the bearer token from the agent's local token
// file (default .dana/control-token). We validate it against GET /status before
// entering the dashboard.
export function TokenGate({
  initial,
  onAuthed,
}: {
  initial: string;
  onAuthed: (token: string) => void;
}) {
  const [value, setValue] = useState(initial);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await new ControlApi(value.trim()).checkToken();
      onAuthed(value.trim());
    } catch {
      setError("Token rejected. Check .dana/control-token and that the agent is running.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="gate">
      <form className="card gate-card" onSubmit={submit}>
        <h1>DaNa</h1>
        <p className="muted">
          Paste the control token from <code>dana-agent/.dana/control-token</code>. The agent
          generates it automatically the first time you start it with <code>--control</code>.
        </p>
        <label className="field-name" htmlFor="dana-token">
          Control token
        </label>
        <input
          id="dana-token"
          type="password"
          placeholder="bearer token"
          value={value}
          autoFocus
          autoComplete="off"
          onChange={(e) => setValue(e.target.value)}
        />
        <button type="submit" disabled={busy || !value.trim()}>
          {busy ? "Checking…" : "Connect"}
        </button>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </form>
    </div>
  );
}
