import type { Status } from "../api/types";

export function StatusBar({
  status,
  connected,
}: {
  status: Status | null;
  connected: boolean;
}) {
  return (
    <div className="statusbar">
      <span className={`dot ${connected ? "ok" : "bad"}`} title={connected ? "live" : "offline"} />
      <span className="pill">{connected ? "live" : "reconnecting…"}</span>
      {status && (
        <>
          <span className="pill">{status.state}</span>
          <span className="pill">{status.session_running ? "session on" : "session off"}</span>
          <span className={`pill ${status.muted ? "warn" : ""}`}>
            {status.muted ? "mic muted" : "mic live"}
          </span>
          <span className="pill">{status.mode}</span>
          <span className="pill">{status.language}</span>
          <span className="pill">up {Math.round(status.uptime_s)}s</span>
        </>
      )}
    </div>
  );
}
