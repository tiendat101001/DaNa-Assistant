import { useState } from "react";
import type { ControlApi } from "../api/client";
import type { Status } from "../api/types";
import { Transcript, type TranscriptLine } from "./Transcript";

// The voice-first conversation view. Speaking stays on the machine mic (the
// agent's pipeline owns capture→STT→brain→TTS→speakers); what this adds is
// everything the CLI couldn't give:
//   - a live FSM chip (is DaNa listening?) and a Talk button that trips the
//     wake gate — no wake phrase needed, one click then just speak;
//   - the live transcript + spoken replies as a chat;
//   - a typed-turn box for when speaking is impractical — text goes through
//     the SAME pipeline (brain answers, TTS speaks it out loud).
export function TalkView({
  api,
  status,
  lines,
  connected,
  onChange,
}: {
  api: ControlApi;
  status: Status | null;
  lines: TranscriptLine[];
  connected: boolean;
  onChange: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const running = status?.session_running ?? false;
  const state = status?.state ?? "idle";
  const muted = status?.muted ?? false;

  const act = async (fn: () => Promise<unknown>, hint?: string) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChange();
      if (hint) setError(hint); // reuse the slot for a transient success hint
    } catch (e) {
      setError(e instanceof Error && e.message.includes("409")
        ? "Start the session first."
        : "Request failed. Check the agent is running.");
    } finally {
      setBusy(false);
    }
  };

  const send = (text: string) => {
    const clean = text.trim();
    if (!clean) return;
    setDraft("");
    void act(() => api.submitText(clean));
  };

  return (
    <div className="talk">
      <aside className="talk-side card">
        <div className={`state-chip ${state}`}>
          <span className="dot" aria-hidden />
          {state === "active" ? "Listening" : state === "idle" ? "Idle" : state}
        </div>
        <p className="muted small">
          {running
            ? state === "active"
              ? "Speak — DaNa is listening (no wake phrase needed)."
              : 'Say "Hey DaNa" or press Talk, then speak.'
            : "Session stopped. Start it to converse."}
        </p>

        <button
          className={`talk-btn ${state === "active" ? "live" : ""}`}
          disabled={busy || !running}
          onClick={() => act(() => api.activate(), "Listening — just speak.")}
          aria-label="Activate DaNa without the wake phrase"
        >
          🎙 Talk
        </button>

        <div className="row">
          <button disabled={busy || running} onClick={() => act(() => api.startSession())}>
            Start
          </button>
          <button
            className="secondary"
            disabled={busy || !running}
            onClick={() => act(() => api.stopSession())}
          >
            Stop
          </button>
        </div>
        <button
          className={`mic secondary ${muted ? "muted" : ""}`}
          disabled={busy}
          onClick={() => act(() => (muted ? api.unmute() : api.mute()))}
          aria-pressed={muted}
        >
          {muted ? "🔇 Unmute mic" : "🎙 Mute mic"}
        </button>

        {connected ? null : <p className="error">Live view disconnected — retrying…</p>}
        {error && (
          <p className="muted small" role="status">
            {error}
          </p>
        )}
      </aside>

      <section className="card talk-main">
        <Transcript lines={lines} />
        <form
          className="talk-input"
          onSubmit={(e) => {
            e.preventDefault();
            send(draft);
          }}
        >
          <input
            value={draft}
            placeholder={running ? "Type instead of speaking — DaNa will answer out loud…" : "Start a session to type to DaNa"}
            disabled={!running || busy}
            onChange={(e) => setDraft(e.target.value)}
            aria-label="Typed turn"
          />
          <button type="submit" disabled={!running || busy || !draft.trim()}>
            Send
          </button>
        </form>
      </section>
    </div>
  );
}
