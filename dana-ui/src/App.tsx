import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ControlApi } from "./api/client";
import { EventStream } from "./api/ws";
import type {
  AssistantTurn,
  AuditEntry,
  ConfirmationRequest,
  Envelope,
  LatencySpan,
  StateTransition,
  Status,
  TranscriptData,
} from "./api/types";
import { TokenGate } from "./features/TokenGate";
import { StatusBar } from "./features/StatusBar";
import { SessionControls } from "./features/SessionControls";
import { MicToggle } from "./features/MicToggle";
import { TalkView } from "./features/TalkView";
import { Transcript, type TranscriptLine } from "./features/Transcript";
import { FsmView } from "./features/FsmView";
import { LatencyWaterfall } from "./features/LatencyWaterfall";
import { AuditView } from "./features/AuditView";
import { ConfigEditor } from "./features/ConfigEditor";
import { ConfirmationPrompt } from "./features/ConfirmationPrompt";

const TOKEN_KEY = "dana.control.token";
const MAX_LINES = 200;
const MAX_SPANS = 40;

export function App() {
  const [token, setToken] = useState<string>(() => localStorage.getItem(TOKEN_KEY) ?? "");
  const [authed, setAuthed] = useState(false);
  const [connected, setConnected] = useState(false);
  const [tab, setTab] = useState<"talk" | "dashboard">("talk");

  const [status, setStatus] = useState<Status | null>(null);
  const [transitions, setTransitions] = useState<StateTransition[]>([]);
  const [lines, setLines] = useState<TranscriptLine[]>([]);
  const [spans, setSpans] = useState<LatencySpan[]>([]);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [confirmations, setConfirmations] = useState<ConfirmationRequest[]>([]);

  const api = useMemo(() => (token ? new ControlApi(token) : null), [token]);
  const streamRef = useRef<EventStream | null>(null);

  const refreshStatus = useCallback(async () => {
    if (!api) return;
    try {
      setStatus(await api.getStatus());
    } catch {
      /* transient */
    }
  }, [api]);

  const onEnvelope = useCallback((env: Envelope) => {
    switch (env.type) {
      case "state_transition": {
        const d = env.data as StateTransition;
        setTransitions((prev) => [d, ...prev].slice(0, 30));
        setStatus((s) => (s ? { ...s, state: d.to } : s));
        break;
      }
      case "transcript_partial":
      case "transcript_final": {
        const d = env.data as TranscriptData;
        const final = env.type === "transcript_final";
        setLines((prev) => appendTranscript(prev, d, final));
        break;
      }
      case "assistant_turn": {
        const d = env.data as AssistantTurn;
        setLines((prev) =>
          [...prev, { role: "assistant", text: d.text, final: true }].slice(-MAX_LINES),
        );
        break;
      }
      case "latency_span": {
        const d = env.data as LatencySpan;
        setSpans((prev) => [d, ...prev].slice(0, MAX_SPANS));
        break;
      }
      case "audit_entry": {
        setAudit((prev) => [env.data as AuditEntry, ...prev].slice(0, 100));
        break;
      }
      case "confirmation_request": {
        const d = env.data as ConfirmationRequest;
        setConfirmations((prev) => [...prev.filter((c) => c.id !== d.id), d]);
        break;
      }
      case "mic_state": {
        const muted = (env.data as { muted: boolean }).muted;
        setStatus((s) => (s ? { ...s, muted } : s));
        break;
      }
      case "config_changed":
        void refreshStatus();
        break;
    }
  }, [refreshStatus]);

  // Connect the event stream + load initial data once authenticated.
  useEffect(() => {
    if (!authed || !api) return;
    const stream = new EventStream(token, onEnvelope, setConnected);
    stream.start();
    streamRef.current = stream;
    void refreshStatus();
    api.getAudit().then((r) => setAudit(r.entries.slice().reverse())).catch(() => {});
    const poll = setInterval(refreshStatus, 5000);
    return () => {
      stream.stop();
      clearInterval(poll);
    };
  }, [authed, api, token, onEnvelope, refreshStatus]);

  const handleAuthed = useCallback(
    (t: string) => {
      localStorage.setItem(TOKEN_KEY, t);
      setToken(t);
      setAuthed(true);
    },
    [],
  );

  const resolveConfirmation = useCallback(
    async (id: string, approved: boolean) => {
      if (!api) return;
      await api.resolveConfirmation(id, approved).catch(() => {});
      setConfirmations((prev) => prev.filter((c) => c.id !== id));
    },
    [api],
  );

  if (!authed || !api) {
    return <TokenGate initial={token} onAuthed={handleAuthed} />;
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>DaNa</h1>
        <nav className="tabs" aria-label="Views">
          <button
            className={tab === "talk" ? "tab active" : "tab"}
            onClick={() => setTab("talk")}
            aria-current={tab === "talk"}
          >
            🎙 Talk
          </button>
          <button
            className={tab === "dashboard" ? "tab active" : "tab"}
            onClick={() => setTab("dashboard")}
            aria-current={tab === "dashboard"}
          >
            🛠 Dashboard
          </button>
        </nav>
        <StatusBar status={status} connected={connected} />
      </header>

      {confirmations.length > 0 && (
        <ConfirmationPrompt items={confirmations} onResolve={resolveConfirmation} />
      )}

      {tab === "talk" ? (
        <main className="talk-wrap">
          <TalkView
            api={api}
            status={status}
            lines={lines}
            connected={connected}
            onChange={refreshStatus}
          />
        </main>
      ) : (
        <main className="grid">
          <section className="card controls">
            <h2>Session</h2>
            <SessionControls api={api} onChange={refreshStatus} running={status?.session_running} />
            <MicToggle api={api} muted={status?.muted ?? false} />
          </section>

          <section className="card">
            <h2>State machine</h2>
            <FsmView state={status?.state ?? "idle"} transitions={transitions} />
          </section>

          <section className="card wide">
            <h2>Transcript</h2>
            <Transcript lines={lines} />
          </section>

          <section className="card wide">
            <h2>Latency waterfall</h2>
            <LatencyWaterfall spans={spans} />
          </section>

          <section className="card wide">
            <h2>Configuration</h2>
            <ConfigEditor api={api} />
          </section>

          <section className="card wide">
            <h2>Audit trail</h2>
            <AuditView entries={audit} />
          </section>
        </main>
      )}
    </div>
  );
}

function appendTranscript(
  prev: TranscriptLine[],
  d: TranscriptData,
  final: boolean,
): TranscriptLine[] {
  const next = prev.slice();
  const last = next[next.length - 1];
  // Fold a partial-then-final of the same role into a single updating line.
  if (last && last.role === d.role && !last.final) {
    next[next.length - 1] = { role: d.role, text: d.text, final, lang: d.lang };
  } else {
    next.push({ role: d.role, text: d.text, final, lang: d.lang });
  }
  return next.slice(-MAX_LINES);
}
