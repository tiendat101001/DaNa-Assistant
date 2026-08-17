import type { StateTransition } from "../api/types";

const STATES = ["idle", "active"];

export function FsmView({
  state,
  transitions,
}: {
  state: string;
  transitions: StateTransition[];
}) {
  return (
    <div>
      <div className="fsm">
        {STATES.map((s) => (
          <div key={s} className={`node ${state === s ? "current" : ""}`}>
            {s}
          </div>
        ))}
      </div>
      <ul className="edges">
        {transitions.slice(0, 8).map((t, i) => (
          <li key={i}>
            <span className="mono">{t.from}</span> → <span className="mono">{t.to}</span>
            <span className="muted"> ({t.cause})</span>
          </li>
        ))}
        {transitions.length === 0 && <li className="muted">No transitions yet.</li>}
      </ul>
    </div>
  );
}
