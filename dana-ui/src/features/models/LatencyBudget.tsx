import { useEffect, useState } from "react";
import type { LatencySpan } from "../../api/types";

export const BUDGET_MS = 600;
const KEY = "dana.latency.estimates";

interface Stage {
  id: string;
  label: string;
  color: string;
  span: string; // latency_span stage to read live values from
}
const STAGES: Stage[] = [
  { id: "vad", label: "End-of-speech (VAD stop)", color: "var(--c1)", span: "vad" },
  { id: "stt", label: "STT", color: "var(--c2)", span: "stt" },
  { id: "llm", label: "LLM first token", color: "var(--c3)", span: "llm_ttft" },
  { id: "tts", label: "TTS first byte", color: "var(--c4)", span: "tts_ttfb" },
];
const DEFAULTS: Record<string, number> = { vad: 200, stt: 150, llm: 150, tts: 100 };

export function useEstimates(): [Record<string, number>, (id: string, v: number) => void] {
  const [est, setEst] = useState<Record<string, number>>(() => {
    try {
      return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
    } catch {
      return { ...DEFAULTS };
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(est));
    } catch {
      /* ignore */
    }
  }, [est]);
  return [est, (id, v) => setEst((p) => ({ ...p, [id]: v }))];
}

export function LatencyBudget({ spans = [] }: { spans?: LatencySpan[] }) {
  const [est, setEst] = useEstimates();
  const sum = STAGES.reduce((a, s) => a + (est[s.id] ?? 0), 0);
  const scale = Math.max(BUDGET_MS, sum);
  const live = (name: string): number | null => {
    const s = spans.find((x) => (x.stage as string) === name && x.ms != null);
    return s?.ms ?? null;
  };
  const liveTotal = live("voice_to_first_audio") ?? live("total");
  const over = sum > BUDGET_MS;

  return (
    <div className="budget">
      <p className="muted">
        Goal: the assistant starts speaking within <strong>{BUDGET_MS} ms</strong> of you finishing
        a sentence. Edit the estimates to see where the time goes.
      </p>
      <div
        className="budget-bar"
        role="img"
        aria-label={`Estimated ${sum} ms of ${BUDGET_MS} ms budget`}
      >
        {STAGES.map((s) => (
          <div
            key={s.id}
            className="budget-seg"
            style={{ width: `${((est[s.id] ?? 0) / scale) * 100}%`, background: s.color }}
            title={`${s.label}: ${est[s.id]} ms`}
          />
        ))}
        <div className="budget-line" style={{ left: `${(BUDGET_MS / scale) * 100}%` }} />
      </div>
      <p className={over ? "error" : "ok-text"}>
        Estimated total: <strong>{sum} ms</strong> {over ? `(${sum - BUDGET_MS} ms over budget)` : "— within budget"}
      </p>
      <div className="budget-grid">
        {STAGES.map((s) => {
          const l = live(s.span);
          return (
            <label key={s.id} className="budget-field">
              <span>
                <span className="swatch" style={{ background: s.color }} /> {s.label}
              </span>
              <span className="row">
                <input
                  type="number"
                  min={0}
                  max={5000}
                  step={10}
                  value={est[s.id]}
                  className="num"
                  onChange={(e) => setEst(s.id, Math.max(0, Number(e.target.value) || 0))}
                />
                <span className="muted">ms</span>
                {l != null && <span className="mono muted">live: {Math.round(l)}</span>}
              </span>
            </label>
          );
        })}
      </div>
      <p>
        Latest measured voice → first audio:{" "}
        {liveTotal != null ? (
          <strong className={liveTotal <= BUDGET_MS ? "ok-text" : "error"}>
            {Math.round(liveTotal)} ms
          </strong>
        ) : (
          <span className="muted">no turn yet</span>
        )}
      </p>
    </div>
  );
}
