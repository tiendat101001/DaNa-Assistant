import { useMemo } from "react";
import type { LatencySpan, LatencyStage } from "../api/types";

const STAGE_ORDER: LatencyStage[] = ["vad", "stt", "llm_ttft", "tts_ttfb", "playback"];
const STAGE_COLOR: Record<string, string> = {
  vad: "var(--c1)",
  stt: "var(--c2)",
  llm_ttft: "var(--c3)",
  tts_ttfb: "var(--c4)",
  playback: "var(--c5)",
  total: "var(--c6)",
};

interface TurnRow {
  turn_id: string;
  stages: Record<string, number>;
  total: number;
}

// Group latency spans by turn_id and render each turn as a stacked bar of its
// per-stage milliseconds (docs/PLAN.md §7).
export function LatencyWaterfall({ spans }: { spans: LatencySpan[] }) {
  const rows = useMemo(() => groupByTurn(spans), [spans]);
  const max = Math.max(1, ...rows.map((r) => r.total));

  if (rows.length === 0) {
    return <p className="muted">No latency data yet.</p>;
  }

  return (
    <div className="waterfall">
      {rows.map((row) => (
        <div key={row.turn_id} className="wf-row">
          <span className="wf-id mono">{row.turn_id}</span>
          <div className="wf-bar">
            {STAGE_ORDER.filter((s) => row.stages[s] != null).map((s) => (
              <div
                key={s}
                className="wf-seg"
                title={`${s}: ${row.stages[s]} ms`}
                style={{
                  width: `${(row.stages[s] / max) * 100}%`,
                  background: STAGE_COLOR[s],
                }}
              >
                {row.stages[s] > max * 0.08 ? s : ""}
              </div>
            ))}
          </div>
          <span className="wf-total mono">{row.total} ms</span>
        </div>
      ))}
      <div className="legend">
        {STAGE_ORDER.map((s) => (
          <span key={s} className="legend-item">
            <span className="swatch" style={{ background: STAGE_COLOR[s] }} /> {s}
          </span>
        ))}
      </div>
    </div>
  );
}

function groupByTurn(spans: LatencySpan[]): TurnRow[] {
  const map = new Map<string, TurnRow>();
  for (const span of spans) {
    const id = span.turn_id ?? "—";
    const row = map.get(id) ?? { turn_id: id, stages: {}, total: 0 };
    if (span.ms != null) {
      if (span.stage === "total") {
        row.total = span.ms;
      } else {
        row.stages[span.stage] = span.ms;
      }
    }
    map.set(id, row);
  }
  const rows = [...map.values()];
  for (const r of rows) {
    if (!r.total) r.total = Object.values(r.stages).reduce((a, b) => a + b, 0);
  }
  return rows.slice(0, 12);
}
