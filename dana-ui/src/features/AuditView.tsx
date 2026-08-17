import type { AuditEntry } from "../api/types";

export function AuditView({ entries }: { entries: AuditEntry[] }) {
  if (entries.length === 0) {
    return <p className="muted">No actions recorded.</p>;
  }
  return (
    <div className="table-wrap">
      <table className="audit">
        <thead>
          <tr>
            <th>time</th>
            <th>tool</th>
            <th>class</th>
            <th>decision</th>
            <th>exec</th>
            <th>ok</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e, i) => (
            <tr key={i}>
              <td className="mono">{e.at ? new Date(e.at * 1000).toLocaleTimeString() : "—"}</td>
              <td>{e.tool}</td>
              <td>{e.reversibility ?? "—"}</td>
              <td>
                <span className={`tag ${e.decision ?? ""}`}>{e.decision ?? "—"}</span>
              </td>
              <td>{e.executed ? "✓" : ""}</td>
              <td>{e.ok == null ? "" : e.ok ? "✓" : "✗"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
