import { useEffect, useRef } from "react";

export interface TranscriptLine {
  role: string;
  text: string;
  final: boolean;
  lang?: string;
}

export function Transcript({ lines }: { lines: TranscriptLine[] }) {
  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [lines]);

  if (lines.length === 0) {
    return <p className="muted">No transcript yet. Start a session and speak.</p>;
  }

  return (
    <div className="transcript">
      {lines.map((l, i) => (
        <div key={i} className={`line ${l.role} ${l.final ? "" : "partial"}`}>
          <span className="role">{l.role}</span>
          <span className="text">{l.text}</span>
          {l.lang && <span className="lang">{l.lang}</span>}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
