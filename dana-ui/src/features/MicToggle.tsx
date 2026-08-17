import { useState } from "react";
import { ControlApi } from "../api/client";

// Hard mic mute. The button reflects the authoritative `muted` from status/WS;
// the actual gate lives in the pipeline (MuteProcessor), not in the UI.
export function MicToggle({ api, muted }: { api: ControlApi; muted: boolean }) {
  const [busy, setBusy] = useState(false);

  const toggle = async () => {
    setBusy(true);
    try {
      await (muted ? api.unmute() : api.mute());
    } finally {
      setBusy(false);
    }
  };

  return (
    <button
      className={`mic ${muted ? "muted" : ""}`}
      disabled={busy}
      onClick={toggle}
      aria-pressed={muted}
      aria-label={muted ? "Unmute microphone" : "Mute microphone"}
    >
      {muted ? "🔇 Unmute mic" : "🎙 Mute mic"}
    </button>
  );
}
