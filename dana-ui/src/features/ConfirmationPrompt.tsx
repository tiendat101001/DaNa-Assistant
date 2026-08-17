import { useEffect, useRef } from "react";
import type { ConfirmationRequest } from "../api/types";

// Mirrors the voice confirmation: a destructive action is paused until the user
// approves or denies (→ POST /confirmations/{id}).
//
// A gated action is safety-critical, so this is a MODAL overlay (dimmed backdrop,
// role="dialog", aria-modal) rather than an inline bar — it must be impossible to
// miss or click past. Focus lands on the SAFE default (Deny) so an accidental
// Enter/Space never approves a destructive action; Escape denies the top item.
// Tab/Shift-Tab are trapped inside the dialog so focus cannot drift to background
// controls while a destructive action awaits approval.
export function ConfirmationPrompt({
  items,
  onResolve,
}: {
  items: ConfirmationRequest[];
  onResolve: (id: string, approved: boolean) => void;
}) {
  const denyRef = useRef<HTMLButtonElement | null>(null);
  const approveRef = useRef<HTMLButtonElement | null>(null);
  const top = items[0];

  useEffect(() => {
    denyRef.current?.focus();
  }, [top?.id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && top) {
        e.preventDefault();
        onResolve(top.id, false); // Escape = deny (fail-safe)
        return;
      }
      // Trap Tab/Shift-Tab inside the two action buttons so focus cannot drift
      // to background controls while a destructive action awaits approval.
      if (e.key === "Tab" && top) {
        e.preventDefault();
        const deny = denyRef.current;
        const approve = approveRef.current;
        if (!deny || !approve) return;
        if (e.shiftKey) {
          (document.activeElement === deny ? approve : deny).focus();
        } else {
          (document.activeElement === approve ? deny : approve).focus();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [top, onResolve]);

  if (!top) return null;

  return (
    <div className="confirm-overlay" role="presentation">
      <div
        className="confirm-modal"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
      >
        <h2 id="confirm-title" className="confirm-title">
          ⚠ Action needs your approval
          {items.length > 1 && <span className="confirm-count"> ({items.length} pending)</span>}
        </h2>
        {items.map((c, i) => (
          <div key={c.id} className={`confirm ${i === 0 ? "confirm-active" : ""}`}>
            <div className="confirm-body">
              <div className="confirm-meta">
                <span className={`tag ${c.reversibility}`}>{c.reversibility}</span>
                <strong>{c.tool}</strong>
              </div>
              <code className="preview">{c.preview}</code>
              {c.args_resolved && Object.keys(c.args_resolved).length > 0 && (
                <ul className="confirm-args">
                  {Object.entries(c.args_resolved).map(([k, v]) => (
                    <li key={k}>
                      <span className="mono">{k}</span>: <span className="mono">{fmt(v)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="row confirm-actions">
              <button
                ref={i === 0 ? denyRef : undefined}
                className="secondary"
                onClick={() => onResolve(c.id, false)}
                aria-label={`Deny ${c.tool}`}
              >
                Deny
              </button>
              <button
                ref={i === 0 ? approveRef : undefined}
                className="danger"
                onClick={() => onResolve(c.id, true)}
                aria-label={`Approve ${c.tool}`}
              >
                Approve
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function fmt(v: unknown): string {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
