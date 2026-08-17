// WebSocket connection to /api/v1/events. Browsers cannot set an Authorization
// header on a WebSocket, so the token goes in the query string (the agent binds
// to loopback only). Auto-reconnects with a short backoff.

import type { Envelope } from "./types";

export type EnvelopeHandler = (env: Envelope) => void;
export type StatusHandler = (connected: boolean) => void;

export class EventStream {
  private ws: WebSocket | null = null;
  private closed = false;
  private retry = 0;

  constructor(
    private token: string,
    private onEnvelope: EnvelopeHandler,
    private onStatus: StatusHandler,
  ) {}

  start(): void {
    this.closed = false;
    this.connect();
  }

  stop(): void {
    this.closed = true;
    this.ws?.close();
    this.ws = null;
  }

  private connect(): void {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/api/v1/events?token=${encodeURIComponent(this.token)}`;
    const ws = new WebSocket(url);
    this.ws = ws;

    ws.onopen = () => {
      this.retry = 0;
      this.onStatus(true);
    };
    ws.onmessage = (ev) => {
      try {
        this.onEnvelope(JSON.parse(ev.data) as Envelope);
      } catch {
        // ignore malformed frames
      }
    };
    ws.onclose = () => {
      this.onStatus(false);
      if (!this.closed) {
        this.retry = Math.min(this.retry + 1, 6);
        setTimeout(() => this.connect(), 500 * this.retry);
      }
    };
    ws.onerror = () => ws.close();
  }
}
