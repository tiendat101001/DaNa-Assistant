import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In production the agent serves the built bundle from control.ui_dist_dir.
// In dev, Vite proxies REST + WebSocket to the loopback control API so the
// dashboard talks to the real agent without CORS or a second origin.
const AGENT = "http://127.0.0.1:8760";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: AGENT, changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
