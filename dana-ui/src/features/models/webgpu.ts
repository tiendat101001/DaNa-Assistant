// In-browser LLM runtime via MediaPipe GenAI (WebGPU). Loaded lazily so the
// ~MB-sized wasm glue never lands in the main bundle.
export const MEDIAPIPE_VERSION = "0.10.30";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@${MEDIAPIPE_VERSION}/wasm`;

export interface GpuInfo {
  available: boolean;
  description?: string;
  vendor?: string;
  architecture?: string;
  reason?: string;
}

interface GpuAdapterLike {
  info?: { vendor?: string; architecture?: string; description?: string };
}
interface GpuLike {
  requestAdapter(): Promise<GpuAdapterLike | null>;
}

export async function detectWebGpu(): Promise<GpuInfo> {
  const gpu = (navigator as unknown as { gpu?: GpuLike }).gpu;
  if (!gpu) {
    return {
      available: false,
      reason:
        "WebGPU is not available in this browser. Use a recent Chrome/Edge over https or localhost.",
    };
  }
  try {
    const a = await gpu.requestAdapter();
    if (!a) return { available: false, reason: "No compatible GPU adapter was found." };
    return {
      available: true,
      vendor: a.info?.vendor,
      architecture: a.info?.architecture,
      description: a.info?.description,
    };
  } catch (e) {
    return { available: false, reason: e instanceof Error ? e.message : String(e) };
  }
}

export interface BrowserLlm {
  generate(prompt: string, onPartial: (text: string) => void): Promise<{ ttftMs: number; totalMs: number }>;
  close(): void;
}

export async function loadBrowserLlm(opts: {
  url: string;
  token: string;
  onStatus: (s: string) => void;
}): Promise<BrowserLlm> {
  opts.onStatus("Loading WebGPU runtime…");
  const { FilesetResolver, LlmInference } = await import("@mediapipe/tasks-genai");
  const genai = await FilesetResolver.forGenAiTasks(WASM_BASE);

  opts.onStatus("Downloading model into the browser…");
  const res = await fetch(opts.url, { headers: { Authorization: `Bearer ${opts.token}` } });
  if (!res.ok) throw new Error(`Model download failed (${res.status})`);
  const buf = await res.arrayBuffer();

  opts.onStatus("Initialising model on the GPU…");
  const llm = await LlmInference.createFromOptions(genai, {
    baseOptions: { modelAssetBuffer: new Uint8Array(buf) },
    maxTokens: 512,
    topK: 40,
    temperature: 0.7,
  });
  opts.onStatus("Ready");

  return {
    async generate(prompt, onPartial) {
      const t0 = performance.now();
      let first: number | null = null;
      let acc = "";
      await llm.generateResponse(prompt, (partial: string) => {
        if (first === null) first = performance.now() - t0;
        acc += partial;
        onPartial(acc);
      });
      const total = performance.now() - t0;
      return { ttftMs: first ?? total, totalMs: total };
    },
    close() {
      llm.close();
    },
  };
}
