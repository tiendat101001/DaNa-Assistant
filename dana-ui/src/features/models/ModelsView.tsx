import { useMemo, useState } from "react";
import type { ControlApi } from "../../api/client";
import { type LocalModel, ModelsApi } from "../../api/models";
import type { LatencySpan } from "../../api/types";
import { HubBrowser } from "./HubBrowser";
import { LatencyBudget } from "./LatencyBudget";
import { LlmHosting } from "./LlmHosting";
import { LocalModels } from "./LocalModels";
import { PullJobs } from "./PullJobs";
import { WebGpuPanel } from "./WebGpuPanel";

export function ModelsView({ api, spans }: { api: ControlApi; spans: LatencySpan[] }) {
  const mapi = useMemo(() => new ModelsApi(api), [api]);
  const [localKey, setLocalKey] = useState(0);
  const [jobKick, setJobKick] = useState(0);
  const [local, setLocal] = useState<LocalModel[]>([]);
  const [browserModel, setBrowserModel] = useState<string | null>(null);

  const bump = () => setLocalKey((k) => k + 1);

  return (
    <main className="grid models">
      <section className="card wide">
        <h2>Latency budget</h2>
        <LatencyBudget spans={spans} />
      </section>
      <section className="card wide">
        <h2>Downloads</h2>
        <PullJobs api={mapi} kick={jobKick} onJobFinished={bump} />
      </section>
      <section className="card wide">
        <h2>Browse litert-community</h2>
        <HubBrowser
          api={mapi}
          refreshKey={localKey}
          onPulled={() => {
            setJobKick((k) => k + 1);
            bump();
          }}
        />
      </section>
      <section className="card wide">
        <h2>My models</h2>
        <LocalModels
          api={mapi}
          refreshKey={localKey}
          onChanged={bump}
          onList={setLocal}
          onRunInBrowser={(m) => setBrowserModel(m.id)}
        />
      </section>
      <section className="card wide">
        <h2>LLM hosting</h2>
        <LlmHosting api={mapi} />
      </section>
      <section className="card wide">
        <h2>Run in browser (WebGPU)</h2>
        <WebGpuPanel api={mapi} models={local} preselect={browserModel} />
      </section>
    </main>
  );
}
