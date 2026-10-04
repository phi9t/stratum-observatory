import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "../styles/schematic.css";
import { SourcesProvider } from "../lib/sources";
import { Page } from "./Page";

// The catalogue has no model behind it: one placeholder source so refs render.
const sources = {
  checkpoint: { repo: "example/toy", revision: "0".repeat(40) },
  sources: [{ id: "reference", kind: "github" as const, repo: "example/toy", revision: "0".repeat(40), path: "model.py", role: "reference" }],
};

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <SourcesProvider sources={sources}>
      <Page />
    </SourcesProvider>
  </StrictMode>,
);
