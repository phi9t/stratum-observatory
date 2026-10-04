import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "../../kit/styles/schematic.css";
import { SourcesProvider, type SourcesFile, type SymbolIndex } from "../../kit/lib/sources";
import { Page } from "./Page";
import sources from "./sources.json";
import symbols from "./symbols.json";

const root = document.getElementById("root");
if (root === null) {
  throw new Error("index.html has no #root");
}
createRoot(root).render(
  <StrictMode>
    <SourcesProvider sources={sources as SourcesFile} symbols={symbols as unknown as SymbolIndex}>
      <Page />
    </SourcesProvider>
  </StrictMode>,
);
