import { createContext, useContext, type ReactNode } from "react";

export type Source = Readonly<{
  id: string;
  kind: "hf" | "github";
  repo: string;
  revision: string;
  path: string;
  role: string;
}>;

export type SourcesFile = Readonly<{
  checkpoint: Readonly<{ repo: string; revision: string }>;
  sources: readonly Source[];
}>;

/** symbols.json, written by `npm run fetch`: source id -> symbol -> [first, last] line. */
export type SymbolIndex = Readonly<Record<string, Readonly<Record<string, readonly [number, number]>>>>;

export type Ref = Readonly<{ source: string; symbol: string }>;

const Context = createContext<Readonly<{ file: SourcesFile; symbols: SymbolIndex }> | null>(null);

export function SourcesProvider({
  sources,
  symbols = {},
  children,
}: {
  sources: SourcesFile;
  symbols?: SymbolIndex;
  children: ReactNode;
}) {
  return <Context.Provider value={{ file: sources, symbols }}>{children}</Context.Provider>;
}

export function useSources(): Readonly<{ file: SourcesFile; symbols: SymbolIndex }> {
  const value = useContext(Context);
  if (value === null) {
    throw new Error("useSources: wrap the page in <SourcesProvider sources={sources}>");
  }
  return value;
}

export function sourceUrl(source: Source, lines?: readonly [number, number]): string {
  const host = source.kind === "hf" ? "https://huggingface.co" : "https://github.com";
  const anchor = lines === undefined ? "" : `#L${String(lines[0])}-L${String(lines[1])}`;
  return `${host}/${source.repo}/blob/${source.revision}/${source.path}${anchor}`;
}

/** Link to the pinned definition of one symbol: `modeling_x.py · Attention L210–288`. */
export function SourceRef({ source: id, symbol }: Ref) {
  const { file, symbols } = useSources();
  const source = file.sources.find((candidate) => candidate.id === id);
  if (source === undefined) {
    throw new Error(`SourceRef: sources.json has no source "${id}"`);
  }
  const lines = symbols[id]?.[symbol];
  const name = source.path.split("/").slice(-1)[0];
  return (
    <a className="ms-ref" href={sourceUrl(source, lines)} rel="noreferrer" target="_blank">
      {name} · {symbol}
      {lines === undefined ? "" : ` L${String(lines[0])}–${String(lines[1])}`}
    </a>
  );
}
