// Widget catalogue: every playable widget with made-up numbers, so you can see
// what each one looks like and copy its wiring. This is NOT a model page: it has
// no facts or sources, and its numbers are placeholders. For a real page, see
// models/gpt2/.
import {
  AttentionGrid, Button, Figure, LayerMap, MoeRouter, Readout, RopeDials, Schematic, Section, Segments, Sentence, Slider,
  n, routeToken, useUrlChoice, useUrlNumber, type CellState, type LayerKind, type Scoring,
} from "..";

export const sections = [
  { id: "layers", label: "LayerMap" },
  { id: "router", label: "MoeRouter" },
  { id: "patterns", label: "AttentionGrid" },
  { id: "rope", label: "RopeDials" },
] as const;
export const sectionIds: string[] = sections.map((section) => section.id);

const ref = [{ source: "reference", symbol: "Toy" }];
const KINDS: readonly LayerKind[] = [
  { id: "dense", label: "dense FFN", lane: "merged" },
  { id: "moe-full", label: "MoE · full attention", lane: "high" },
  { id: "moe-linear", label: "MoE · linear attention", lane: "low" },
];
const LAYERS = 48;
const kindOf = (index: number): string => (index < 3 ? "dense" : index % 4 === 3 ? "moe-full" : "moe-linear");
const SCORINGS: readonly Scoring[] = ["sigmoid", "softmax", "sqrtsoftplus"];
const PATTERNS = ["causal", "window", "sparse"] as const;
const WINDOW = 6;
const STRIDE = 5;

export function Page() {
  const [layer, setLayer] = useUrlNumber("layer", 0, 0, LAYERS - 1);
  const [seed, setSeed] = useUrlNumber("seed", 7, 0, 9999);
  const [scoring, setScoring] = useUrlChoice<Scoring>("scoring", "sigmoid", SCORINGS);
  const [pattern, setPattern] = useUrlChoice("pattern", "window", PATTERNS);
  const [query, setQuery] = useUrlNumber("q", 17, 0, 23);
  const [position, setPosition] = useUrlNumber("pos", 12, 0, 255);
  const routing = routeToken({ experts: 256, topK: 8, scoring, seed, groups: 8, topGroups: 4, normalize: scoring !== "softmax", scale: 2.5 });
  const cell = (q: number, k: number): CellState => {
    if (k > q) return "masked";
    if (k === q) return "self";
    if (pattern === "causal") return "attend";
    if (q - k < WINDOW) return "window";
    return pattern === "sparse" && k % STRIDE === 0 ? "sparse" : "masked";
  };
  return (
    <Schematic
      authority={<>Placeholder numbers; not a model</>}
      eyebrow="Model schematic kit"
      name="Widget catalogue"
      sections={sections}
      spec={[{ value: n(LAYERS), label: "toy layers" }]}
      tagline="Every playable widget in the kit, wired the way a model page wires it."
    >
      <Section id="layers" lede="One tile per layer, coloured by kind. The legend counts each kind." source={ref} title="LayerMap">
        <LayerMap count={LAYERS} kindOf={kindOf} kinds={KINDS} label="Toy layers" onSelect={setLayer} selected={layer} />
        <Sentence>Layer {n(layer)} is a {KINDS.find((kind) => kind.id === kindOf(layer))?.label} layer.</Sentence>
      </Section>
      <Section id="router" lede="Seeded synthetic logits through a real routing rule: score, keep the best groups, take top-k, renormalise, scale." source={ref} title="MoeRouter">
        <Segments legend="Scoring function" onChange={setScoring} options={SCORINGS.map((id) => ({ id, label: id }))} value={scoring} />
        <div className="ms-control">
          <Button onClick={() => setSeed((current) => current + 1)}>Route another token</Button>
        </div>
        <Sentence>
          Token {n(seed)} goes to experts {[...routing.chosen].sort((a, b) => a - b).map(n).join(", ")} in groups {[...routing.chosenGroups].sort((a, b) => a - b).map(n).join(", ")}.
        </Sentence>
        <Figure caption="The logits are synthetic; the selection rule is what a page takes from its checkpoint." label="Router drawing">
          <MoeRouter experts={256} groups={8} routing={routing} shared={1} title="Toy router" />
        </Figure>
        <Readout items={[{ term: "Selected", value: n(routing.chosen.length), note: "of 256" }, { term: "Weight sum", value: routing.weights.reduce((sum, w) => sum + w, 0).toFixed(3) }]} label="Router readout" />
      </Section>
      <Section id="patterns" lede="Which keys a query may read, for three common patterns." source={ref} title="AttentionGrid">
        <Segments legend="Pattern" onChange={setPattern} options={PATTERNS.map((id) => ({ id, label: id }))} value={pattern} />
        <Slider label="Query position" max={23} min={0} onChange={setQuery} value={query} />
        <Figure label="Attention pattern drawing">
          <AttentionGrid
            cell={cell}
            legend={[{ state: "attend", label: "readable" }, { state: "window", label: "inside the window" }, { state: "sparse", label: "selected far key" }, { state: "self", label: "self" }, { state: "masked", label: "masked" }]}
            onQuery={setQuery}
            query={query}
            size={24}
            title="Toy attention pattern"
          />
        </Figure>
      </Section>
      <Section id="rope" lede="Each frequency pair rotates by position times its frequency: fast pairs spin, slow pairs barely move." source={ref} title="RopeDials">
        <Slider label="Position" max={255} min={0} onChange={setPosition} value={position} />
        <Figure label="Rotary position drawing">
          <RopeDials pairs={[0, 1, 4, 8, 16, 31]} position={position} rotaryDim={64} theta={10000} title="Toy rotary positions" />
        </Figure>
      </Section>
    </Schematic>
  );
}
