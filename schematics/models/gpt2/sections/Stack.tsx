import { LayerMap, Prose, Section, Sentence, n, useUrlNumber } from "../../../kit";
import * as f from "../facts";

export const KINDS = [{ id: "block", label: "attention + MLP block", lane: "high" }] as const;

export function useLayer() {
  return useUrlNumber("layer", 0, 0, f.layers.value - 1);
}

export function Stack() {
  const [layer, setLayer] = useLayer();
  return (
    <Section
      id="stack"
      lede="Every layer is the same module with its own weights. Pick one; the block and attention drawings below follow the selection."
      source={[{ source: "reference", symbol: "GPT2Model" }]}
      title="The stack"
    >
      <LayerMap count={f.layers.value} kindOf={() => "block"} kinds={KINDS} label="GPT-2 layers" onSelect={setLayer} selected={layer} />
      <Sentence>
        Layer {n(layer)} is block h[{n(layer)}]: it reads the stream left by {layer === 0 ? "the embeddings" : `layer ${n(layer - 1)}`} and{" "}
        {layer === f.layers.value - 1 ? "hands its result to the final LayerNorm." : `passes its result to layer ${n(layer + 1)}.`}
      </Sentence>
      <Prose>
        GPT-2 has one layer kind. The code can scale attention by one over the layer number, but this checkpoint's config leaves that off, so
        the only thing that differs between layers is the learned weights.
      </Prose>
    </Section>
  );
}
