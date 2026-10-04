import { Section, Figure, TensorScale, FactList, Takeaways, Check, n, shape } from "../../../kit";
import * as f from "../facts";
export function Data() {
  return (
    <Section
      id="data"
      title="Data representation"
      source={[{ source: "reference", symbol: "Qwen3MoeAttention" }]}
      lede={
        <>
          Attention expands the query path to {n(f.qExpansion.value)} times the residual width. Keys and values stay
          narrow: each stored KV head is reused by {n(f.kvReuse.value)} query heads.
        </>
      }
    >
      <Figure
        label="Query and KV widths"
        caption={`To scale for ${n(f.DISPLAY_TOKENS)} positions: height is channels per token. The cache stores the narrow key and value tensors, before the eager implementation repeats their heads.`}
      >
        <TensorScale
          pxPerToken={6}
          pxPerWidth={160 / f.qWidth.value}
          title="Asymmetric query and cache widths"
          stages={[
            {
              label: "INPUT",
              tensors: [
                { key: "x", name: "residual", tokens: f.DISPLAY_TOKENS, width: f.hidden.value, lane: "merged" },
              ],
            },
            {
              label: "PROJECT",
              tensors: [
                { key: "q", from: "x", name: "queries", tokens: f.DISPLAY_TOKENS, width: f.qWidth.value, lane: "high" },
                { key: "k", from: "x", name: "keys", tokens: f.DISPLAY_TOKENS, width: f.kvWidth.value, lane: "low" },
                { key: "v", from: "x", name: "values", tokens: f.DISPLAY_TOKENS, width: f.kvWidth.value, lane: "low" },
              ],
            },
            {
              label: "RETAIN",
              tensors: [
                {
                  key: "ck",
                  from: "k",
                  name: "rotated keys",
                  tokens: f.DISPLAY_TOKENS,
                  width: f.kvWidth.value,
                  lane: "instance",
                },
                {
                  key: "cv",
                  from: "v",
                  name: "unrotated values",
                  tokens: f.DISPLAY_TOKENS,
                  width: f.kvWidth.value,
                  lane: "instance",
                },
              ],
            },
          ]}
          footnote="Area counts scalars; projections are separate and query states are not cached."
        />
      </Figure>
      <FactList
        items={[
          [
            "Input",
            <>
              Integer ids {shape("B", "S")}; configured BOS {n(f.bos.value)} and EOS {n(f.eos.value)}.
            </>,
          ],
          ["Output", <>Logits {shape("B", "S", f.vocab.value)} are raw vocabulary scores, not sampled tokens.</>],
          [
            "State",
            <>
              Per layer, rotated K and V each have shape {shape("B", f.kvHeads.value, "T", f.headDim.value)}. Positions
              index rotations, not a learned table.
            </>,
          ],
        ]}
      />
      <Check question="Does sharing a KV head force its query heads to produce the same attention weights?">
        No. They share keys and values, but each query head has its own projected query; its dot products and softmax
        can differ.
      </Check>
      <Takeaways
        items={[
          <>
            Explicit head dimension {n(f.headDim.value)} makes the query width {n(f.qWidth.value)}; dividing the
            residual width by query-head count would give the wrong architecture.
          </>,
          <>
            GQA reduces logical KV storage by {n(f.kvReuse.value)} times relative to storing separate KV for every query
            head.
          </>,
        ]}
      />
    </Section>
  );
}
