import { Section, Figure, TensorScale, Table, Takeaways, Check, Prose, n, shape } from "../../../kit";
import * as f from "../facts";
import { Graph, type Step } from "./graph";
export function Data() {
  const vision: Step[] = [
    {
      id: "patch",
      label: "RGB patch projection",
      width: f.visionWidth.value,
      what: "Project each spatial patch with a bias-free strided convolution.",
      formula: `RGB patches ${n(f.patch.value)} × ${n(f.patch.value)} → vectors`,
      input: shape("patches", f.rgb.value, f.patch.value, f.patch.value),
      output: shape("patches", f.visionWidth.value),
      params: [["Conv2d weight", shape(f.visionWidth.value, f.rgb.value, f.patch.value, f.patch.value)]],
      refs: [
        { source: "multimodal", symbol: "MoonVision3dPatchEmbed.forward" },
        { source: "serving-vl", symbol: "KimiK3VisionTower" },
      ],
    },
    {
      id: "vision",
      label: "positions + MoonViT",
      width: f.visionWidth.value,
      what: "Add spatial/temporal positions, run the vision encoder and final norm.",
      formula: `${n(f.visionLayers.value)} bidirectional vision blocks; 2D rotary q/k`,
      output: shape("patches", f.visionWidth.value),
      refs: [
        { source: "multimodal", symbol: "MoonViT3dEncoder.forward" },
        { source: "serving-vl", symbol: "KimiK3VisionTower" },
      ],
      note: "Vision internals are outside op-level scope; unlike text attention, vision does apply rotary positions.",
    },
    {
      id: "pool",
      label: "time mean + gather",
      width: f.visionMerged.value,
      what: "Average frames and gather spatial neighbors into grouped channels.",
      formula: `[t,h,w,C] → [h/${n(f.mergeKernel.value[0])},w/${n(f.mergeKernel.value[1])},${n(f.visionMerged.value)}]`,
      output: shape("merged patches", f.visionMerged.value),
      refs: [
        { source: "multimodal", symbol: "tpool_patch_merger" },
        { source: "serving-vl", symbol: "KimiK3VisionTower" },
      ],
    },
    {
      id: "project",
      label: "projector + norm",
      width: f.hidden.value,
      what: "Project grouped media patches to text width and normalize.",
      formula: `${n(f.visionMerged.value)} → ${n(f.visionMerged.value)} → ${n(f.hidden.value)}; GELU; RMSNorm`,
      params: [
        ["proj.0.weight", shape(f.visionMerged.value, f.visionMerged.value)],
        ["proj.2.weight", shape(f.hidden.value, f.visionMerged.value)],
        ["post_norm.weight", shape(f.hidden.value)],
      ],
      refs: [
        { source: "multimodal", symbol: "PatchMergerMLPV2.forward" },
        { source: "serving-vl", symbol: "KimiK3MultiModalProjector" },
      ],
    },
    {
      id: "replace",
      label: "replace placeholders",
      width: f.hidden.value,
      what: "Merge projected media features into the embedding sequence before the text decoder.",
      formula: "inputs_embeds[media positions] ← projected patches",
      refs: [
        { source: "multimodal", symbol: "KimiK3ForConditionalGeneration.forward" },
        { source: "serving", symbol: "KimiK3ForConditionalGeneration.forward" },
      ],
    },
  ];
  return (
    <Section
      id="data"
      title="Data representation"
      source={[
        { source: "reference", symbol: "KimiMLAAttention.forward" },
        { source: "multimodal", symbol: "KimiK3ForConditionalGeneration.forward" },
      ]}
      lede="The stream has a stable width across the decoder. Narrow latents serve different purposes: compressing attention state and reducing the width that routed experts process."
    >
      <Figure
        label="Stream and latent widths to scale"
        caption="Each column represents the same synthetic token count; height is channel width. Query and KV latents are intermediates, not additional residual streams."
      >
        <TensorScale
          title="Widths inside the decoder"
          pxPerToken={20}
          pxPerWidth={0.022}
          footnote="Equal token count in every column; heights share the same scalar-width scale."
          stages={[
            {
              label: "Stream",
              tensors: [{ key: "x", name: "hidden", tokens: f.DEMO.tokens, width: f.hidden.value, lane: "merged" }],
            },
            {
              label: "Query latent",
              tensors: [
                { key: "qa", from: "x", name: "q_a", tokens: f.DEMO.tokens, width: f.qRank.value, lane: "high" },
              ],
            },
            {
              label: "KV latent",
              tensors: [
                {
                  key: "kv",
                  from: "x",
                  name: "KV content",
                  tokens: f.DEMO.tokens,
                  width: f.kvRank.value,
                  lane: "instance",
                },
              ],
            },
            {
              label: "Expert latent",
              tensors: [
                {
                  key: "expert",
                  from: "x",
                  name: "routed input",
                  tokens: f.DEMO.tokens,
                  width: f.latent.value,
                  lane: "low",
                },
              ],
            },
          ]}
        />
      </Figure>
      <Table
        caption="Tensor interfaces"
        head={["Tensor", "Shape", "Meaning"]}
        rows={[
          ["Input ids", shape("B", "S"), "Tokenized input; integer indices into the embedding table."],
          ["Stream", shape("B", "S", f.hidden.value), "Position-local representation between decoder sublayers."],
          ["Logits", shape("B", "S", f.vocab.value), "Vocabulary scores; sampling is outside this page."],
          ["Depth bank", shape("B×S", "blocks", f.hidden.value), "Completed additive blocks, local to this forward."],
          [
            "KDA matrix",
            shape("B", f.heads.value, f.dim.value, f.dim.value),
            "Persistent token memory with fixed extent.",
          ],
        ]}
      />
      <Prose>
        The media path supplies embeddings, not a second decoder. Its exact interface is shown below; internal vision
        operations are an explicit scope boundary.
      </Prose>
      <Graph
        id="media-interface"
        title="Vision-to-text interface"
        heading="How media enters"
        steps={vision}
        source="KimiLinearModel.forward"
        serving="KimiK3ForConditionalGeneration.forward"
      />
      <Check question="Are the KV latent and expert latent interchangeable because both are narrow?">
        No. The KV latent builds attention keys and values; the expert latent is a tokenwise routed feed-forward
        representation with different width and weights.
      </Check>
      <Takeaways
        items={[
          "Shapes name axes explicitly; B is batch, S current sequence and T stored token extent.",
          "Media preprocessing and complete vision arithmetic are scoped out; media insertion is shown.",
          "The logits are scores, not sampled text.",
        ]}
      />
    </Section>
  );
}
