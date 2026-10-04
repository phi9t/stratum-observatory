import { Section, Sub, Prose, OpGraph, Takeaways, Check, Table, n, shape } from "../../../kit";
import * as f from "../facts";
import { graph, stream } from "../graphs";
export function Auxiliary() {
  return (
    <Section
      id="auxiliary"
      title="Image input and draft output"
      source={[
        { source: "vision", symbol: "ViT" },
        { source: "reference", symbol: "DSparkBlock" },
      ]}
      lede="The released checkpoint also has an image encoder and a draft path. These are shown structurally; the clean executable reference scopes them out."
    >
      <Sub>Image features replace an input span</Sub>
      <Prose>
        The image encoder uses bidirectional attention over one image with spatial RoPE. Its aligner groups neighboring
        patch vectors into a language-width feature; learned delimiters fill the image-span boundary positions.
      </Prose>
      <OpGraph
        id="vision"
        title="Image input path"
        {...graph(
          [
            {
              id: "patch",
              label: "patch projection",
              width: f.visionDim.value,
              output: shape("P", f.visionDim.value),
              what: "Flatten each RGB patch and apply a biased embedding projection.",
              formula: "x = flatten(patch) Wpatchᵀ + bias",
              input: shape("P", "RGB", f.patch.value, f.patch.value),
              params: [
                ["proj.weight", shape(f.visionDim.value, "RGB·patch²")],
                ["proj.bias", shape(f.visionDim.value)],
              ],
            },
            {
              id: "vblocks",
              label: "vision blocks",
              width: f.visionDim.value,
              output: shape("P", f.visionDim.value),
              what: "Pre-RMSNorm, full spatial attention, residual add, pre-RMSNorm, SwiGLU, residual add.",
              formula: `repeat ${n(f.visionLayers.value)}: x+=Attn(norm(x)); x+=MLP(norm(x))`,
            },
            {
              id: "vnorm",
              label: "final RMSNorm",
              width: f.visionDim.value,
              output: shape("P", f.visionDim.value),
              what: "Normalize image features before spatial grouping.",
              formula: "x = RMSNorm(x)",
              params: [["vision.norm.weight", shape(f.visionDim.value)]],
            },
            {
              id: "unfold",
              label: "pad + unfold",
              width: f.visionDim.value * f.downsample.value ** 2,
              output: shape("P_group", f.visionDim.value * f.downsample.value ** 2),
              what: "Pad the spatial grid to a multiple of the grouping ratio and concatenate non-overlapping patch groups.",
              formula: `unfold(x, kernel=stride=${n(f.downsample.value)})`,
            },
            {
              id: "align",
              params: [
                ["aligner.w1.weight", shape(f.hidden.value, f.visionDim.value * f.downsample.value ** 2)],
                ["aligner.w1.bias", shape(f.hidden.value)],
                ["aligner.w2.weight", shape(f.hidden.value, f.hidden.value)],
                ["aligner.w2.bias", shape(f.hidden.value)],
              ],
              label: "linear GELU linear",
              width: f.hidden.value,
              output: shape("P_group", f.hidden.value),
              what: "Biased projections and GELU align grouped image features to language width.",
              formula: "y = W2 GELU(W1 x + b1) + b2",
            },
            {
              id: "replace",
              label: "replace image span",
              width: f.hidden.value,
              output: stream,
              what: "Image slots receive features; start/end/newline slots receive learned delimiter embeddings.",
              formula: "embedding[image_span] = features + delimiters",
            },
          ],
          "ViT",
          "",
          "vision",
        )}
      />
      <Sub>Inside a vision block</Sub>
      <Prose>
        Unlike language attention, the image attention uses separate values, biased projections and a spatially
        bidirectional pattern.
      </Prose>
      <OpGraph
        id="vision-attn"
        title="Spatial attention"
        {...graph(
          [
            {
              id: "vqkv",
              params: [
                ["norm1.weight", shape(f.visionDim.value)],
                ["attn.wqkv.weight", shape(f.visionQKV.value, f.visionDim.value)],
                ["attn.wqkv.bias", shape(f.visionQKV.value)],
              ],
              input: shape("P", f.visionDim.value),
              label: "norm + qkv",
              width: f.visionDim.value,
              output: shape("P", f.visionHeads.value, f.visionDim.value / f.visionHeads.value),
              what: "RMSNorm first, then biased qkv projection, split and reshape into heads.",
              formula: "q,k,v = heads(split(Wqkv RMSNorm(x)+bias))",
            },
            {
              id: "vrope",
              label: "spatial RoPE",
              width: f.visionDim.value,
              output: shape("P", f.visionHeads.value, f.visionDim.value / f.visionHeads.value),
              what: "Rotate query and key with separate height and width positions; leave values unrotated.",
              formula: "q,k = spatial_RoPE(q,k)",
            },
            {
              id: "vsoftmax",
              label: "full attention",
              width: f.visionDim.value,
              output: shape("P", f.visionHeads.value, f.visionDim.value / f.visionHeads.value),
              what: "Each patch attends all image patches; there is no text causal mask.",
              formula: "o = softmax(qkᵀ / sqrt(head_dim)) v",
            },
            {
              id: "vout",
              params: [
                ["attn.wo.weight", shape(f.visionDim.value, f.visionDim.value)],
                ["attn.wo.bias", shape(f.visionDim.value)],
              ],
              label: "out + residual",
              width: f.visionDim.value,
              output: shape("P", f.visionDim.value),
              what: "Merge heads, apply biased output projection, add saved residual.",
              formula: "x′ = x + Wo merge(o) + bo",
            },
            {
              id: "vmlp",
              params: [
                ["norm2.weight", shape(f.visionDim.value)],
                ["mlp.w1.weight", shape(2 * f.visionInter.value, f.visionDim.value)],
              ],
              label: "norm + SwiGLU",
              width: f.visionInter.value,
              output: shape("P", f.visionInter.value),
              what: "Normalize, project to gate/up without bias, activate gate and multiply.",
              formula: "a = SiLU(gate(norm(x′))) · up(norm(x′))",
            },
            {
              id: "vdown",
              params: [["mlp.w2.weight", shape(f.visionDim.value, f.visionInter.value)]],
              label: "down + residual",
              width: f.visionDim.value,
              output: shape("P", f.visionDim.value),
              what: "Bias-free down projection and second residual add.",
              formula: "y = x′ + Wdown a",
            },
          ],
          "Block",
          "",
          "vision",
        )}
      />
      <Sub>dSpark: draft blocks beside the backbone</Sub>
      <Prose>
        The first draft stage reads concatenated means of backbone residual copies at target layers{" "}
        {f.draftTargets.value.map(n).join(", ")}. Its window attends main history and all draft slots; this is not
        ordinary causal backbone attention.
      </Prose>
      <OpGraph
        id="draft"
        title="Draft forward boundary"
        {...graph(
          [
            {
              id: "target",
              label: "target means",
              width: f.hidden.value * f.draftTargets.value.length,
              output: shape("B", "S", f.hidden.value * f.draftTargets.value.length),
              what: "Capture residual means before attention in the designated target layers, then concatenate.",
              formula: "main_hidden = concat(mean_copy(H_target))",
            },
            {
              id: "main",
              params: [
                ["main_proj.weight", shape(f.hidden.value, f.hidden.value * f.draftTargets.value.length)],
                ["main_norm.weight", shape(f.hidden.value)],
              ],
              label: "project + norm",
              width: f.hidden.value,
              output: stream,
              what: "First draft stage projects concatenated target features to a normalized main input.",
              formula: "main_x = RMSNorm(main_proj(main_hidden))",
            },
            {
              id: "noise",
              col: 1,
              row: 1,
              input: shape("B", f.draftSize.value),
              label: "draft embedding",
              width: f.residualWidth.value,
              output: shape("B", f.draftSize.value, f.copies.value, f.hidden.value),
              what: "Fill draft positions with the noise token, replace the first position with the current token, reuse the main embedding, and expand copies.",
              formula: "H = repeat_copy(embed([current, noise…]))",
            },
            {
              id: "draftblocks",
              col: 2,
              label: "draft stages",
              width: f.residualWidth.value,
              output: shape("B", f.draftSize.value, f.copies.value, f.hidden.value),
              what: "Hyper-connected pure-window attention and smaller MoE; prefill only seeds the main window cache.",
              formula: `repeat ${n(f.draftLayers.value)}: DSparkBlock(H,main_x)`,
            },
            {
              id: "drafthead",
              col: 3,
              label: "collapse + head",
              width: f.vocab.value,
              output: shape("B", f.draftSize.value, f.vocab.value),
              what: "Collapse final copies with handed-off pre, normalize, reuse the main output head.",
              formula: "logits = head(RMSNorm(hc_pre(H)))",
            },
            {
              id: "markov",
              col: 4,
              label: "Markov + confidence",
              width: f.draftSize.value,
              output: shape("B", f.draftSize.value),
              what: "Sequentially add a Markov head’s vocabulary bias and sample; confidence projects final hidden concatenated with Markov embeddings.",
              formula: "logits_i += markov(previous_id); confidence = proj([hidden, markov_embed])",
            },
          ],
          "DSparkBlock",
        )}
        edges={[
          { from: "target", to: "main" },
          { from: "main", to: "draftblocks", kind: "state" },
          { from: "noise", to: "draftblocks" },
          { from: "draftblocks", to: "drafthead" },
          { from: "drafthead", to: "markov" },
        ]}
      />
      <Table
        caption="Draft scope"
        head={["Property", "Checkpoint value"]}
        rows={[
          ["Draft stages", n(f.draftLayers.value)],
          ["Draft block width", n(f.draftSize.value)],
          ["Draft routed / active experts", `${n(f.draftExperts.value)} / ${n(f.draftTopK.value)}`],
          ["Markov embedding width", n(f.markovRank.value)],
          ["Main embedding and head", "Shared with draft stages"],
        ]}
      />
      <Check question="Does a dSpark forward implementation establish a complete speculative-decoding server?">
        No. The publisher implements the draft forward pass; a comment in its model code and its inference README say
        the generation script does not use it (that script is not pinned here). Acceptance, rollback, batching and
        performance are outside this page.
      </Check>
      <Takeaways
        items={[
          "Vision produces language-width vectors before residual copies are expanded.",
          "Draft features are captured before target attention, not from its output.",
          "Auxiliary structure is pinned, but neither auxiliary path has executable parity coverage here.",
        ]}
      />
    </Section>
  );
}
