import {
  Section,
  Slider,
  Sentence,
  Readout,
  Table,
  Ledger,
  Prose,
  SourceRef,
  Takeaways,
  Check,
  useUrlNumber,
  n,
} from "../../../kit";
import * as f from "../facts";
export function Serving() {
  const [batch, setBatch] = useUrlNumber("batch", 1, 1, f.DEMO.batchMax);
  const [context, setContext] = useUrlNumber("ctx", f.DEMO.contextDefault, 1, f.maxContext.value);
  const kda = batch * f.kdaCount.value * (f.recurrenceScalars.value + f.convolutionScalars.value);
  const expanded = batch * context * f.fullCount.value * f.logicalCacheWidth.value;
  const absorbed = batch * context * f.fullCount.value * f.absorbedCacheWidth.value;
  return (
    <Section
      id="serving"
      title="What serving changes"
      source={[
        { source: "serving", symbol: "KimiK3LinearForCausalLM.post_load_weights" },
        { source: "reference", symbol: "KimiDynamicCache" },
      ]}
      lede="KDA retains fixed-size state per sequence. Only full-attention layers grow with token context; an absorbed MLA implementation can keep latents instead of expanded keys and values."
    >
      <Slider label="Batch" min={1} max={f.DEMO.batchMax} value={batch} onChange={setBatch} />
      <Slider label="Context length" min={1} max={f.maxContext.value} value={context} onChange={setContext} />
      <Sentence>
        Batch {n(batch)} and context {n(context)} retain {n(kda)} KDA state scalars. Full attention adds {n(expanded)}{" "}
        expanded publisher scalars or {n(absorbed)} absorbed serving scalars.
      </Sentence>
      <Readout
        label="Computed values"
        items={[
          {
            term: "Expanded full KV",
            value: n(expanded),
            note: `B × T × ${n(f.fullCount.value)} × ${n(f.logicalCacheWidth.value)}`,
          },
          {
            term: "Absorbed full KV",
            value: n(absorbed),
            note: `B × T × ${n(f.fullCount.value)} × ${n(f.absorbedCacheWidth.value)}`,
          },
          { term: "KDA recurrent + conv", value: n(kda), note: "B × recurrent layers × fixed state" },
          {
            term: "Illustrative uniform BF16 bytes",
            value: n((kda + absorbed) * f.DEMO.bytes),
            note: "Scalar count × bytes; excludes allocator, scratch, metadata and depth bank",
          },
        ]}
      />
      <Prose>
        The uniform BF16 estimate is a teaching assumption, not SGLang's actual allocation: recurrent accumulation and
        convolution may use different precisions. Depth snapshots are per-forward token activations and are excluded
        from this persistent token-state budget.
      </Prose>
      <Table
        caption="Implementation differences and reconciliation"
        head={["Change", "Evidence and consequence"]}
        rows={[
          [
            "Absorb KV expansion",
            <>
              SGLang precomputes w_kc and w_vc from kv_b. Its base MLA attention has latent-sized value state.{" "}
              <SourceRef source="serving" symbol="KimiK3LinearForCausalLM.post_load_weights" />{" "}
              <SourceRef source="serving-mla" symbol="DeepseekV2AttentionMLA.__init__" />
            </>,
          ],
          [
            "Fuse KDA input and decode",
            <>
              The full-rank branch merges q/k/v/g projections. Covered decode kernels may consume an
              output-normalization gate stash; fallback applies the norm separately.{" "}
              <SourceRef source="serving" symbol="KimiK3DeltaAttention.forward" />
            </>,
          ],
          [
            "Reduce latent before norm",
            <>
              Partial routed sums are reduced in compressed space before RMSNorm. Shared experts remain a separate
              original-width branch. <SourceRef source="serving" symbol="KimiK3MoE._reduce_latent" />
            </>,
          ],
          [
            "Depth mixture scheduling",
            <>
              Serving precomputes gain × score projection, fuses snapshot and aggregation, and delays prefix addition
              where supported. <SourceRef source="serving" symbol="KimiK3DecoderLayer._forward_attn_residual" />
            </>,
          ],
          [
            "Numerical differences",
            <>
              Publisher MLA latent norms use their constructor default; SGLang base MLA uses config epsilon. Packed
              expert arithmetic and fused rounding are unverified here.{" "}
              <SourceRef source="serving-mla" symbol="DeepseekV2AttentionMLA.__init__" />
            </>,
          ],
        ]}
      />
      <Check question="Does doubling context double the whole cache budget?">
        It doubles the full-attention token state, while KDA's recurrent and convolution histories stay fixed for the
        same batch. The total therefore has both a fixed and a context-linear term.
      </Check>
      <Takeaways
        items={[
          `Absorbed versus expanded full-attention state differs by ${n(f.logicalCacheWidth.value / f.absorbedCacheWidth.value)} times in scalar count.`,
          `Serving fusions and distributed schedules are code-inspected, not benchmarked or executed here.`,
          `Packed expert weights do not imply that attention projections or shared experts use the same storage precision.`,
        ]}
      />
    </Section>
  );
}
export function Weights() {
  return (
    <Section
      id="weights"
      title="Where the weights live"
      source={[
        { source: "reference", symbol: "KimiSparseMoeBlock.__init__" },
        { source: "serving", symbol: "KimiK3DeltaAttention.__init__" },
      ]}
      lede="Resident routed expert matrices dominate. The ledger counts logical unquantized parameter shapes, not packed storage bytes or the parameters active for one token."
    >
      <Ledger
        caption="Publisher text decoder logical parameters"
        unit="Parameters"
        rows={f.ledger.value}
        footnote="All routed expert matrices counted once; head is untied."
      />
      <Prose>{f.parameterNote}</Prose>
      <Table
        caption="Implementation differences and reconciliation"
        head={["Reconciliation item", "Count"]}
        rows={[
          ["Text decoder", n(f.parameterTotal.value)],
          ["Vision tower and projector", n(f.visionParams.value)],
          ["A_log surplus (inferred from a serving-code comment)", n(f.serializationPadding.value)],
          ["Hub safetensors.total", n(f.hubTotal)],
          ["Remainder after that inference", n(f.unexplainedDifference - f.serializationPadding.value)],
        ]}
      />
      <Takeaways
        items={[
          "Resident and active parameter counts answer different questions.",
          "The separate vision ledger accounts for most of the gap to the hub total; the rest matches an A_log surplus inferred from a serving-code comment, not a verified tensor shape.",
          "Logical parameter counts do not measure packed checkpoint byte size.",
        ]}
      />
    </Section>
  );
}
