import { Section, Ledger, Prose, Boundary, Takeaways, n } from "../../../kit";
import * as f from "../facts";
export function Weights() {
  return (
    <Section
      id="weights"
      title="Stored weights are not active weights"
      source={[{ source: "reference", symbol: "Transformer.__init__" }]}
      lede="This ledger counts logical scalar weights in the text backbone, including all routed experts and Engram tables. It is checked by the clean shape-mode forward pass."
    >
      <Ledger
        caption="Logical text-backbone parameter ledger"
        unit="Logical parameters"
        rows={f.ledger.value}
        footnote="Text backbone; no quantization scales, vision or draft modules."
      />
      <Prose>
        One routed expert contains {n(f.expertParams.value)} logical weights. Only {n(f.topK.value)} routed experts per
        layer process each token, but all {n(f.experts.value)} sets remain stored. Engram has a large input-dependent
        lookup table; table storage is not a dense compute-per-token count.
      </Prose>
      <Boundary>{f.parameterNote}</Boundary>
      <Takeaways
        items={[
          "Stored parameter count includes inactive routed experts.",
          "Engram table size does not imply every row is read for every token.",
          "Logical weights, packed elements and auxiliary storage are distinct inventories.",
        ]}
      />
    </Section>
  );
}
