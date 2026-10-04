import { FactList, Figure, Section, TensorScale, n } from "../../../kit";
import * as f from "../facts";

const LAYOUT = { shownTokens: 32, pxPerToken: 5, pxPerWidth: 0.045 };

export function Data() {
  const tokens = LAYOUT.shownTokens;
  return (
    <Section
      id="data"
      lede="The stream keeps one shape from the first block to the last. Inside a block it is briefly widened: three times for attention's joint q, k, v projection and four times inside the MLP."
      source={[{ source: "reference", symbol: "GPT2Model.forward" }]}
      title="Data representation"
    >
      <Figure
        caption={`Drawn for a ${n(tokens)}-token sequence. The MLP's hidden activation is the widest tensor the model ever makes; the key/value cache keeps two stream-width tensors per layer.`}
        label="Tensor scale drawing"
      >
        <TensorScale
          footnote="To scale: one column is one token, height is width, area is scalars per sequence."
          pxPerToken={LAYOUT.pxPerToken}
          pxPerWidth={LAYOUT.pxPerWidth}
          stages={[
            { label: "residual stream", tensors: [{ key: "x", name: "hidden_states", tokens, width: f.hidden.value, lane: "merged" }] },
            { label: "inside attention", tensors: [{ key: "qkv", from: "x", name: "q, k, v joined", tokens, width: f.qkvWidth.value, lane: "high", note: "split three ways" }] },
            { label: "after the first add", tensors: [{ key: "x2", from: "qkv", name: "hidden_states", tokens, width: f.hidden.value, lane: "merged" }] },
            { label: "inside the MLP", tensors: [{ key: "mlp", from: "x2", name: "c_fc output", tokens, width: f.mlpWidth.value, lane: "low" }] },
            { label: "after the second add", tensors: [{ key: "y", from: "mlp", name: "hidden_states", tokens, width: f.hidden.value, lane: "merged" }] },
          ]}
          title="GPT-2 tensors drawn to scale"
        />
      </Figure>
      <FactList
        items={[
          ["Input", <>Integer token ids, at most {n(f.positions.value)} per sequence: positions come from a learned table with exactly that many rows, so there is no position {n(f.positions.value)}.</>],
          ["Output", <>One score per vocabulary entry ({n(f.vocab.value)}) at every position. Token {n(f.eosToken.value)} doubles as beginning and end of text.</>],
          ["Order", <>Nothing but the position table tells the model where a token is. Attention itself is order-blind.</>],
        ]}
      />
    </Section>
  );
}
