import {
  ArrowDefs,
  Figure,
  Node,
  nodeWidth,
  Wire,
  ColumnLabel,
  Section,
  Table,
  Tensor,
  Takeaways,
  Boundary,
  n,
  shape,
  useUrlNumber,
} from "../../../kit";
import * as f from "../facts";
const LAYOUT = { width: 1100, height: 360, tile: 12, gap: 4, columns: 24 };
export function Overview() {
  const [layer] = useUrlNumber("layer", 0, 0, f.layers.value - 1);
  return (
    <Section
      id="overview"
      title="The whole circuit"
      source={[
        { source: "reference", symbol: "KimiLinearModel.forward" },
        { source: "multimodal", symbol: "KimiK3ForConditionalGeneration.forward" },
      ]}
      lede="A hybrid text decoder keeps both recurrent token memory and a bank of completed depth blocks. Media features can replace input placeholders before that decoder begins."
    >
      <Figure
        label="Kimi-K3 whole-model circuit"
        caption="Layer tiles use blue for KDA and orange for full attention; the dense prefix has an ink outline. Violet denotes token state and media inputs, teal depth memory; ink is the stream. Geometry is schematic, not tensor scale."
      >
        <svg
          className="ms-fig"
          role="img"
          aria-label="Token and media input, hybrid decoder, depth aggregation and untied head"
          width={LAYOUT.width}
          height={LAYOUT.height}
          viewBox={`0 0 ${LAYOUT.width} ${LAYOUT.height}`}
        >
          <ArrowDefs id="k3-overview-arrow" />
          <ColumnLabel x={20}>input</ColumnLabel>
          <ColumnLabel x={155}>embed</ColumnLabel>
          <ColumnLabel x={330}>{`${n(f.layers.value)} decoder layers`}</ColumnLabel>
          <ColumnLabel x={780}>output</ColumnLabel>
          <Node x={20} y={100} h={46} lane="merged" kind="tensor" title="token ids" lines={[shape("B", "S")]} />
          <Wire d="M 115 123 H 155" arrow="k3-overview-arrow" />
          <Node
            x={155}
            y={100}
            h={46}
            lane="merged"
            title="embedding"
            lines={[`${n(f.vocab.value)} × ${n(f.hidden.value)}`]}
          />
          <Wire
            d={`M ${155 + nodeWidth("embedding", [`${n(f.vocab.value)} × ${n(f.hidden.value)}`])} 123 H 330`}
            arrow="k3-overview-arrow"
          />
          <g className="ms-node" data-lane="merged">
            <rect x={330} y={72} width={420} height={132} rx={6} />
            <text x={540} y={98} textAnchor="middle" className="ms-node__title">
              Hybrid decoder
            </text>
          </g>
          {Array.from({ length: f.layers.value }, (_, i) => (
            <rect
              key={i}
              x={346 + (i % LAYOUT.columns) * (LAYOUT.tile + LAYOUT.gap)}
              y={118 + Math.floor(i / LAYOUT.columns) * (LAYOUT.tile + LAYOUT.gap)}
              width={LAYOUT.tile}
              height={LAYOUT.tile}
              data-lane={f.isKda(i) ? "high" : "low"}
              style={{
                fill: i === layer ? "var(--ms-mark)" : "var(--lane-tint)",
                stroke: i === 0 ? "var(--ms-ink)" : "var(--lane)",
              }}
            />
          ))}
          <Wire d="M 750 123 H 780" arrow="k3-overview-arrow" />
          <Node x={780} y={100} h={46} lane="skip" title="depth mix" lines={[n(f.hidden.value)]} />
          <Wire d={`M ${780 + nodeWidth("depth mix", [n(f.hidden.value)])} 123 H 914`} arrow="k3-overview-arrow" />
          <Node x={914} y={100} h={46} lane="merged" title="RMSNorm" lines={[n(f.hidden.value)]} />
          <Wire d="M 965 146 V 215" arrow="k3-overview-arrow" />
          <Node
            x={912}
            y={215}
            h={46}
            lane="merged"
            title="untied head"
            lines={[`${n(f.hidden.value)} → ${n(f.vocab.value)}`]}
          />
          <Wire d="M 965 261 V 302" arrow="k3-overview-arrow" />
          <Node
            x={912}
            y={302}
            h={44}
            lane="merged"
            kind="tensor"
            title="logits"
            lines={[shape("B", "S", f.vocab.value)]}
          />
          <Node
            x={155}
            y={252}
            h={46}
            lane="instance"
            title="vision + projector"
            lines={[`${n(f.visionLayers.value)} layers → ${n(f.hidden.value)}`]}
          />
          <Wire d="M 235 252 V 146" lane="instance" kind="state" arrow="k3-overview-arrow" />
          <Node
            x={440}
            y={252}
            h={46}
            lane="skip"
            kind="state"
            title="depth bank"
            lines={[`${n(f.bankBlocks.value)} completed blocks`]}
          />
          <Wire d="M 530 252 V 204" lane="skip" kind="skip" arrow="k3-overview-arrow" />
        </svg>
      </Figure>
      <Table
        caption="Stages of the text forward pass"
        head={["Stage", "What happens", "Output"]}
        rows={[
          [
            "Embedding",
            "Look up token rows, or accept projected media embeddings.",
            <Tensor name="x" shape={shape("B", "S", f.hidden.value)} />,
          ],
          [
            "Hybrid stack",
            `${n(f.kdaCount.value)} KDA and ${n(f.fullCount.value)} full-attention layers; depth mixtures feed each sublayer.`,
            <Tensor name="prefix" shape={shape("B", "S", f.hidden.value)} />,
          ],
          [
            "Final depth mix",
            "Combine the live prefix with completed blocks before the final norm.",
            <Tensor name="hidden" shape={shape("B", "S", f.hidden.value)} />,
          ],
          [
            "Untied head",
            "A separate matrix scores the vocabulary.",
            <Tensor name="logits" shape={shape("B", "S", f.vocab.value)} />,
          ],
        ]}
      />
      <Boundary>
        Text prefill arithmetic is drawn at operation level. The vision interface is drawn separately; full vision
        internals and quantized execution are scoped out. Whole-model GPU numerical parity remains unverified.
      </Boundary>
      <Takeaways
        items={[
          "Token-time memory and depth memory have different lifetimes and axes.",
          "Full attention has no text rotary transform in this checkpoint.",
          "The output head is separate from the token embedding.",
        ]}
      />
    </Section>
  );
}
