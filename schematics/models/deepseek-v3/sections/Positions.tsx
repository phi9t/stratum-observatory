import { Figure, Section, Slider, Sentence, Readout, Check, Takeaways, n, useUrlNumber } from "../../../kit";
import * as f from "../facts";
const LAYOUT = { radius: 34, pitch: 150, left: 60, top: 64, height: 155, width: 660 };
export function Positions() {
  const [position, setPosition] = useUrlNumber("position", f.DEMO.context, 0, f.positions.value - 1);
  return (
    <Section
      id="positions"
      title="Positions at different frequencies"
      source={[
        { source: "reference", symbol: "DeepseekV3YarnRotaryEmbedding" },
        { source: "reference", symbol: "apply_rotary_pos_emb" },
      ]}
      lede="Position rotates only the designated query and key channels. YaRN preserves high frequencies and slows low frequencies, with a smooth transition between them."
    >
      <Slider label="Absolute position" min={0} max={f.positions.value - 1} value={position} onChange={setPosition} />
      <Sentence>
        At position {n(position)}, each channel pair has a different phase. The lowest-frequency pairs use a frequency
        divided by {n(f.ropeFactor.value)}.
      </Sentence>
      <Figure
        label="YaRN rotation dials"
        caption="These are the checkpoint's blended YaRN frequencies, not plain RoPE. Rays show phase modulo a complete turn; vector magnitude is unchanged because the cosine/sine amplitude ratio is unity here."
      >
        <svg
          role="img"
          aria-label="Phase of selected rotary pairs"
          className="ms-fig ms-dial"
          width={LAYOUT.width}
          height={LAYOUT.height}
          viewBox={`0 0 ${LAYOUT.width} ${LAYOUT.height}`}
        >
          {f.DEMO.pairs.map((pair, index) => {
            const angle = position * f.yarnFrequency(pair);
            const cx = LAYOUT.left + index * LAYOUT.pitch;
            return (
              <g key={pair}>
                <circle cx={cx} cy={LAYOUT.top} r={LAYOUT.radius} />
                <line data-ghost="true" x1={cx} x2={cx + LAYOUT.radius} y1={LAYOUT.top} y2={LAYOUT.top} />
                <line
                  x1={cx}
                  x2={cx + LAYOUT.radius * Math.cos(angle)}
                  y1={LAYOUT.top}
                  y2={LAYOUT.top - LAYOUT.radius * Math.sin(angle)}
                />
                <text className="ms-fig__note" textAnchor="middle" x={cx} y={118}>
                  pair {n(pair)}
                </text>
                <text className="ms-node__line" textAnchor="middle" x={cx} y={136}>
                  {n(angle % (2 * Math.PI))} rad
                </text>
              </g>
            );
          })}
        </svg>
      </Figure>
      <Readout
        label="YaRN values"
        items={[
          { term: "Rotated width", value: n(f.rotary.value), note: `${n(f.rotaryPairs.value)} pairs` },
          {
            term: "Correction bounds",
            value: `${n(f.correctionLow.value)}…${n(f.correctionHigh.value)}`,
            note: "frequency pair indices",
          },
          { term: "Attention scale", value: n(f.attentionScale.value), note: "modified separately" },
          { term: "Configured positions", value: n(f.positions.value), note: "not a measured capability" },
        ]}
      />
      <Check question="If cosine/sine amplitude stays unchanged, why is the attention scale still different from plain inverse square root scaling?">
        The amplitude ratio divides matching mscale terms, giving unity; attention independently multiplies its
        dot-product scale by the all-dimension mscale squared. Cancelling in the rotary table does not cancel that
        second use.
      </Check>
      <Takeaways
        items={[
          `Only ${n(f.rotary.value)} of each head's ${n(f.qDim.value)} query/key channels rotate.`,
          `The original position span ${n(f.original.value)} and factor ${n(f.ropeFactor.value)} determine frequency blending; they do not establish measured long-context quality.`,
        ]}
      />
    </Section>
  );
}
