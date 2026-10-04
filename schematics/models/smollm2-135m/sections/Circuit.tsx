import { ArrowDefs, Node, Note, Wire, nodeWidth, n, shape } from "../../../kit";
import * as f from "../facts";
const LAYOUT = {
  width: 1100,
  height: 300,
  y: 130,
  ids: 16,
  embed: 160,
  stack: 320,
  stackW: 340,
  norm: 700,
  head: 850,
  tile: 16,
  gap: 4,
  inset: 20,
  perRow: 15,
  half: 22,
  tieY: 44,
  sideY: 224,
  plateH: 100,
};
export function Circuit({ layer }: { layer: number }) {
  const L = {
    ...LAYOUT,
    idsW: nodeWidth("token ids", [shape("B", "S")]),
    embedW: nodeWidth("embedding", [`${n(f.vocab.value)} × ${n(f.hidden.value)}`]),
    normW: nodeWidth("RMSNorm", [n(f.hidden.value)]),
    headW: nodeWidth("tied LM head", [`${n(f.hidden.value)} → ${n(f.vocab.value)}`]),
    sideW: nodeWidth("positions + causal mask", ["enter each attention branch"]),
  };
  return (
    <svg className="ms-fig" width={L.width} height={L.height} role="img" aria-labelledby="circuit-title">
      <title id="circuit-title">SmolLM2-135M: token embedding, decoder stack and tied output head</title>
      <ArrowDefs id="circuit-arrow" />
      <Node h={44} lane="merged" x={L.ids} y={L.y - L.half} title="token ids" lines={[shape("B", "S")]} kind="tensor" />
      <Wire arrow="circuit-arrow" d={`M ${L.ids + L.idsW} ${L.y} H ${L.embed}`} />
      <Node
        h={44}
        lane="merged"
        x={L.embed}
        y={L.y - L.half}
        title="embedding"
        lines={[`${n(f.vocab.value)} × ${n(f.hidden.value)}`]}
      />
      <Wire arrow="circuit-arrow" d={`M ${L.embed + L.embedW} ${L.y} H ${L.stack}`} />
      <rect className="ms-node__plate" x={L.stack} y={L.y - L.tieY} width={L.stackW} height={L.plateH} rx={7} />
      <text className="ms-node__title" x={L.stack + L.stackW / 2} y={L.y - L.half} textAnchor="middle">
        {n(f.layers.value)} pre-norm layers
      </text>
      {Array.from({ length: f.layers.value }, (_, i) => (
        <g className="ms-stack__tile" data-lane="high" data-selected={i === layer ? "true" : undefined} key={i}>
          <rect
            x={L.stack + L.inset + (i % L.perRow) * (L.tile + L.gap)}
            y={L.y + Math.floor(i / L.perRow) * (L.tile + L.gap)}
            width={L.tile}
            height={L.tile}
            rx={3}
          />
        </g>
      ))}
      <text className="ms-node__line" x={L.stack + L.stackW / 2} y={L.y + L.tieY + L.gap} textAnchor="middle">
        {shape("B", "S", f.hidden.value)}
      </text>
      <Wire arrow="circuit-arrow" d={`M ${L.stack + L.stackW} ${L.y} H ${L.norm}`} />
      <Node h={44} lane="merged" x={L.norm} y={L.y - L.half} title="RMSNorm" lines={[n(f.hidden.value)]} />
      <Wire arrow="circuit-arrow" d={`M ${L.norm + L.normW} ${L.y} H ${L.head}`} />
      <Node
        h={44}
        lane="merged"
        x={L.head}
        y={L.y - L.half}
        title="tied LM head"
        lines={[`${n(f.hidden.value)} → ${n(f.vocab.value)}`]}
      />
      <Wire
        d={`M ${L.embed + L.embedW / 2} ${L.y - L.half} V ${L.tieY} H ${L.head + L.headW / 2} V ${L.y - L.half}`}
        kind="state"
        lane="instance"
      />
      <Note x={(L.embed + L.head) / 2} y={32} anchor="middle" lane="instance">
        same embedding matrix · counted once
      </Note>
      <Node
        h={44}
        lane="merged"
        x={L.head}
        y={L.sideY}
        title="logits"
        lines={[shape("B", "S", f.vocab.value)]}
        kind="tensor"
      />
      <Wire arrow="circuit-arrow" d={`M ${L.head + L.headW / 2} ${L.y + L.half} V ${L.sideY}`} />
      <Node
        h={44}
        x={L.stack}
        y={L.sideY}
        title="positions + causal mask"
        lines={["enter each attention branch"]}
        lane="instance"
        kind="state"
      />
      <Wire
        arrow="circuit-arrow"
        d={`M ${L.stack + L.sideW / 2} ${L.sideY} V ${L.y - L.tieY + L.plateH}`}
        lane="instance"
        kind="state"
      />
    </svg>
  );
}
