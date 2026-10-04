import { useId } from "react";

import { ArrowDefs, ColumnLabel, Figure, Node, Note, Section, Table, Tensor, Wire, n, shape } from "../../../kit";
import * as f from "../facts";

const LAYOUT = {
  width: 1100,
  height: 250,
  y: 96,
  ids: 16,
  wte: 150,
  wpe: 150,
  add: 330,
  stack: 400,
  stackW: 330,
  norm: 780,
  head: 920,
  tile: 22,
};

export function Overview() {
  const titleId = useId();
  const arrow = useId();
  const L = LAYOUT;
  const blocks = Array.from({ length: f.layers.value }, (_u, index) => index);
  return (
    <Section
      id="overview"
      lede="Token ids become vectors, a learned position vector is added, twelve identical blocks each add two corrections to the stream, and the last vector at each position is compared against every token embedding to score the next token."
      source={[
        { source: "reference", symbol: "GPT2Model.forward" },
        { source: "reference", symbol: "GPT2LMHeadModel.forward" },
      ]}
      title="The whole circuit"
    >
      <Figure
        caption="The output head has no weights of its own: it reuses the token embedding matrix, transposed. The blue line is that tie."
        label="Forward pass drawing"
      >
        <svg
          aria-labelledby={titleId}
          className="ms-fig"
          height={L.height}
          role="img"
          viewBox={`0 0 ${String(L.width)} ${String(L.height)}`}
          width={L.width}
        >
          <title id={titleId}>GPT-2 forward pass from token ids to logits</title>
          <ArrowDefs id={arrow} />
          <ColumnLabel x={L.ids}>input</ColumnLabel>
          <ColumnLabel x={L.wte}>embed</ColumnLabel>
          <ColumnLabel x={L.stack}>{`${n(f.layers.value)} blocks`}</ColumnLabel>
          <ColumnLabel x={L.norm}>final norm</ColumnLabel>
          <ColumnLabel x={L.head}>head</ColumnLabel>

          <Node
            h={44}
            kind="tensor"
            lane="merged"
            lines={[shape("B", "S")]}
            title="input_ids"
            w={104}
            x={L.ids}
            y={L.y - 22}
          />
          <Wire arrow={arrow} d={`M 120 ${String(L.y)} H ${String(L.wte)}`} />
          <Node
            h={44}
            lane="high"
            lines={[`${n(f.vocab.value)} × ${n(f.hidden.value)}`]}
            title="wte lookup"
            w={140}
            x={L.wte}
            y={L.y - 22}
          />
          <Node
            h={44}
            lane="instance"
            lines={[`${n(f.positions.value)} × ${n(f.hidden.value)}`]}
            title="wpe lookup"
            w={140}
            x={L.wpe}
            y={L.y + 54}
          />
          <Note lane="instance" x={L.wpe} y={L.y + 116}>
            position 0, 1, 2, … of each token
          </Note>
          <Wire arrow={arrow} d={`M 290 ${String(L.y)} H ${String(L.add - 12)}`} lane="high" />
          <Wire
            arrow={arrow}
            d={`M 290 ${String(L.y + 76)} H ${String(L.add)} V ${String(L.y + 12)}`}
            lane="instance"
          />
          <g className="ms-sum">
            <circle cx={L.add} cy={L.y} r={10} />
            <path d={`M ${String(L.add - 5)} ${String(L.y)} h 10 M ${String(L.add)} ${String(L.y - 5)} v 10`} />
          </g>
          <Wire arrow={arrow} d={`M ${String(L.add + 10)} ${String(L.y)} H ${String(L.stack)}`} />

          <rect className="ms-node__plate" height={84} rx={7} width={L.stackW} x={L.stack} y={L.y - 42} />
          <text className="ms-node__title" textAnchor="middle" x={L.stack + L.stackW / 2} y={L.y - 20}>
            pre-norm blocks · attention then MLP
          </text>
          {blocks.map((index) => (
            <g className="ms-overview__tap" key={index}>
              <rect height={L.tile} rx={3} width={L.tile} x={L.stack + 16 + index * (L.tile + 3)} y={L.y - 6} />
            </g>
          ))}
          <text className="ms-node__line" textAnchor="middle" x={L.stack + L.stackW / 2} y={L.y + 34}>
            {shape("B", "S", f.hidden.value)} in and out of every block
          </text>
          <Wire arrow={arrow} d={`M ${String(L.stack + L.stackW)} ${String(L.y)} H ${String(L.norm)}`} />
          <Node h={44} lane="merged" lines={[n(f.hidden.value)]} title="LayerNorm" w={100} x={L.norm} y={L.y - 22} />
          <Wire arrow={arrow} d={`M ${String(L.norm + 100)} ${String(L.y)} H ${String(L.head)}`} />
          <Node
            h={44}
            lane="high"
            lines={[`${n(f.hidden.value)} → ${n(f.vocab.value)}`]}
            title="lm_head"
            w={150}
            x={L.head}
            y={L.y - 22}
          />
          <Wire
            d={`M ${String(L.wte + 70)} ${String(L.y - 22)} V 40 H ${String(L.head + 75)} V ${String(L.y - 22)}`}
            kind="state"
            lane="high"
          />
          <Note anchor="middle" lane="high" x={(L.wte + L.head) / 2 + 70} y={34}>
            same matrix: lm_head.weight is wte.weight
          </Note>
          <Node
            h={40}
            kind="tensor"
            lane="merged"
            lines={[shape("B", "S", f.vocab.value)]}
            title="logits"
            w={150}
            x={L.head}
            y={L.y + 56}
          />
          <Wire arrow={arrow} d={`M ${String(L.head + 75)} ${String(L.y + 22)} V ${String(L.y + 56)}`} />
        </svg>
      </Figure>
      <Table
        caption="Stages of the forward pass"
        head={["Stage", "What happens", "Output"]}
        rows={[
          [
            "Token embedding",
            "Each id selects one row of wte.",
            <Tensor key="t" name="inputs_embeds" shape={shape("B", "S", f.hidden.value)} />,
          ],
          [
            "Position embedding",
            "Each position index selects one row of wpe; the two are added.",
            <Tensor key="t" name="hidden_states" shape={shape("B", "S", f.hidden.value)} />,
          ],
          [
            "Blocks",
            `${n(f.layers.value)} blocks, each: norm, attention, add, norm, MLP, add.`,
            <Tensor key="t" name="hidden_states" shape={shape("B", "S", f.hidden.value)} />,
          ],
          [
            "Final norm",
            "One more LayerNorm, because blocks normalise their inputs, not their outputs.",
            <Tensor key="t" name="hidden_states" shape={shape("B", "S", f.hidden.value)} />,
          ],
          [
            "Head",
            "A bias-free projection onto the token embeddings.",
            <Tensor key="t" name="logits" shape={shape("B", "S", f.vocab.value)} />,
          ],
        ]}
      />
    </Section>
  );
}
