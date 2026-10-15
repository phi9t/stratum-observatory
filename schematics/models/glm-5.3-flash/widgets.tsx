import { useId } from "react";

import { approx, Figure, n, type Lane } from "../../kit";
import * as f from "./facts";

function laneForLayer(layer: number): Lane {
  const kind = f.layerKind(layer);
  return kind === "sparse-moe" ? "high" : kind === "kda-moe" ? "low" : "instance";
}

export function ArchitectureCircuit({ selected }: { selected: number }) {
  const title = useId();
  const tile = 18;
  const gap = 4;
  const left = 24;
  const top = 64;
  const width = left * 2 + f.layers.value * (tile + gap) - gap;
  return (
    <Figure
      caption={
        <>
          Sparse attention appears at layers where <code>(i + 1) % 4 == 0</code>. The selected tile is layer{" "}
          {n(selected)}.
        </>
      }
      label="GLM-5.3-Flash layer circuit"
    >
      <svg
        aria-labelledby={title}
        className="ms-fig"
        height={180}
        role="img"
        viewBox={`0 0 ${width} 180`}
        width={width}
      >
        <title id={title}>GLM-5.3-Flash layer circuit</title>
        <text className="ms-fig__column" x={left} y={22}>
          45 base decoder layers · KDA KDA KDA sparse repeats
        </text>
        {Array.from({ length: f.layers.value }, (_unused, layer) => {
          const x = left + layer * (tile + gap);
          const sparse = f.isSparseAttentionLayer(layer);
          const dense = !f.isMoELayer(layer);
          return (
            <g key={layer}>
              <rect
                className="ms-scale__tensor"
                data-lane={laneForLayer(layer)}
                height={tile}
                rx={3}
                width={tile}
                x={x}
                y={top}
              />
              {selected === layer ? (
                <rect className="ms-grid__row" height={tile + 8} width={tile + 8} x={x - 4} y={top - 4} />
              ) : null}
              <text className="ms-fig__note" textAnchor="middle" x={x + tile / 2} y={top + 38}>
                {layer % 4 === 3 || layer < f.denseFirst.value ? n(layer) : ""}
              </text>
              {sparse ? <path className="ms-wire ms-wire--state" d={`M ${x + tile / 2} ${top - 8} v 8`} /> : null}
              {dense ? (
                <circle className="ms-router__bar" data-shared="true" cx={x + tile / 2} cy={top + tile + 18} r={3} />
              ) : null}
            </g>
          );
        })}
        <g transform={`translate(${left} 126)`}>
          <rect className="ms-scale__tensor" data-lane="instance" height={12} width={30} x={0} y={0} />
          <text className="ms-fig__note" x={40} y={10}>
            KDA + dense MLP prefix
          </text>
          <rect className="ms-scale__tensor" data-lane="low" height={12} width={30} x={210} y={0} />
          <text className="ms-fig__note" x={250} y={10}>
            KDA + MoE
          </text>
          <rect className="ms-scale__tensor" data-lane="high" height={12} width={30} x={370} y={0} />
          <text className="ms-fig__note" x={410} y={10}>
            sparse MLA + own indexer + MoE
          </text>
        </g>
      </svg>
    </Figure>
  );
}

export function MhcDiagram() {
  const title = useId();
  const lanes = Array.from({ length: f.hcMult.value }, (_unused, i) => i);
  return (
    <Figure
      caption="Each attention and MLP sublayer receives a collapsed stream and returns expanded streams."
      label="mHC residual streams"
    >
      <svg aria-labelledby={title} className="ms-fig" height={250} role="img" viewBox="0 0 860 250" width={860}>
        <title id={title}>mHC residual streams</title>
        <text className="ms-fig__column" x={24} y={24}>
          four-stream mHC residual
        </text>
        {lanes.map((lane) => (
          <g key={lane}>
            <rect
              className="ms-scale__tensor"
              data-lane="skip"
              height={20}
              rx={4}
              width={150}
              x={28}
              y={54 + lane * 32}
            />
            <text className="ms-node__line" x={42} y={68 + lane * 32}>
              stream {n(lane)}
            </text>
          </g>
        ))}
        <g className="ms-node" data-lane="skip">
          <rect height={76} rx={7} width={120} x={240} y={74} />
          <text className="ms-node__title" textAnchor="middle" x={300} y={104}>
            pre
          </text>
          <text className="ms-node__line" textAnchor="middle" x={300} y={122}>
            collapse
          </text>
        </g>
        <g className="ms-node" data-lane="high">
          <rect height={76} rx={7} width={135} x={424} y={74} />
          <text className="ms-node__title" textAnchor="middle" x={491.5} y={104}>
            attention or MLP
          </text>
          <text className="ms-node__line" textAnchor="middle" x={491.5} y={122}>
            [B, S, 4096]
          </text>
        </g>
        <g className="ms-node" data-lane="skip">
          <rect height={76} rx={7} width={120} x={622} y={74} />
          <text className="ms-node__title" textAnchor="middle" x={682} y={104}>
            post + comb
          </text>
          <text className="ms-node__line" textAnchor="middle" x={682} y={122}>
            expand + mix
          </text>
        </g>
        <path className="ms-wire" d="M 178 86 H 240" />
        <path className="ms-wire" d="M 178 118 H 240" />
        <path className="ms-wire" d="M 178 150 H 240" />
        <path className="ms-wire" d="M 360 112 H 424" />
        <path className="ms-wire" d="M 559 112 H 622" />
        {lanes.map((lane) => (
          <path className="ms-wire ms-wire--skip" d={`M 742 112 H 792 V ${64 + lane * 32} H 820`} key={lane} />
        ))}
        <text className="ms-fig__note" x={24} y={216}>
          Saved activation boundary: {n(f.layers.value)} x {n(f.hcMult.value)} x {n(f.hidden.value)} x{" "}
          {n(f.bytesPerBf16.value)} B = {n(f.mHcSavedBytesPerToken.value)} B/token.
        </text>
      </svg>
    </Figure>
  );
}

export function IndexPoolDiagram() {
  const title = useId();
  return (
    <Figure
      caption={
        <>
          The indexer selects {n(f.indexPoolsSelected.value)} pooled candidates, expands them back to raw indices, then
          appends up to {n(f.indexKpool.value - 1)} visible tail positions.
        </>
      }
      label="IndexPool sparse selection"
    >
      <svg aria-labelledby={title} className="ms-fig" height={350} role="img" viewBox="0 0 336 350" width={336}>
        <title id={title}>IndexPool sparse selection</title>
        <text className="ms-fig__column" x={20} y={24}>
          IndexPool: four keys to one scored key
        </text>
        {Array.from({ length: f.indexKpool.value }, (_unused, i) => (
          <g key={i}>
            <rect className="ms-scale__tensor" data-lane="high" height={28} rx={4} width={68} x={20} y={58 + i * 36} />
            <text className="ms-node__line" textAnchor="middle" x={54} y={76 + i * 36}>
              key {n(i)}
            </text>
            <path className="ms-wire" d={`M 88 ${72 + i * 36} H 112`} />
          </g>
        ))}
        <g className="ms-node" data-lane="high">
          <rect height={76} rx={7} width={92} x={112} y={84} />
          <text className="ms-node__title" textAnchor="middle" x={158} y={113}>
            weighted
          </text>
          <text className="ms-node__line" textAnchor="middle" x={158} y={131}>
            pool + APE
          </text>
        </g>
        <path className="ms-wire" d="M 204 122 H 226" />
        <g className="ms-node" data-lane="high">
          <rect height={76} rx={7} width={90} x={226} y={84} />
          <text className="ms-node__title" textAnchor="middle" x={271} y={113}>
            top pools
          </text>
          <text className="ms-node__line" textAnchor="middle" x={271} y={131}>
            {n(f.indexPoolsSelected.value)} pools
          </text>
        </g>
        <path className="ms-wire" d="M 271 160 V 198 H 204" />
        <g className="ms-node" data-lane="high">
          <rect height={76} rx={7} width={92} x={112} y={198} />
          <text className="ms-node__title" textAnchor="middle" x={158} y={227}>
            expand
          </text>
          <text className="ms-node__line" textAnchor="middle" x={158} y={245}>
            {n(f.indexTopK.value)} indices
          </text>
        </g>
        <path className="ms-wire" d="M 204 236 H 226" />
        <g className="ms-node" data-lane="instance">
          <rect height={76} rx={7} width={70} x={226} y={198} />
          <text className="ms-node__title" textAnchor="middle" x={261} y={227}>
            tail
          </text>
          <text className="ms-node__line" textAnchor="middle" x={261} y={245}>
            +{n(f.indexKpool.value - 1)}
          </text>
        </g>
        <text className="ms-fig__note" x={20} y={314}>
          Correction: base sparse layers are full-indexed.
        </text>
        <text className="ms-fig__note" x={20} y={332}>
          They do not consume another layer&apos;s top-k.
        </text>
      </svg>
    </Figure>
  );
}

export function MemoryBudgetBars() {
  const title = useId();
  const rows = [
    ["EP8 expert shard", f.memoryEvidence.routedExpertEp8GiB, "low"],
    ["EP8 static floor", f.memoryEvidence.staticFloorEp8GiB, "low"],
    ["EP8 + all-gather", f.memoryEvidence.staticPlusAllGatherEp8GiB, "instance"],
    ["Measured peak", f.memoryEvidence.measuredPeakEp8GiB, "instance"],
    ["EP4 static estimate", f.memoryEvidence.staticFloorEp4GiB, "high"],
  ] as const;
  const max = Math.max(...rows.map((row) => row[1]));
  return (
    <Figure
      caption="Memory values are TorchTitan campaign evidence or estimates cited in the training table."
      label="LoRA SFT memory budget"
    >
      <svg aria-labelledby={title} className="ms-fig" height={340} role="img" viewBox="0 0 336 340" width={336}>
        <title id={title}>LoRA SFT memory budget</title>
        <text className="ms-fig__column" x={20} y={24}>
          per-GPU memory budget
        </text>
        {rows.map((row, index) => {
          const y = 52 + index * 48;
          const w = (row[1] / max) * 296;
          return (
            <g key={row[0]}>
              <text className="ms-fig__note" x={20} y={y}>
                {row[0]}
              </text>
              <text className="ms-node__line" textAnchor="end" x={316} y={y}>
                {row[1].toFixed(2)} GiB
              </text>
              <rect className="ms-gauge__track" height={16} rx={4} width={296} x={20} y={y + 10} />
              <rect className="ms-gauge__fill" data-lane={row[2]} height={16} rx={4} width={w} x={20} y={y + 10} />
            </g>
          );
        })}
        <text className="ms-fig__note" x={20} y={302}>
          Full fine-tuning is derived from text params,
        </text>
        <text className="ms-fig__note" x={20} y={320}>
          fp32 optimizer state and bf16 weights: about {f.fullFinetuneTb.value.toFixed(1)} TB.
        </text>
      </svg>
    </Figure>
  );
}

export function ParallelismDiagram({ mode }: { mode: "fsdp" | "ep" }) {
  const title = useId();
  const ep = mode === "ep";
  const shownRanks = Array.from({ length: f.memoryEvidence.epDegree / 2 }, (_unused, rank) => rank);
  return (
    <Figure
      caption={
        ep
          ? "Expert parallelism keeps expert weights sharded and moves token batches to the owning rank before combining them."
          : "Plain FSDP all-gathers expert weights around each sparse layer and microbatch; the token path stays local."
      }
      label="FSDP versus expert parallel dispatch"
    >
      <svg aria-labelledby={title} className="ms-fig" height={350} role="img" viewBox="0 0 336 350" width={336}>
        <title id={title}>FSDP versus expert parallel dispatch</title>
        <text className="ms-fig__column" x={20} y={24}>
          {ep ? "EP token movement" : "FSDP expert-weight movement"}
        </text>
        {shownRanks.map((rank) => (
          <g key={rank} transform={`translate(${20 + rank * 62} 58)`}>
            <rect
              className="ms-scale__tensor"
              data-lane={ep ? "low" : "high"}
              height={76}
              rx={8}
              width={50}
              x={0}
              y={0}
            />
            <text className="ms-node__title" textAnchor="middle" x={25} y={22}>
              r{n(rank)}
            </text>
            <text className="ms-node__line" textAnchor="middle" x={25} y={44}>
              {ep ? `${n(f.epExpertsPerRank.value)} exp` : "FSDP"}
            </text>
            <text className="ms-node__line" textAnchor="middle" x={25} y={62}>
              {ep ? "local" : "shard"}
            </text>
          </g>
        ))}
        <g transform="translate(268 58)">
          <rect
            className="ms-scale__tensor"
            data-lane={ep ? "low" : "high"}
            height={76}
            rx={8}
            width={48}
            x={0}
            y={0}
          />
          <text className="ms-node__title" textAnchor="middle" x={24} y={32}>
            ...
          </text>
          <text className="ms-node__line" textAnchor="middle" x={24} y={56}>
            r4-r7
          </text>
        </g>
        {ep ? (
          <>
            <g transform="translate(20 188)">
              <rect className="ms-scale__tensor" data-lane="instance" height={38} rx={6} width={70} x={0} y={0} />
              <text className="ms-node__title" textAnchor="middle" x={35} y={24}>
                tokens
              </text>
              <path className="ms-wire ms-wire--state" d="M 74 19 H 102" />
              <rect className="ms-scale__tensor" data-lane="low" height={38} rx={6} width={82} x={106} y={0} />
              <text className="ms-node__title" textAnchor="middle" x={147} y={24}>
                dispatch
              </text>
              <path className="ms-wire" d="M 192 19 H 212" />
              <rect className="ms-scale__tensor" data-lane="low" height={38} rx={6} width={66} x={216} y={0} />
              <text className="ms-node__title" textAnchor="middle" x={249} y={24}>
                join
              </text>
            </g>
            <path className="ms-wire ms-wire--state" d="M 168 134 V 188" />
            <text className="ms-fig__note" textAnchor="middle" x={168} y={168}>
              grouped GEMM on owner ranks
            </text>
            <text className="ms-fig__note" textAnchor="middle" x={168} y={276}>
              four ranks shown; ellipsis stands for ranks 4-7
            </text>
            <text className="ms-fig__note" textAnchor="middle" x={168} y={296}>
              EP{n(f.memoryEvidence.epDegree)} = {n(f.memoryEvidence.epDegree)} ranks x {n(f.epExpertsPerRank.value)}{" "}
              routed experts
            </text>
            <text className="ms-fig__note" textAnchor="middle" x={168} y={316}>
              = {n(f.experts.value)} total experts
            </text>
          </>
        ) : (
          <>
            <path className="ms-wire ms-wire--skip" d="M 45 174 H 292" />
            <path className="ms-wire ms-wire--skip" d="M 168 134 V 232" />
            <text className="ms-fig__note" textAnchor="middle" x={168} y={202}>
              all-gather expert weights
            </text>
            <text className="ms-fig__note" textAnchor="middle" x={168} y={220}>
              for each sparse layer and microbatch
            </text>
            <text className="ms-fig__note" textAnchor="middle" x={168} y={276}>
              token rows stay local; expert weights move around the ranks
            </text>
          </>
        )}
      </svg>
    </Figure>
  );
}

export function ContextBudgetFigure({ tokens }: { tokens: number }) {
  const title = useId();
  const mhcGiB = (tokens * f.mHcSavedBytesPerToken.value) / 1024 ** 3;
  const perLayerMhcGiB = (tokens * f.mHcSavedPerLayerBytesPerToken.value) / 1024 ** 3;
  const scoreScalars = tokens * f.indexHeads.value * Math.ceil(tokens / f.indexKpool.value);
  const scoreGiB = (scoreScalars * f.bytesPerBf16.value) / 1024 ** 3;
  const budgetGiB = f.activationBudgetEp8GiB.value;
  const rows = [
    ["activation room", budgetGiB, "instance"],
    ["mHC residuals · whole model", mhcGiB, "skip"],
    ["mHC residuals · per layer", perLayerMhcGiB, "skip"],
    ["indexer · per sparse layer", scoreGiB, "high"],
  ] as const;
  const max = Math.max(...rows.map((row) => row[1]), 1);
  return (
    <Figure
      caption="All long-context bars are estimates. The score term is shown for one sparse layer; the mHC residual term is whole-model unless labelled per layer."
      label="Long-context activation budget"
    >
      <svg aria-labelledby={title} className="ms-fig" height={348} role="img" viewBox="0 0 336 348" width={336}>
        <title id={title}>Long-context activation budget</title>
        <text className="ms-fig__column" x={20} y={24}>
          derived budget at {n(tokens)} tokens
        </text>
        {rows.map((row, index) => {
          const y = 54 + index * 52;
          const w = Math.max(2, (row[1] / max) * 296);
          return (
            <g key={row[0]}>
              <text className="ms-fig__note" x={20} y={y}>
                {row[0]}
              </text>
              <text className="ms-node__line" textAnchor="end" x={316} y={y}>
                {row[1].toFixed(2)} GiB est.
              </text>
              <rect className="ms-gauge__track" height={16} rx={4} width={296} x={20} y={y + 10} />
              <rect className="ms-gauge__fill" data-lane={row[2]} height={16} rx={4} width={w} x={20} y={y + 10} />
            </g>
          );
        })}
        <text className="ms-fig__note" x={20} y={274}>
          Activation room estimate: {f.memoryEvidence.b200CapacityGiB.toFixed(2)} GiB -{" "}
          {f.memoryEvidence.staticFloorEp8GiB.toFixed(2)} static -
        </text>
        <text className="ms-fig__note" x={20} y={292}>
          {f.memoryEvidence.activationBufferReserveGiB.toFixed(2)} buffers = {budgetGiB.toFixed(2)} GiB/GPU.
        </text>
        <text className="ms-fig__note" x={20} y={314}>
          score shape per sparse layer:
        </text>
        <text className="ms-fig__note" x={20} y={332}>
          [L, {n(f.indexHeads.value)} heads, ceil(L / {n(f.indexKpool.value)})].
        </text>
      </svg>
    </Figure>
  );
}

export function ParameterSummary() {
  const text = f.textBackboneStoredParameters.value;
  return `${approx(text)} text stored tensors, ${approx(f.mtpParameters.value)} MTP, ${approx(
    f.visionParameters.value,
  )} vision`;
}
