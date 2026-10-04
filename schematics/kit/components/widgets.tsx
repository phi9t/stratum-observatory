import { useId, type ReactNode } from "react";

import { approx, n } from "../lib/facts";
import { normals } from "../lib/rng";
import type { Lane } from "./shell";

// ---------------------------------------------------------------------------
// LayerMap: every layer is a tile. Kinds colour the tiles and make the legend.

export type LayerKind = Readonly<{ id: string; label: string; lane: Lane }>;

export function LayerMap({
  label,
  kinds,
  kindOf,
  count,
  selected,
  onSelect,
  first = 0,
}: {
  label: string;
  kinds: readonly LayerKind[];
  /** Kind id for layer index i (0-based position in the stack). */
  kindOf: (index: number) => string;
  count: number;
  selected: number;
  onSelect: (index: number) => void;
  /** Number shown on the first tile; use the implementation's own indexing. */
  first?: number;
}) {
  const lane = (index: number): Lane => {
    const kind = kinds.find((candidate) => candidate.id === kindOf(index));
    if (kind === undefined) {
      throw new Error(`LayerMap: layer ${String(index)} has unknown kind "${kindOf(index)}"`);
    }
    return kind.lane;
  };
  return (
    <div className="ms-layermap">
      <ol aria-label={label} className="ms-layermap__tiles">
        {Array.from({ length: count }, (_u, index) => (
          <li key={index}>
            <button
              aria-label={`Layer ${String(index + first)}, ${kindOf(index)}`}
              aria-pressed={index === selected}
              data-lane={lane(index)}
              onClick={() => onSelect(index)}
              type="button"
            >
              {index + first}
            </button>
          </li>
        ))}
      </ol>
      <ul aria-label={`${label} legend`} className="ms-layermap__legend">
        {kinds.map((kind) => (
          <li key={kind.id}>
            <span aria-hidden="true" className="ms-swatch" data-lane={kind.lane} />
            {kind.label} (
            {String(Array.from({ length: count }, (_u, i) => i).filter((i) => kindOf(i) === kind.id).length)})
          </li>
        ))}
      </ul>
    </div>
  );
}

// ---------------------------------------------------------------------------
// AttentionGrid: which key positions a query position can read.

export type CellState = "masked" | "attend" | "window" | "sparse" | "self";

export function AttentionGrid({
  title,
  size,
  query,
  onQuery,
  cell,
  legend,
}: {
  title: string;
  /** Positions shown on each axis (keep to 16–48). */
  size: number;
  query: number;
  onQuery: (position: number) => void;
  /** State of the cell (query row q, key column k). */
  cell: (q: number, k: number) => CellState;
  legend: readonly Readonly<{ state: CellState; label: string }>[];
}) {
  const titleId = useId();
  const px = Math.max(8, Math.min(18, Math.floor(432 / size)));
  const left = 64;
  const top = 34;
  const side = size * px;
  const positions = Array.from({ length: size }, (_u, i) => i);
  return (
    <svg
      aria-labelledby={titleId}
      className="ms-fig"
      height={top + side + 34 + legend.length * 0}
      role="img"
      viewBox={`0 0 ${String(left + side + 240)} ${String(top + side + 34)}`}
      width={left + side + 240}
    >
      <title id={titleId}>{title}</title>
      <text className="ms-fig__column" x={left} y={18}>
        key position →
      </text>
      <text
        className="ms-fig__column"
        textAnchor="end"
        transform={`rotate(-90 ${String(left - 34)} ${String(top)})`}
        x={left - 34}
        y={top}
      >
        query ↓
      </text>
      {positions.map((q) =>
        positions.map((k) => (
          <rect
            className="ms-grid__cell"
            data-state={cell(q, k) === "masked" ? undefined : cell(q, k)}
            height={px}
            key={`${String(q)}-${String(k)}`}
            width={px}
            x={left + k * px}
            y={top + q * px}
          />
        )),
      )}
      <rect className="ms-grid__row" height={px} width={side} x={left} y={top + query * px} />
      <text className="ms-fig__note" textAnchor="end" x={left - 8} y={top + query * px + px / 2 + 4}>
        q={n(query)}
      </text>
      {positions.map((q) => (
        <rect
          className="ms-grid__hit"
          height={px}
          key={q}
          onClick={() => onQuery(q)}
          width={side}
          x={left}
          y={top + q * px}
        />
      ))}
      {legend.map((entry, index) => (
        <g key={entry.state}>
          <rect
            className="ms-grid__cell"
            data-state={entry.state === "masked" ? undefined : entry.state}
            height={12}
            width={12}
            x={left + side + 24}
            y={top + index * 22}
          />
          <text className="ms-fig__note" x={left + side + 44} y={top + index * 22 + 10.5}>
            {entry.label}
          </text>
        </g>
      ))}
      <text className="ms-fig__note" x={left} y={top + side + 22}>
        Row q={n(query)} reads {n(positions.filter((k) => cell(query, k) !== "masked").length)} of {n(size)} shown
        positions.
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// MoeRouter: seeded toy logits through the checkpoint's real routing rule.

export type Scoring = "softmax" | "sigmoid" | "sqrtsoftplus";

export type Routing = Readonly<{
  chosen: readonly number[];
  weights: readonly number[];
  chosenGroups: readonly number[];
  scores: readonly number[];
}>;

/**
 * The routing arithmetic, separate from the drawing so pages can test it.
 * `groups` / `topGroups` implement group-limited selection (keep the best groups by
 * the sum of each group's top two scores, then take top-k inside them).
 */
export function routeToken(options: {
  experts: number;
  topK: number;
  scoring: Scoring;
  seed: number;
  groups?: number;
  topGroups?: number;
  /** Renormalise the selected scores to sum to 1. */
  normalize: boolean;
  /** Multiplier applied to the selected weights. */
  scale?: number;
}): Routing {
  const logits = normals(options.seed, options.experts);
  const max = Math.max(...logits);
  const total = logits.reduce((sum, value) => sum + Math.exp(value - max), 0);
  const scores = logits.map((value) =>
    options.scoring === "softmax"
      ? Math.exp(value - max) / total
      : options.scoring === "sigmoid"
        ? 1 / (1 + Math.exp(-value))
        : Math.sqrt(Math.log1p(Math.exp(value))),
  );
  const groups = options.groups ?? 1;
  const perGroup = options.experts / groups;
  if (!Number.isInteger(perGroup)) {
    throw new RangeError("routeToken: experts must divide evenly into groups");
  }
  const groupScore = Array.from({ length: groups }, (_u, g) =>
    scores
      .slice(g * perGroup, (g + 1) * perGroup)
      .sort((a, b) => b - a)
      .slice(0, 2)
      .reduce((sum, value) => sum + value, 0),
  );
  const chosenGroups = groupScore
    .map((score, g) => [score, g] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, options.topGroups ?? groups)
    .map(([, g]) => g);
  const chosen = scores
    .map((score, index) => [score, index] as const)
    .filter(([, index]) => chosenGroups.includes(Math.floor(index / perGroup)))
    .sort((a, b) => b[0] - a[0])
    .slice(0, options.topK)
    .map(([, index]) => index);
  const selectedSum = chosen.reduce((sum, index) => sum + scores[index], 0);
  const weights = chosen.map(
    (index) => (options.normalize ? scores[index] / selectedSum : scores[index]) * (options.scale ?? 1),
  );
  return { chosen, weights, chosenGroups, scores };
}

export function MoeRouter({
  title,
  routing,
  experts,
  groups = 1,
  shared = 0,
}: {
  title: string;
  routing: Routing;
  experts: number;
  groups?: number;
  /** Always-on shared experts, drawn apart: they do not consume router logits. */
  shared?: number;
}) {
  const titleId = useId();
  const barW = Math.max(1, Math.min(6, Math.floor(1000 / experts)));
  const gap = barW >= 3 ? 1 : 0;
  const left = 24;
  const base = 150;
  const full = 110;
  const span = experts * barW;
  const peak = Math.max(...routing.scores);
  const perGroup = experts / groups;
  const sharedX = left + span + 36;
  return (
    <svg
      aria-labelledby={titleId}
      className="ms-fig"
      height={base + 44}
      role="img"
      viewBox={`0 0 ${String(sharedX + shared * 14 + 150)} ${String(base + 44)}`}
      width={sharedX + shared * 14 + 150}
    >
      <title id={titleId}>{title}</title>
      <text className="ms-fig__column" x={left} y={18}>
        router scores · {n(experts)} routed experts
      </text>
      {groups > 1
        ? Array.from({ length: groups }, (_u, g) => (
            <rect
              className="ms-router__group"
              data-chosen={routing.chosenGroups.includes(g) ? "true" : undefined}
              height={full + 10}
              key={g}
              width={perGroup * barW - 2}
              x={left + g * perGroup * barW + 1}
              y={base - full - 6}
            />
          ))
        : null}
      {routing.scores.map((score, index) => {
        const h = Math.max(1, (score / peak) * full);
        return (
          <rect
            className="ms-router__bar"
            data-chosen={routing.chosen.includes(index) ? "true" : undefined}
            height={h}
            key={index}
            width={barW - gap}
            x={left + index * barW}
            y={base - h}
          />
        );
      })}
      <line className="ms-router__axis" x1={left} x2={left + span} y1={base + 0.5} y2={base + 0.5} />
      <text className="ms-fig__note" x={left} y={base + 20}>
        selected:{" "}
        {[...routing.chosen]
          .sort((a, b) => a - b)
          .map(n)
          .join(", ")}
      </text>
      <text className="ms-fig__note" x={left} y={base + 36}>
        weights: {routing.weights.map((w) => w.toFixed(3)).join("  ")}
      </text>
      {shared > 0 ? (
        <>
          <text className="ms-fig__column" x={sharedX} y={18}>
            shared · always on
          </text>
          {Array.from({ length: shared }, (_u, s) => (
            <rect
              className="ms-router__bar"
              data-shared="true"
              height={full}
              key={s}
              width={10}
              x={sharedX + s * 14}
              y={base - full}
            />
          ))}
          <text className="ms-fig__note" x={sharedX} y={base + 20}>
            no router logit
          </text>
        </>
      ) : null}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// RopeDials: rotation of a few frequency pairs at one position.

export function RopeDials({
  title,
  position,
  pairs,
  theta,
  rotaryDim,
}: {
  title: string;
  position: number;
  /** Frequency-pair indices to show, e.g. [0, 1, 4, 15]. */
  pairs: readonly number[];
  theta: number;
  /** Channels that are rotated (two per pair). */
  rotaryDim: number;
}) {
  const titleId = useId();
  const r = 34;
  const pitch = 132;
  return (
    <svg
      aria-labelledby={titleId}
      className="ms-fig ms-dial"
      height={150}
      role="img"
      viewBox={`0 0 ${String(40 + pairs.length * pitch)} 150`}
      width={40 + pairs.length * pitch}
    >
      <title id={titleId}>{title}</title>
      {pairs.map((pair, index) => {
        const frequency = theta ** ((-2 * pair) / rotaryDim);
        const angle = position * frequency;
        const cx = 60 + index * pitch;
        const cy = 64;
        return (
          <g key={pair}>
            <circle cx={cx} cy={cy} r={r} />
            <line data-ghost="true" x1={cx} x2={cx + r} y1={cy} y2={cy} />
            <line x1={cx} x2={cx + r * Math.cos(angle)} y1={cy} y2={cy - r * Math.sin(angle)} />
            <text className="ms-fig__note" textAnchor="middle" x={cx} y={cy + r + 20}>
              pair {n(pair)}
            </text>
            <text className="ms-node__line" textAnchor="middle" x={cx} y={cy + r + 36}>
              {(angle % (2 * Math.PI)).toFixed(2)} rad
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Ledger: where parameters (or bytes) live. Values come from derived facts.

export function Ledger({
  caption,
  unit,
  rows,
  footnote,
}: {
  caption: string;
  unit: string;
  rows: readonly Readonly<{ label: string; detail: ReactNode; value: number; lane?: Lane }>[];
  footnote?: ReactNode;
}) {
  const total = rows.reduce((sum, row) => sum + row.value, 0);
  return (
    <div className="ms-table-scroll">
      <table className="ms-table ms-ledger">
        <caption>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Part</th>
            <th scope="col">Made of</th>
            <th scope="col">{unit}</th>
            <th scope="col">Share</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const share = (row.value / total) * 100;
            return (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                <td>{row.detail}</td>
                <td className="ms-num">
                  {n(row.value)} <span>({approx(row.value)})</span>
                </td>
                <td className="ms-ledger__share" data-lane={row.lane}>
                  <span
                    aria-hidden="true"
                    className="ms-ledger__bar"
                    style={{ inlineSize: `${String(Math.max(share, 0.4))}%` }}
                  />
                  <span>{share.toFixed(share < 0.1 ? 3 : 2)}%</span>
                </td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr>
            <th scope="row">Total</th>
            <td className="ms-budget__note">{footnote}</td>
            <td className="ms-num">
              {n(total)} <span>({approx(total)})</span>
            </td>
            <td>100%</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TensorScale: tensors drawn to scale. One column per token, height is width.

export type ScaleTensor = Readonly<{
  /** Stable key; a tensor in the next stage with `from` equal to this key is joined by a ribbon. */
  key: string;
  from?: string;
  name: string;
  tokens: number;
  width: number;
  lane: Lane;
  note?: string;
}>;

export function TensorScale({
  title,
  stages,
  pxPerToken,
  pxPerWidth,
  footnote,
}: {
  title: string;
  stages: readonly Readonly<{ label: string; tensors: readonly ScaleTensor[] }>[];
  /** Horizontal pixels per token and vertical pixels per unit of width; choose so the largest tensor is about 250 × 160. */
  pxPerToken: number;
  pxPerWidth: number;
  footnote: string;
}) {
  const titleId = useId();
  const top = 62;
  const colGap = 64;
  const rowGap = 64;
  type Placed = ScaleTensor & { x: number; y: number; w: number; h: number };
  const placed: Placed[][] = [];
  let x = 24;
  let maxBottom = 0;
  for (const stage of stages) {
    let y = top;
    const column: Placed[] = [];
    let colWidth = 150;
    for (const tensor of stage.tensors) {
      const w = Math.max(3, tensor.tokens * pxPerToken);
      const h = Math.max(3, tensor.width * pxPerWidth);
      column.push({ ...tensor, x, y, w, h });
      colWidth = Math.max(colWidth, w, tensor.name.length * 7.7);
      y += h + rowGap;
    }
    maxBottom = Math.max(maxBottom, y - rowGap);
    placed.push(column);
    x += colWidth + colGap;
  }
  const width = x - colGap + 24;
  const height = maxBottom + 58;
  const find = (stage: number, key: string): Placed | undefined => placed[stage]?.find((tensor) => tensor.key === key);
  return (
    <svg
      aria-labelledby={titleId}
      className="ms-fig ms-scale"
      height={height}
      role="img"
      viewBox={`0 0 ${String(width)} ${String(height)}`}
      width={width}
    >
      <title id={titleId}>{title}</title>
      <g className="ms-scale__ribbon">
        {placed.flatMap((column, stage) =>
          column.map((tensor) => {
            const source = tensor.from === undefined ? undefined : find(stage - 1, tensor.from);
            return source === undefined ? null : (
              <polygon
                data-lane={tensor.lane}
                key={`${String(stage)}:${tensor.key}`}
                points={`${source.x + source.w},${source.y} ${tensor.x},${tensor.y} ${tensor.x},${tensor.y + tensor.h} ${source.x + source.w},${source.y + source.h}`}
              />
            );
          }),
        )}
      </g>
      {placed.map((column, stage) => (
        <g key={stages[stage].label}>
          <text className="ms-fig__column" x={column[0]?.x ?? 0} y={20}>
            {stages[stage].label}
          </text>
          {column.map((tensor) => (
            <g key={tensor.key}>
              <text className="ms-scale__name" data-lane={tensor.lane} x={tensor.x} y={tensor.y - 22}>
                {tensor.name}
              </text>
              <text className="ms-scale__shape" x={tensor.x} y={tensor.y - 7}>
                {n(tensor.tokens)} × {n(tensor.width)}
              </text>
              <rect
                className="ms-scale__tensor"
                data-lane={tensor.lane}
                height={tensor.h}
                width={tensor.w}
                x={tensor.x}
                y={tensor.y}
              />
              <text className="ms-scale__count" x={tensor.x} y={tensor.y + tensor.h + 15}>
                {n(tensor.tokens * tensor.width)} scalars{tensor.note === undefined ? "" : ` · ${tensor.note}`}
              </text>
            </g>
          ))}
        </g>
      ))}
      <text className="ms-fig__note" x={24} y={height - 10}>
        {footnote}
      </text>
    </svg>
  );
}
