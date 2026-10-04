import { useId, type KeyboardEvent, type ReactNode } from "react";

import { SourceRef, type Ref } from "../lib/sources";
import { useUrlChoice } from "../lib/urlState";
import { Button, Sentence } from "./controls";
import { Figure, type Lane } from "./shell";

// A block drawn op by op. You give ops on a (col, row) grid and edges between
// them; the kit sizes every box from its text, routes orthogonal wires, and
// keeps the selected op in the URL. Selecting an op fills the inspector.

export type OpDetail = Readonly<{
  /** One or two sentences: what this op does and why it is here. */
  what: string;
  /** Plain-text formula, e.g. "y = x * rsqrt(mean(x^2) + eps) * g". */
  formula?: string;
  /** Named inputs with shapes, e.g. [["x", shape("B", "S", d)]]. */
  inputs?: readonly (readonly [string, string])[];
  output?: readonly [string, string];
  /** Parameter tensors with shapes, or "none". */
  params?: readonly (readonly [string, string])[];
  /** Pinned definitions that implement this op, in every implementation you read. */
  refs: readonly Ref[];
  /** Anything a reader could get wrong: cached vs fresh, train vs inference, impl differences. */
  note?: string;
}>;

export type Op = Readonly<{
  id: string;
  label: string;
  /** Second line in mono: usually the output width or shape. */
  sub?: string;
  /** op = computation, tensor = named value, sum/mul = residual add or gate, state = cache or buffer. */
  kind?: "op" | "tensor" | "sum" | "mul" | "state";
  lane?: Lane;
  col: number;
  row: number;
  detail: OpDetail;
}>;

export type Edge = Readonly<{
  from: string;
  to: string;
  /** data = main path, skip = residual (routed above the row), state = cache read/write. */
  kind?: "data" | "skip" | "state";
  label?: string;
}>;

const TITLE_PX = 7.7; // per character, 12.5px sans semibold, measured wide
const LINE_PX = 6.75; // per character, 11px mono
const PAD = 26;
const GAP_X = 46;
const ROW_H = 78;
const NODE_H = 46;
const ROUND = 26;
const MARGIN = 24;
const SKIP_LANE = 18;

type Box = Readonly<{ op: Op; x: number; y: number; w: number; h: number }>;

function layout(ops: readonly Op[], skipCount: number) {
  const cols = Math.max(...ops.map((op) => op.col)) + 1;
  const rows = Math.max(...ops.map((op) => op.row)) + 1;
  const width = (op: Op): number =>
    op.kind === "sum" || op.kind === "mul"
      ? ROUND
      : Math.max(64, Math.ceil(Math.max(op.label.length * TITLE_PX, (op.sub?.length ?? 0) * LINE_PX) + PAD));
  const colWidth = Array.from({ length: cols }, (_u, col) =>
    Math.max(0, ...ops.filter((op) => op.col === col).map(width)),
  );
  const colX: number[] = [];
  let x = MARGIN;
  for (const w of colWidth) {
    colX.push(x);
    x += w + GAP_X;
  }
  const top = MARGIN + skipCount * SKIP_LANE + (skipCount > 0 ? 8 : 0);
  const boxes = new Map<string, Box>();
  for (const op of ops) {
    const w = width(op);
    const h = op.kind === "sum" || op.kind === "mul" ? ROUND : NODE_H;
    boxes.set(op.id, {
      op,
      w,
      h,
      x: colX[op.col] + (colWidth[op.col] - w) / 2,
      y: top + op.row * ROW_H + (NODE_H - h) / 2,
    });
  }
  return { boxes, width: x - GAP_X + MARGIN, height: top + rows * ROW_H - (ROW_H - NODE_H) + MARGIN, top };
}

function route(edge: Edge, from: Box, to: Box, skipIndex: number, top: number): string {
  const fy = from.y + from.h / 2;
  const ty = to.y + to.h / 2;
  if (edge.kind === "skip") {
    const lane = top - 8 - skipIndex * SKIP_LANE;
    const fx = from.x + from.w / 2;
    const tx = to.x + to.w / 2;
    return `M ${fx} ${from.y} V ${lane} H ${tx} V ${to.y}`;
  }
  if (from.op.col === to.op.col) {
    const cx = from.x + from.w / 2;
    const down = to.y > from.y;
    return `M ${cx} ${down ? from.y + from.h : from.y} V ${down ? to.y : to.y + to.h}`;
  }
  if (to.op.col > from.op.col) {
    const fx = from.x + from.w;
    const tx = to.x;
    const mid = tx - GAP_X / 2;
    return fy === ty ? `M ${fx} ${fy} H ${tx}` : `M ${fx} ${fy} H ${mid} V ${ty} H ${tx}`;
  }
  // Backward edge: drop below the lower box and return underneath.
  const below = Math.max(from.y + from.h, to.y + to.h) + 16;
  const fx = from.x + from.w / 2;
  const tx = to.x + to.w / 2;
  return `M ${fx} ${from.y + from.h} V ${below} H ${tx} V ${to.y + to.h}`;
}

function Inspector({ op }: { op: Op }) {
  const d = op.detail;
  const rows: (readonly [string, ReactNode])[] = [];
  for (const [name, value] of d.inputs ?? []) {
    rows.push([`in · ${name}`, value]);
  }
  if (d.output !== undefined) {
    rows.push([`out · ${d.output[0]}`, d.output[1]]);
  }
  for (const [name, value] of d.params ?? []) {
    rows.push([`param · ${name}`, value]);
  }
  return (
    <div aria-label="Operation inspector" className="ms-inspector" role="region">
      <div className="ms-inspector__title">{op.label}</div>
      <div>
        <p className="ms-inspector__what">{d.what}</p>
        {d.formula === undefined ? null : <div className="ms-inspector__formula">{d.formula}</div>}
        {d.note === undefined ? null : <p className="ms-inspector__what">{d.note}</p>}
      </div>
      <dl>
        {rows.map(([term, value]) => (
          <div key={term} style={{ display: "contents" }}>
            <dt>{term}</dt>
            <dd>{value}</dd>
          </div>
        ))}
        <dt>source</dt>
        <dd>
          {d.refs.map((ref) => (
            <div key={`${ref.source}:${ref.symbol}`}>
              <SourceRef {...ref} />
            </div>
          ))}
        </dd>
      </dl>
    </div>
  );
}

/**
 * `id` is the URL key that stores the selected op, so it must be unique on the page.
 * Ops are stepped in array order: list them in execution order.
 */
export function OpGraph({
  id,
  title,
  ops,
  edges,
  caption,
}: {
  id: string;
  title: string;
  ops: readonly Op[];
  edges: readonly Edge[];
  caption?: ReactNode;
}) {
  const titleId = useId();
  const arrow = useId();
  const ids = ops.map((op) => op.id);
  if (new Set(ids).size !== ids.length) {
    throw new Error(`OpGraph ${id}: duplicate op id`);
  }
  const [selectedId, select] = useUrlChoice(id, ids[0], ids);
  const index = ids.indexOf(selectedId);
  const selected = ops[index];
  const skips = edges.filter((edge) => edge.kind === "skip");
  const { boxes, width, height, top } = layout(ops, skips.length);
  const box = (opId: string): Box => {
    const found = boxes.get(opId);
    if (found === undefined) {
      throw new Error(`OpGraph ${id}: edge names unknown op "${opId}"`);
    }
    return found;
  };
  const onKey = (event: KeyboardEvent, opId: string): void => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      select(opId);
    } else if (event.key === "ArrowRight" && index < ops.length - 1) {
      event.preventDefault();
      select(ids[index + 1]);
    } else if (event.key === "ArrowLeft" && index > 0) {
      event.preventDefault();
      select(ids[index - 1]);
    }
  };

  return (
    <div className="ms-opgraph">
      <div className="ms-control">
        <Button disabled={index === 0} onClick={() => select(ids[index - 1])}>
          Previous op
        </Button>
        <Button disabled={index === ops.length - 1} onClick={() => select(ids[index + 1])}>
          Next op
        </Button>
        <span className="ms-control__value">
          op {String(index + 1)} of {String(ops.length)}
        </span>
      </div>
      <Sentence>
        {selected.label}: {selected.detail.what}
      </Sentence>
      <Figure caption={caption} label={title}>
        <svg
          aria-labelledby={titleId}
          className="ms-fig"
          height={height}
          role="group"
          viewBox={`0 0 ${String(width)} ${String(height)}`}
          width={width}
        >
          <title id={titleId}>{title}</title>
          <defs>
            <marker id={arrow} markerHeight="7" markerUnits="userSpaceOnUse" markerWidth="7" orient="auto" refX="6.2" refY="3.5" viewBox="0 0 7 7">
              <path className="ms-arrowhead" d="M 0 0.8 L 7 3.5 L 0 6.2 z" />
            </marker>
          </defs>
          {edges.map((edge) => {
            const from = box(edge.from);
            const to = box(edge.to);
            const d = route(edge, from, to, skips.indexOf(edge), top);
            const className =
              edge.kind === "skip" ? "ms-wire ms-wire--skip" : edge.kind === "state" ? "ms-wire ms-wire--state" : "ms-wire";
            return (
              <g key={`${edge.from}>${edge.to}:${edge.kind ?? "data"}`}>
                <path className={className} d={d} data-lane={from.op.lane ?? "merged"} markerEnd={`url(#${arrow})`} />
                {edge.label === undefined ? null : (
                  <text className="ms-wire__label" textAnchor="middle" x={(from.x + from.w + to.x) / 2} y={Math.min(from.y, to.y) + from.h / 2 - 7}>
                    {edge.label}
                  </text>
                )}
              </g>
            );
          })}
          {ops.map((op) => {
            const b = box(op.id);
            const round = op.kind === "sum" || op.kind === "mul";
            const cx = b.x + b.w / 2;
            const cy = b.y + b.h / 2;
            return (
              <g
                aria-label={`${op.label}${op.sub === undefined ? "" : `, ${op.sub}`}`}
                aria-pressed={op.id === selectedId}
                className="ms-node"
                data-kind={op.kind ?? "op"}
                data-lane={op.lane ?? "merged"}
                data-op={op.id}
                data-selected={op.id === selectedId ? "true" : undefined}
                key={op.id}
                onClick={() => select(op.id)}
                onKeyDown={(event) => onKey(event, op.id)}
                role="button"
                tabIndex={0}
              >
                {round ? (
                  <>
                    <circle cx={cx} cy={cy} r={ROUND / 2} />
                    <path
                      className="ms-wire"
                      d={
                        op.kind === "sum"
                          ? `M ${cx - 6} ${cy} h 12 M ${cx} ${cy - 6} v 12`
                          : `M ${cx - 4.5} ${cy - 4.5} l 9 9 M ${cx + 4.5} ${cy - 4.5} l -9 9`
                      }
                    />
                  </>
                ) : (
                  <>
                    <rect height={b.h} rx={op.kind === "tensor" || op.kind === "state" ? 3 : 7} width={b.w} x={b.x} y={b.y} />
                    <text className="ms-node__title" textAnchor="middle" x={cx} y={op.sub === undefined ? cy + 4 : cy - 3}>
                      {op.label}
                    </text>
                    {op.sub === undefined ? null : (
                      <text className="ms-node__line" textAnchor="middle" x={cx} y={cy + 13}>
                        {op.sub}
                      </text>
                    )}
                  </>
                )}
              </g>
            );
          })}
        </svg>
      </Figure>
      <Inspector op={selected} />
    </div>
  );
}
