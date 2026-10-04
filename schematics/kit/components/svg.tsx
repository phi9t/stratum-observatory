import type { ReactNode } from "react";

import type { Lane } from "./shell";

// Primitives for hand-laid figures (the whole-model overview, usually). For
// anything that is a graph of operations, prefer <OpGraph>, which lays itself out.

const TITLE_PX = 7.7;
const LINE_PX = 6.75;

/** Width a Node needs so its text fits; use it instead of guessing box widths. */
export function nodeWidth(title: string | undefined, lines: readonly string[] = []): number {
  const text = Math.max((title?.length ?? 0) * TITLE_PX, ...lines.map((line) => line.length * LINE_PX));
  return Math.max(64, Math.ceil(text + 26));
}

export function ArrowDefs({ id }: { id: string }) {
  return (
    <defs>
      <marker
        id={id}
        markerHeight="7"
        markerUnits="userSpaceOnUse"
        markerWidth="7"
        orient="auto"
        refX="6.2"
        refY="3.5"
        viewBox="0 0 7 7"
      >
        <path className="ms-arrowhead" d="M 0 0.8 L 7 3.5 L 0 6.2 z" />
      </marker>
    </defs>
  );
}

/** Box with a title and mono lines, vertically centred. `w` defaults to the width the text needs. */
export function Node({
  x,
  y,
  h,
  w,
  lane,
  title,
  lines = [],
  kind = "op",
  children,
}: {
  x: number;
  y: number;
  h: number;
  w?: number;
  lane: Lane;
  title?: string;
  lines?: readonly string[];
  kind?: "op" | "tensor" | "state";
  children?: ReactNode;
}) {
  const width = w ?? nodeWidth(title, lines);
  const rows = lines.length + (title === undefined ? 0 : 1);
  const pitch = 15;
  const first = y + h / 2 - ((rows - 1) * pitch) / 2 + 4;
  return (
    <g className="ms-node" data-kind={kind} data-lane={lane}>
      <rect height={h} rx={kind === "op" ? 7 : 3} width={width} x={x} y={y} />
      {title === undefined ? null : (
        <text className="ms-node__title" textAnchor="middle" x={x + width / 2} y={first}>
          {title}
        </text>
      )}
      {lines.map((line, index) => (
        <text
          className="ms-node__line"
          key={line}
          textAnchor="middle"
          x={x + width / 2}
          y={first + (index + (title === undefined ? 0 : 1)) * pitch}
        >
          {line}
        </text>
      ))}
      {children}
    </g>
  );
}

/** Wire along an SVG path. Keep segments horizontal or vertical. */
export function Wire({
  d,
  lane = "merged",
  kind = "data",
  arrow,
}: {
  d: string;
  lane?: Lane;
  kind?: "data" | "skip" | "state";
  /** Marker id from <ArrowDefs>; omit for a plain line. */
  arrow?: string;
}) {
  const className = kind === "skip" ? "ms-wire ms-wire--skip" : kind === "state" ? "ms-wire ms-wire--state" : "ms-wire";
  return (
    <path className={className} d={d} data-lane={lane} markerEnd={arrow === undefined ? undefined : `url(#${arrow})`} />
  );
}

/** Residual sum (plus) or gate (times) junction. */
export function Junction({ x, y, kind }: { x: number; y: number; kind: "sum" | "mul" }) {
  return (
    <g className="ms-sum">
      <circle cx={x} cy={y} r={9} />
      <path
        d={
          kind === "sum"
            ? `M ${x - 5} ${y} h 10 M ${x} ${y - 5} v 10`
            : `M ${x - 4} ${y - 4} l 8 8 M ${x + 4} ${y - 4} l -8 8`
        }
      />
    </g>
  );
}

export function ColumnLabel({ x, y = 20, children }: { x: number; y?: number; children: string }) {
  return (
    <text className="ms-fig__column" x={x} y={y}>
      {children}
    </text>
  );
}

export function Note({
  x,
  y,
  lane,
  anchor,
  children,
}: {
  x: number;
  y: number;
  lane?: Lane;
  anchor?: "start" | "middle" | "end";
  children: string;
}) {
  return (
    <text className="ms-fig__note" data-lane={lane} textAnchor={anchor} x={x} y={y}>
      {children}
    </text>
  );
}

// ---------------------------------------------------------------------------
// StackOverview: the whole model on one line, with the layer stack drawn as
// tiles coloured by kind. Use it as the page's first figure.

export type OverviewStage = Readonly<{ title: string; line: string; lane: Lane; kind?: "op" | "tensor" }>;

const OVERVIEW = { tile: 12, gap: 3, perRow: 32, nodeH: 46, pad: 18, link: 34, top: 44 };

export function StackOverview({
  title,
  before,
  after,
  count,
  kinds,
  kindOf,
  stackTitle,
  stackLine,
  side,
  selected,
}: {
  title: string;
  /** Stages left of the stack: input tensor, embedding. */
  before: readonly OverviewStage[];
  /** Stages right of the stack: final norm, head, output tensor. */
  after: readonly OverviewStage[];
  count: number;
  kinds: readonly Readonly<{ id: string; label: string; lane: Lane }>[];
  kindOf: (index: number) => string;
  stackTitle: string;
  /** One mono line under the tiles, usually the stream shape. */
  stackLine: string;
  /** Inputs that enter every layer from the side (positions, masks, caches). */
  side?: OverviewStage;
  /** Layer to mark, so the overview follows the layer map. */
  selected?: number;
}) {
  const O = OVERVIEW;
  const rows = Math.ceil(count / O.perRow);
  const perRow = Math.min(count, O.perRow);
  const tilesW = perRow * (O.tile + O.gap) - O.gap;
  const plateW = Math.max(tilesW + 2 * O.pad, nodeWidth(stackTitle, [stackLine]) + 8);
  const plateH = 38 + rows * (O.tile + O.gap) + 26;
  const mid = O.top + plateH / 2;
  const widths = (stages: readonly OverviewStage[]): number[] =>
    stages.map((stage) => nodeWidth(stage.title, [stage.line]));
  const beforeW = widths(before);
  const afterW = widths(after);
  let x = 16;
  const beforeX = beforeW.map((w) => {
    const at = x;
    x += w + O.link;
    return at;
  });
  const plateX = x;
  x += plateW + O.link;
  const afterX = afterW.map((w) => {
    const at = x;
    x += w + O.link;
    return at;
  });
  const width = x - O.link + 16;
  const sideW = side === undefined ? 0 : nodeWidth(side.title, [side.line]);
  const sideY = O.top + plateH + 30;
  const legendY = sideY + (side === undefined ? 0 : O.nodeH + 26);
  const height = legendY + 18;
  const laneOf = (index: number): Lane => kinds.find((kind) => kind.id === kindOf(index))?.lane ?? "merged";
  const tilesX = plateX + (plateW - tilesW) / 2;
  return (
    <svg
      aria-label={title}
      className="ms-fig"
      height={height}
      role="img"
      viewBox={`0 0 ${width} ${height}`}
      width={width}
    >
      <ArrowDefs id="ms-overview-arrow" />
      {before.map((stage, index) => (
        <g key={stage.title}>
          <Node
            h={O.nodeH}
            kind={stage.kind ?? "op"}
            lane={stage.lane}
            lines={[stage.line]}
            title={stage.title}
            x={beforeX[index]}
            y={mid - O.nodeH / 2}
          />
          <Wire
            arrow="ms-overview-arrow"
            d={`M ${beforeX[index] + beforeW[index]} ${mid} H ${index === before.length - 1 ? plateX : beforeX[index + 1]}`}
            lane={stage.lane}
          />
        </g>
      ))}
      <rect className="ms-node__plate" height={plateH} rx={8} width={plateW} x={plateX} y={O.top} />
      <text className="ms-node__title" textAnchor="middle" x={plateX + plateW / 2} y={O.top + 22}>
        {stackTitle}
      </text>
      {Array.from({ length: count }, (_u, index) => (
        <rect
          className="ms-stack__tile"
          data-lane={laneOf(index)}
          data-selected={index === selected ? "true" : undefined}
          height={O.tile}
          key={index}
          rx={2}
          width={O.tile}
          x={tilesX + (index % O.perRow) * (O.tile + O.gap)}
          y={O.top + 34 + Math.floor(index / O.perRow) * (O.tile + O.gap)}
        />
      ))}
      <text className="ms-node__line" textAnchor="middle" x={plateX + plateW / 2} y={O.top + plateH - 10}>
        {stackLine}
      </text>
      {after.map((stage, index) => (
        <g key={stage.title}>
          <Wire
            arrow="ms-overview-arrow"
            d={`M ${index === 0 ? plateX + plateW : afterX[index - 1] + afterW[index - 1]} ${mid} H ${afterX[index]}`}
          />
          <Node
            h={O.nodeH}
            kind={stage.kind ?? "op"}
            lane={stage.lane}
            lines={[stage.line]}
            title={stage.title}
            x={afterX[index]}
            y={mid - O.nodeH / 2}
          />
        </g>
      ))}
      {side === undefined ? null : (
        <>
          <Node
            h={O.nodeH}
            kind="state"
            lane={side.lane}
            lines={[side.line]}
            title={side.title}
            x={plateX + (plateW - sideW) / 2}
            y={sideY}
          />
          <Wire
            arrow="ms-overview-arrow"
            d={`M ${plateX + plateW / 2} ${sideY} V ${O.top + plateH}`}
            kind="state"
            lane={side.lane}
          />
        </>
      )}
      {kinds.map((kind, index) => (
        <g key={kind.id}>
          <rect
            className="ms-stack__tile"
            data-lane={kind.lane}
            height={O.tile}
            rx={2}
            width={O.tile}
            x={16 + index * 230}
            y={legendY - 10}
          />
          <text className="ms-fig__note" x={16 + index * 230 + 20} y={legendY}>
            {kind.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
