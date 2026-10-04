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
      <marker id={id} markerHeight="7" markerUnits="userSpaceOnUse" markerWidth="7" orient="auto" refX="6.2" refY="3.5" viewBox="0 0 7 7">
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
        <text className="ms-node__line" key={line} textAnchor="middle" x={x + width / 2} y={first + (index + (title === undefined ? 0 : 1)) * pitch}>
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
  return <path className={className} d={d} data-lane={lane} markerEnd={arrow === undefined ? undefined : `url(#${arrow})`} />;
}

/** Residual sum (plus) or gate (times) junction. */
export function Junction({ x, y, kind }: { x: number; y: number; kind: "sum" | "mul" }) {
  return (
    <g className="ms-sum">
      <circle cx={x} cy={y} r={9} />
      <path d={kind === "sum" ? `M ${x - 5} ${y} h 10 M ${x} ${y - 5} v 10` : `M ${x - 4} ${y - 4} l 8 8 M ${x + 4} ${y - 4} l -8 8`} />
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

export function Note({ x, y, lane, anchor, children }: { x: number; y: number; lane?: Lane; anchor?: "start" | "middle" | "end"; children: string }) {
  return (
    <text className="ms-fig__note" data-lane={lane} textAnchor={anchor} x={x} y={y}>
      {children}
    </text>
  );
}
