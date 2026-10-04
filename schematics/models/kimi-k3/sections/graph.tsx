import { OpGraph, shape, n, type Op, type Edge, type Lane, type Ref } from "../../../kit";
import * as f from "../facts";
export type Step = {
  id: string;
  col?: number;
  row?: number;
  inputs?: readonly (readonly [string, string])[];
  label: string;
  width: number;
  what: string;
  formula: string;
  input?: string;
  output?: string;
  params?: readonly (readonly [string, string])[];
  note?: string;
  kind?: Op["kind"];
  lane?: Lane;
  refs?: readonly Ref[];
};
export function Graph({
  id,
  title,
  heading,
  steps,
  source,
  serving,
  edges,
}: {
  id: string;
  title: string;
  heading?: string;
  steps: readonly Step[];
  source: string;
  serving: string;
  edges?: readonly Edge[];
}) {
  const ops: Op[] = steps.map((s, i) => ({
    id: s.id,
    label: s.label,
    sub: n(s.width),
    width: s.width,
    col: s.col ?? i,
    row: s.row ?? 0,
    lane: s.lane ?? "high",
    kind: s.kind,
    detail: {
      what: s.what,
      formula: s.formula,
      inputs: s.inputs ?? [
        [
          i ? steps[i - 1].id : "x",
          s.input ?? (i ? steps[i - 1].output : undefined) ?? shape("B", "S", i ? steps[i - 1].width : f.hidden.value),
        ],
      ],
      output: [s.id, s.output ?? shape("B", "S", s.width)],
      params: s.params ?? [],
      refs: s.refs ?? [
        { source: "reference", symbol: source },
        { source: "serving", symbol: serving },
      ],
      note: s.note,
    },
  }));
  return (
    <OpGraph
      id={id}
      title={title}
      heading={heading}
      ops={ops}
      edges={edges ?? steps.slice(1).map((s, i) => ({ from: steps[i].id, to: s.id }))}
    />
  );
}
