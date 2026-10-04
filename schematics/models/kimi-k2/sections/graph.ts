import { shape, type Op, type Edge, type Lane } from "../../../kit";
import * as f from "../facts";
/** Consistent inspectors: dimensions and parameter shapes come from facts. */
export function op(
  id: string,
  label: string,
  width: number,
  grid: { col: number; row: number },
  symbol: string,
  what: string,
  formula: string,
  input: string,
  output = shape("B", "S", width),
  params: readonly (readonly [string, string])[] = [],
  lane: Lane = "high",
  note?: string,
): Op {
  const serving =
    symbol === "MoEGate" || symbol === "DeepseekV3MoE"
      ? "DeepseekV2MoE"
      : symbol === "DeepseekV3MLP"
        ? "DeepseekV2MLP"
        : symbol === "DeepseekV3DecoderLayer"
          ? "DeepseekV2DecoderLayer"
          : "DeepseekV2MLAAttention";
  return {
    id,
    label,
    width,
    col: grid.col,
    row: grid.row,
    lane: formula === "x" ? "merged" : lane,
    kind:
      label === "add"
        ? "sum"
        : label === "multiply"
          ? "mul"
          : id === "cache"
            ? "state"
            : formula === "x"
              ? "tensor"
              : "op",
    sub: String(width),
    detail: {
      what,
      formula,
      inputs: [["input", input]],
      output: [id, output],
      params,
      refs: [
        { source: "reference", symbol },
        { source: "serving", symbol: serving },
      ],
      note,
    },
  };
}
export function chain(ops: readonly Op[]): Edge[] {
  return ops.slice(1).map((o, i) => ({ from: ops[i].id, to: o.id }));
}
export const stream = shape("B", "S", f.hidden.value);
export const normFormula = `x × rsqrt(mean(x²) + ${f.eps.value}) × gain`;
