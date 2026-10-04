import { type Op, type Edge, type OpDetail, type Lane, type Ref, shape, n } from "../../kit";
import * as f from "./facts";

export const stream = shape("B", "S", f.hidden.value);
export const attentionRefs: readonly Ref[] = [
  { source: "reference", symbol: "DeepseekV3Attention.forward" },
  { source: "serving", symbol: "DeepseekV2Attention.forward" },
  { source: "mla-backend", symbol: "MultiHeadLatentAttentionWrapper.forward" },
  { source: "publisher", symbol: "MLA.forward" },
  { source: "training", symbol: "Attention.forward" },
];
export const ffnRefs: readonly Ref[] = [
  { source: "reference", symbol: "DeepseekV3MLP.forward" },
  { source: "serving", symbol: "DeepseekV2MLP.forward" },
  { source: "publisher", symbol: "MLP.forward" },
];
export const routerRefs: readonly Ref[] = [
  { source: "reference", symbol: "MoEGate.forward" },
  { source: "serving", symbol: "DeepseekV2MoE" },
  { source: "publisher", symbol: "Gate.forward" },
  { source: "training-moe", symbol: "DeepSeekV3Router._select_experts" },
];
export const blockRefs: readonly Ref[] = [
  { source: "reference", symbol: "DeepseekV3DecoderLayer.forward" },
  { source: "serving", symbol: "DeepseekV2DecoderLayer.forward" },
  { source: "publisher", symbol: "Block.forward" },
  { source: "training", symbol: "DeepSeekV3TransformerBlock.forward" },
];
export function operation(
  id: string,
  label: string,
  col: number,
  width: number,
  detail: OpDetail,
  lane: Lane = "high",
  row = 0,
  kind: Op["kind"] = "op",
  sub = n(width),
): Op {
  return { id, label, col, row, width, lane, kind, sub, detail: { params: [], ...detail } };
}
export const chain = (ops: readonly Op[]): Edge[] => ops.slice(1).map((op, i) => ({ from: ops[i].id, to: op.id }));
export const normFormula = `y = x · rsqrt(mean(x²) + ${f.epsilon.value}) · gain`;
export function linearDetail(what: string, inp: number, out: number, refs: readonly Ref[], note?: string): OpDetail {
  return {
    what,
    formula: "y = x Wᵀ",
    inputs: [["x", shape("B", "S", inp)]],
    output: ["y", shape("B", "S", out)],
    params: [["weight", shape(out, inp)]],
    refs,
    note,
  };
}
