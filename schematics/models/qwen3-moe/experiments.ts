import { Ctx, input } from "../../tnsr/tensor";
import { normals, type Routing } from "../../kit";
import { Qwen3MoeRouter, forward } from "./reference/model";
import * as f from "./facts";
export function routerExperiment(seed: number): Routing {
  const ctx = new Ctx("value", null, seed, f.GAIN);
  const x = input(ctx, [1, f.TOY.hidden], () => normals(seed, f.TOY.hidden));
  const result = Qwen3MoeRouter(ctx, "demo.gate", "demo", x, {
    ...f.TOY,
    experts: f.experts.value,
    topK: f.topK.value,
  });
  return {
    scores: Array.from(result.scores.data!),
    chosen: Array.from(result.selected.indices.data!),
    weights: Array.from(result.weights.data!),
    chosenGroups: [0],
  };
}
export function toyRun(seed: number) {
  const ctx = new Ctx("value", null, seed, f.GAIN);
  const ids = [Array.from({ length: f.TOY.tokens }, (_, i) => (i * f.TOY.topK + seed) % f.TOY.vocab)];
  const logits = forward(ctx, f.TOY, ids);
  return { ctx, ids, logits };
}
