import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { Ctx } from "../../../kit/ref/tensor";
import { forward } from "../reference/model";
import * as f from "../facts";
import metadata from "../storage-evidence.json";
import summary from "../storage-summary.json";
describe("checkpoint accounting and scenario bounds", () => {
  it("rejects impossible cache scenarios rather than presenting negative storage", () => {
    expect(() => f.cacheScalars(-1, 1)).toThrow();
    expect(() => f.cacheScalars(1, 0)).toThrow();
    expect(() => f.cacheScalars(1, f.context.value + 1)).toThrow();
    expect(() => f.cacheScalars(1.5, 1)).toThrow();
  });
  it("reconciles every stored tensor name and scalar with pinned metadata", () => {
    const index = JSON.parse(readFileSync(".sources-cache/kimi-k2/tensor-index/model.safetensors.index.json", "utf8"));
    const info = JSON.parse(readFileSync(".sources-cache/kimi-k2/model-info.json", "utf8"));
    const ctx = new Ctx("shape");
    forward(ctx, f.real, [[0]]);
    const actual = Object.keys(index.weight_map).sort();
    const names = [...ctx.params.keys()];
    let scales = 0;
    for (const [name, dims] of ctx.params) {
      if (
        name.endsWith(".weight") &&
        dims.length === 2 &&
        name !== "lm_head.weight" &&
        name !== "model.embed_tokens.weight" &&
        !name.endsWith(".gate.weight")
      ) {
        names.push(name.replace(/\.weight$/, ".weight_scale_inv"));
        scales += Math.ceil(dims[0] / f.blocks.value[0]) * Math.ceil(dims[1] / f.blocks.value[1]);
      }
    }
    for (let i = 0; i < f.layers.value; i++) names.push(`model.layers.${i}.self_attn.rotary_emb.inv_freq`);
    expect(names.sort()).toEqual(actual);
    for (const shard of metadata.shards) {
      for (const [name, record] of Object.entries(shard.tensors)) {
        if (name.endsWith(".weight")) expect(ctx.params.get(name), name).toEqual(record.shape);
        if (name.endsWith("inv_freq")) expect(record.shape).toEqual([f.savedFrequencyWidth.value]);
        if (name.endsWith("weight_scale_inv")) {
          const dims = ctx.params.get(name.replace("weight_scale_inv", "weight"))!;
          expect(record.shape).toEqual([
            Math.ceil(dims[0] / f.blocks.value[0]),
            Math.ceil(dims[1] / f.blocks.value[1]),
          ]);
        }
      }
    }
    expect(summary.header_sha256).toEqual(metadata.shards.map((s) => s.header_sha256));
    expect(scales).toBe(f.scaleScalars.value);
    expect(f.storedTotal.value).toBe(info.safetensors.total);
  });
});
