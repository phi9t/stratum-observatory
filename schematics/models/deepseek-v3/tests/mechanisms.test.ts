import { normals } from "../../../kit";
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import * as f from "../facts";

describe("checkpoint-specific rules", () => {
  it("does not put a dense layer in the MoE region or vice versa", () => {
    expect(f.kindOf(2)).toBe("dense");
    expect(f.kindOf(3)).toBe("moe");
    expect(f.kindOf(60)).toBe("moe");
    expect(() => f.kindOf(61)).toThrow(RangeError);
  });
  it("uses correction bias for selection but never for combination weights", () => {
    const logits = Array(256).fill(0);
    const bias = Array(256).fill(-10);
    [64, 65, 96, 97, 128, 129, 160, 161].forEach((i) => (bias[i] = 5));
    const r = f.route(logits, bias);
    expect([...r.chosen].sort((a, b) => a - b)).toEqual([64, 65, 96, 97, 128, 129, 160, 161]);
    expect(r.weights).toEqual(Array(8).fill(0.3125));
    expect(r.weights.reduce((a, b) => a + b, 0)).toBe(2.5);
  });
  it("agrees with kit routing for zero correction bias", () => {
    for (const seed of [0, 7, 81]) {
      const logits = normals(seed, 256);
      const actual = f.route(logits, Array(256).fill(0));
      const kit = f.syntheticRoute(seed, false);
      expect(actual.chosen).toEqual(kit.chosen);
      actual.weights.forEach((w, i) => expect(w).toBeCloseTo(kit.weights[i], 14));
    }
  });
  it("cannot select a masked group even when every corrected score is negative", () => {
    const r = f.route(
      Array(256).fill(0),
      Array.from({ length: 256 }, (_, i) => -1 - Math.floor(i / 32)),
    );
    expect(r.chosen.every((i) => i < 128)).toBe(true);
    expect(() => f.route([0], [0])).toThrow(RangeError);
  });
  it("interpolates low frequencies, preserves high ones, and modifies attention scale", () => {
    expect(f.yarnFrequency(0)).toBe(1);
    expect(f.yarnFrequency(31)).toBeCloseTo(10000 ** (-62 / 64) / 40, 12);
    expect(f.attentionScale.value).toBeCloseTo((1 + 0.1 * Math.log(40)) ** 2 / Math.sqrt(192), 12);
    expect(f.rotaryAmplitude.value).toBe(1);
  });
  it("keeps eager, flash padded, and compressed cache units separate", () => {
    expect(f.cacheScalars(1, 1, "compressed")).toBe(61 * 576);
    expect(f.cacheScalars(1, 1, "eager")).toBe(61 * 128 * (192 + 128));
    expect(f.cacheScalars(1, 1, "flash")).toBe(61 * 128 * 192 * 2);
    expect(() => f.cacheScalars(0, 1, "eager")).toThrow(RangeError);
  });
  it("reconciles index scale entries and duplicate MTP names without weights", () => {
    const index = JSON.parse(
      readFileSync(".sources-cache/deepseek-v3/weights-index/model.safetensors.index.json", "utf8"),
    );
    const names = Object.keys(index.weight_map);
    expect(names.filter((x) => x.endsWith("weight_scale_inv")).length).toBe(f.scaleTensorCount.value);
    expect(names).toContain("model.layers.61.embed_tokens.weight");
    expect(names).toContain("model.layers.61.shared_head.head.weight");
    expect(f.storedTotal.value).toBe(684531386000);
    expect(f.parameterTotal.value).toBe(671026419200);
    expect(f.scaleScalars.value).toBe(41540496);
  });
});
