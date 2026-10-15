import { describe, expect, it } from "vitest";

import * as f from "../facts";

describe("GLM-5.3-Flash mechanisms", () => {
  it("uses the checked three-to-one KDA versus sparse attention schedule", () => {
    const sparseByPattern = Array.from({ length: f.layers.value }, (_unused, index) => index).filter(
      (index) => (index + 1) % 4 === 0,
    );
    expect(f.sparseLayers.value).toEqual(sparseByPattern);
    expect(f.kdaCount.value).toBe(34);
    expect(f.sparseAttentionCount.value).toBe(11);
  });

  it("does not use cross-layer top-k reuse in the base decoder", () => {
    expect(f.indexerTypes.value).toHaveLength(f.layers.value);
    expect(f.allSparseFullIndexer.value).toBe(true);
    expect(f.indexerTypes.value.every((kind) => kind === "full")).toBe(true);
  });

  it("keeps dense MLP only in the first three layers and MoE afterward", () => {
    expect(f.denseLayerCount.value).toBe(f.denseFirst.value);
    expect(f.moeLayerCount.value).toBe(f.layers.value - f.denseFirst.value);
    expect(f.mlpLayerTypes.value.slice(0, f.denseFirst.value).every((kind) => kind === "dense")).toBe(true);
    expect(f.mlpLayerTypes.value.slice(f.denseFirst.value).every((kind) => kind === "sparse")).toBe(true);
  });

  it("reconciles text, MTP and vision stored tensors to the Hub total", () => {
    expect(f.routedExpertParameters.value).toBe(f.memoryEvidence.routedExpertParams);
    expect(f.textBackboneStoredParameters.value + f.mtpParameters.value + f.visionParameters.value).toBe(
      f.parameterTotal.value,
    );
    expect(f.parameterTotal.value).toBe(321323031390);
  });

  it("labels the mHC saved activation term as whole-model rather than per-layer", () => {
    const perLayer = f.hcMult.value * f.hidden.value * f.bytesPerBf16.value;
    expect(f.mHcSavedPerLayerBytesPerToken.value).toBe(perLayer);
    expect(f.mHcSavedBytesPerToken.value).toBe(f.layers.value * perLayer);
    expect((f.mHcSavedBytesPerToken.value * 16384) / 1024 ** 3).toBe(22.5);
    expect(Math.round(f.activationBudgetEp8GiB.value)).toBe(99);
  });

  it("keeps the Countdown ladder rows on completed evidence", () => {
    const rung2 = f.sftLadder.find((row) => row.rung === "2");
    const rung4 = f.sftLadder.find((row) => row.rung === "4");

    expect(rung2?.status).toContain("0.006 -> attention 0.440/0.476");
    expect(rung2?.status).toContain("attention+expert 0.600");
    expect(rung2?.status).toContain("no router collapse");
    expect(rung2?.status).not.toContain("planned");

    expect(rung4?.status).toContain("94-step seeds");
    expect(rung4?.status).toContain("steps 63-94");
    expect(rung4?.status).toContain("max |delta| 0.0");
    expect(f.rung4Evidence.resumeMaxAbsDelta).toBe(0);
    expect(f.rung4Evidence.evalStatus).toContain("pending");
  });
});
