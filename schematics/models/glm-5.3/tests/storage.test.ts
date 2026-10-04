import { execFileSync } from "node:child_process";
import { expect, it } from "vitest";
import audit from "../reference/storage-audit.json";
import { parameterTotal, mtpParams, correctionBuffers, hubAccounted } from "../facts";
it("reconciles all pinned tensor names, Hub scalar count and storage byte total", () => {
  expect(() =>
    execFileSync("python3", ["models/glm-5.3/reference/storage-audit.py", "--check"], { stdio: "pipe" }),
  ).not.toThrow();
  expect(audit.totals.decoder_trainable).toBe(parameterTotal.value);
  expect(audit.totals.mtp_trainable).toBe(mtpParams.value);
  expect(audit.totals.correction_buffers).toBe(correctionBuffers.value);
  expect(audit.hub_total).toBe(hubAccounted.value);
  expect(audit.storage_bytes).toBe(audit.index_storage_bytes);
});
