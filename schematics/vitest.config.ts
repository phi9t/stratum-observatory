import { defineConfig } from "vitest/config";

export default defineConfig({
  test: { include: ["kit/**/*.test.ts", "tnsr/**/*.test.ts", "models/**/tests/*.test.ts"] },
});
