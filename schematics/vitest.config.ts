import { defineConfig } from "vitest/config";

export default defineConfig({
  test: { include: ["kit/**/*.test.ts", "models/**/tests/*.test.ts"] },
});
