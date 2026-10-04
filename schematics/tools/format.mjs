// usage: npm run format -- <slug>   (formats models/<slug> with the repo's prettier settings)
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

const slug = process.argv[2];
if (!slug || !existsSync(`models/${slug}`)) {
  console.error("usage: npm run format -- <slug>");
  process.exit(1);
}
const result = spawnSync("npx", ["prettier", "--write", `models/${slug}/**/*.{ts,tsx}`], { stdio: "inherit" });
process.exit(result.status ?? 1);
