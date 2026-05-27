import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, rmSync } from "node:fs";

import { describe, expect, it } from "vitest";

describe("extension build output", () => {
  it("emits popup, options, and a classic injected content bundle", () => {
    rmSync("dist", { recursive: true, force: true });

    execFileSync("pnpm", ["run", "build"], {
      cwd: process.cwd(),
      stdio: "inherit",
      shell: true,
    });

    const manifest = readFileSync("dist/manifest.json", "utf8");
    const contentFile = "dist/content.js";

    expect(manifest).toContain("\"default_popup\": \"popup.html\"");
    expect(manifest).toContain("\"options_page\": \"options.html\"");
    expect(existsSync(contentFile)).toBe(true);
    expect(readFileSync(contentFile, "utf8")).not.toMatch(/^\s*import\s/m);
  });
});
