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

    const manifest = JSON.parse(readFileSync("dist/manifest.json", "utf8")) as {
      action?: { default_popup?: string };
      options_page?: string;
      web_accessible_resources?: Array<{ resources?: string[] }>;
    };
    const manifestSource = readFileSync("manifest.config.ts", "utf8");
    const contentFile = "dist/content.js";
    const resourceFiles =
      manifest.web_accessible_resources?.flatMap((entry) => entry.resources ?? []) ?? [];

    expect(manifest.action?.default_popup).toBe("popup.html");
    expect(manifest.options_page).toBe("options.html");
    expect(resourceFiles).toContain("styles/overlay.css");
    expect(resourceFiles).toContain("vendor/onnxruntime/ort.webgpu.min.js");
    expect(resourceFiles).not.toContain("content.js");
    expect(existsSync(contentFile)).toBe(true);
    expect(readFileSync(contentFile, "utf8")).not.toMatch(/^\s*import\s/m);
    expect(manifestSource).toContain("CONTENT_STYLE_FILE");
    expect(manifestSource).toContain("ORT_RUNTIME_FILES");
  });
});
