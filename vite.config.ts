import { defineConfig } from "vite";
import { crx } from "@crxjs/vite-plugin";
import react from "@vitejs/plugin-react";
import manifest from "./manifest.config";
import { resolve } from "path";
import { rollup, type Plugin } from "rollup";
import esbuild from "rollup-plugin-esbuild";
import { readFileSync } from "fs";
import { dirname, resolve as pathResolve } from "path";

function rawImportPlugin(): Plugin {
  return {
    name: "raw-import",
    resolveId(source, importer) {
      if (source.endsWith("?raw")) {
        const clean = source.replace(/\?raw$/, "");
        if (importer) {
          const dir = dirname(importer);
          const abs = pathResolve(dir, clean);
          return "\0raw:" + abs;
        }
        return "\0raw:" + clean;
      }
      return null;
    },
    load(id) {
      if (id.startsWith("\0raw:")) {
        const realPath = id.slice(5);
        const content = readFileSync(realPath, "utf-8");
        return `export default ${JSON.stringify(content)};`;
      }
      return null;
    },
  };
}

function iifeContentScript(isDebug: boolean): import("vite").Plugin {
  return {
    name: "iife-content-script",
    async closeBundle() {
      const input = resolve(__dirname, "src/content/index.ts");
      const bundle = await rollup({
        input,
        plugins: [rawImportPlugin(), esbuild({ target: "chrome113", minify: !isDebug })],
      });
      await bundle.write({
        file: resolve(__dirname, "dist/content.js"),
        format: "iife",
        inlineDynamicImports: true,
      });
      await bundle.close();
      console.log("\x1b[32m✓\x1b[0m content.js rebuilt as IIFE");
    },
  };
}

export default defineConfig(({ mode }) => {
  const isDebugBuild = mode === "debug";

  return {
    build: {
      minify: isDebugBuild ? false : undefined,
      sourcemap: isDebugBuild,
      target: "chrome113",
    },
    resolve: {
      alias: {
        "@src": "/src",
      },
    },
    define: {
      __VSR_DEBUG__: JSON.stringify(isDebugBuild),
    },
    plugins: [react(), crx({ manifest }), iifeContentScript(isDebugBuild)],
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: "./tests/setup.ts",
    },
  };
});
