import { defineConfig, type Plugin } from "vite";
import { crx } from "@crxjs/vite-plugin";
import react from "@vitejs/plugin-react";
import manifest from "./manifest.config";
import { resolve } from "path";
import esbuild from "esbuild";
import { readFileSync } from "fs";

function rawPlugin(): esbuild.Plugin {
  return {
    name: "raw-import",
    setup(build) {
      build.onResolve({ filter: /\?raw$/ }, (args) => ({
        path: resolve(args.resolveDir, args.path.replace(/\?raw$/, "")),
        namespace: "raw",
      }));
      build.onLoad({ filter: /.*/, namespace: "raw" }, async (args) => ({
        contents: `export default ${JSON.stringify(readFileSync(args.path, "utf-8"))}`,
        loader: "js",
      }));
    },
  };
}

function iifeContentScript(isDebug: boolean, define: Record<string, string>): Plugin {
  return {
    name: "iife-content-script",
    async closeBundle() {
      await esbuild.build({
        entryPoints: [resolve(__dirname, "src/content/index.ts")],
        bundle: true,
        format: "iife",
        outfile: resolve(__dirname, "dist/content.js"),
        target: "chrome113",
        minify: !isDebug,
        sourcemap: isDebug,
        define,
        alias: { "@src": resolve(__dirname, "src") },
        plugins: [rawPlugin()],
        logLevel: "info",
      });
      console.log("\x1b[32m✓\x1b[0m content.js rebuilt as IIFE");
    },
  };
}

export default defineConfig(({ mode }) => {
  const isDebugBuild = mode === "debug";
  const define = { __VSR_DEBUG__: JSON.stringify(isDebugBuild) };

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
    define,
    plugins: [react(), crx({ manifest }), iifeContentScript(isDebugBuild, define)],
  };
});
