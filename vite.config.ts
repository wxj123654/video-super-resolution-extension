import { defineConfig } from "vite";
import { crx } from "@crxjs/vite-plugin";
import manifest from "./manifest.config";
import { resolve } from "path";

export default defineConfig(({ mode }) => {
  const isDebugBuild = mode === "debug";

  return {
    build: {
      minify: isDebugBuild ? false : undefined,
      sourcemap: isDebugBuild,
      target: "chrome113",
      rollupOptions: {
        input: {
          content: resolve(__dirname, "src/content/index.ts"),
        },
        output: {
          entryFileNames: (chunkInfo) => {
            if (chunkInfo.name === "content") return "content.js";
            return "assets/[name]-[hash].js";
          },
        },
      },
    },
    resolve: {
      alias: {
        "@src": "/src",
      },
    },
    define: {
      __VSR_DEBUG__: JSON.stringify(isDebugBuild),
    },
    plugins: [crx({ manifest })],
  };
});
