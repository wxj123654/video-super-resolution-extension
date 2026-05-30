import { defineManifest } from "@crxjs/vite-plugin";
import {
  CONTENT_STYLE_FILE,
  ORT_RUNTIME_FILES,
} from "./src/shared/extension/injection";

export default defineManifest({
  manifest_version: 3,
  name: "Video GPU Super Resolution",
  description:
    "Realtime WebGL/WebGPU super-resolution for HTML5 videos on the current page.",
  version: "0.3.0",
  minimum_chrome_version: "113",
  action: {
    default_title: "Video GPU Super Resolution",
    default_popup: "popup.html",
  },
  options_page: "options.html",
  permissions: ["activeTab", "scripting", "storage"],
  host_permissions: ["https://huggingface.co/*", "https://cdn-lfs.huggingface.co/*"],
  web_accessible_resources: [
    {
      resources: [
        "models/*",
        ...ORT_RUNTIME_FILES,
        CONTENT_STYLE_FILE,
      ],
      matches: ["<all_urls>"],
    },
  ],
});
