const DEFAULT_SETTINGS = {
  enabled: false,
  scale: 1.5,
  sharpness: 0.65,
  mode: "balanced",
  overlayOpacity: 0.8,
  displayMode: "overlay",
  engine: "tiny-cnn"
};

const VALID_ENGINES = new Set(["webgpu", "tiny-cnn", "ecbsr"]);

const els = {
  enabled: document.querySelector("#enabled"),
  scale: document.querySelector("#scale"),
  sharpness: document.querySelector("#sharpness"),
  sharpnessValue: document.querySelector("#sharpnessValue"),
  overlayOpacity: document.querySelector("#overlayOpacity"),
  overlayOpacityValue: document.querySelector("#overlayOpacityValue"),
  displayMode: document.querySelector("#displayMode"),
  engine: document.querySelector("#engine"),
  mode: document.querySelector("#mode"),
  status: document.querySelector("#status"),
  rescan: document.querySelector("#rescan"),
  runDiagnostics: document.querySelector("#runDiagnostics"),
  diagnosticsOutput: document.querySelector("#diagnosticsOutput")
};

let activeTabId = null;
let settings = { ...DEFAULT_SETTINGS };

init();

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  activeTabId = tab?.id ?? null;

  const stored = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  settings = normalizeSettings({ ...DEFAULT_SETTINGS, ...stored });
  renderSettings();

  if (!activeTabId) {
    setStatus("没有可用标签页");
    return;
  }

  try {
    await ensureContentScript();
    const state = await send({ type: "VSR_UPDATE", settings });
    applyState(state, settings.enabled ? "已连接，增强已启用" : "已连接，当前关闭");
  } catch (error) {
    setStatus("当前页面不支持注入");
  }
}

for (const key of ["enabled", "scale", "sharpness", "overlayOpacity", "displayMode", "engine", "mode"]) {
  els[key].addEventListener("input", () => {
    settings = readSettings();
    renderSettings();
    chrome.storage.sync.set(settings);
    send({ type: "VSR_UPDATE", settings })
      .then((state) => applyState(state, "已更新"))
      .catch(() => setStatus("无法连接当前页面"));
  });
}

els.rescan.addEventListener("click", () => {
  send({ type: "VSR_RESCAN" })
    .then((state) => applyState(state, "已重新扫描"))
    .catch(() => setStatus("无法重新扫描"));
});

els.runDiagnostics.addEventListener("click", () => {
  runDiagnostics().catch((error) => {
    els.diagnosticsOutput.textContent = `检测失败\n${error instanceof Error ? error.message : String(error)}`;
  });
});

function readSettings() {
  return normalizeSettings({
    enabled: els.enabled.checked,
    scale: Number(els.scale.value),
    sharpness: Number(els.sharpness.value),
    overlayOpacity: Number(els.overlayOpacity.value),
    displayMode: els.displayMode.value,
    engine: els.engine.value,
    mode: els.mode.value
  });
}

function normalizeSettings(value) {
  return {
    ...DEFAULT_SETTINGS,
    ...value,
    engine: VALID_ENGINES.has(value.engine) ? value.engine : DEFAULT_SETTINGS.engine
  };
}

function renderSettings() {
  els.enabled.checked = settings.enabled;
  els.scale.value = String(settings.scale);
  els.sharpness.value = String(settings.sharpness);
  els.sharpnessValue.value = settings.sharpness.toFixed(2);
  els.overlayOpacity.value = String(settings.overlayOpacity);
  els.overlayOpacityValue.value = settings.overlayOpacity.toFixed(2);
  els.displayMode.value = settings.displayMode;
  els.engine.value = settings.engine;
  els.mode.value = settings.mode;
}

async function ensureContentScript() {
  try {
    await send({ type: "VSR_PING" });
  } catch {
    await chrome.scripting.insertCSS({
      target: { tabId: activeTabId },
      files: ["styles/overlay.css"]
    });
    await chrome.scripting.executeScript({
      target: { tabId: activeTabId },
      files: [
        "src/upscaler-core.js",
        "src/vendor/onnxruntime/ort.webgpu.min.js",
        "src/shaders/tiny-cnn.js",
        "src/shaders/webgpu.js",
        "src/upscaler-webgl.js",
        "src/upscaler-webgpu.js",
        "src/upscaler-onnx.js",
        "src/upscaler.js",
        "src/content.js"
      ]
    });
  }
}

function send(message) {
  return chrome.tabs.sendMessage(activeTabId, message);
}

function setStatus(text) {
  els.status.textContent = text;
}

function applyState(state, fallbackMessage) {
  if (state?.engine && state.engine !== settings.engine) {
    settings = { ...settings, engine: state.engine };
    chrome.storage.sync.set({ engine: state.engine });
    renderSettings();
  }
  if (state?.displayMode && state.displayMode !== settings.displayMode) {
    settings = { ...settings, displayMode: state.displayMode };
    chrome.storage.sync.set({ displayMode: state.displayMode });
    renderSettings();
  }
  setStatus(state?.message ?? fallbackMessage);
}

async function runDiagnostics() {
  els.runDiagnostics.disabled = true;
  els.diagnosticsOutput.textContent = "检测中...";

  try {
    const extensionDiagnostics = await gatherExtensionDiagnostics();
    const pageDiagnostics = activeTabId ? await gatherPageDiagnostics(activeTabId) : { error: "没有可用标签页" };
    els.diagnosticsOutput.textContent = formatDiagnostics(extensionDiagnostics, pageDiagnostics);
  } finally {
    els.runDiagnostics.disabled = false;
  }
}

async function gatherExtensionDiagnostics() {
  const base = {
    context: "extension",
    secureContext: window.isSecureContext,
    crossOriginIsolated: window.crossOriginIsolated,
    navigatorGpu: Boolean(navigator.gpu),
    userAgent: navigator.userAgent
  };

  if (!navigator.gpu) {
    return { ...base, adapter: null, adapterAttempts: ["navigator.gpu unavailable"] };
  }

  return {
    ...base,
    ...(await probeAdapter())
  };
}

async function gatherPageDiagnostics(tabId) {
  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId },
      world: "MAIN",
      func: async () => {
        const policy = document.permissionsPolicy || document.featurePolicy;
        const featureChecks = {};
        for (const feature of ["webgpu", "gpu"]) {
          try {
            featureChecks[feature] = policy?.allowsFeature ? policy.allowsFeature(feature) : null;
          } catch {
            featureChecks[feature] = "error";
          }
        }

        async function probeAdapterInPage() {
          if (!navigator.gpu) {
            return { adapter: null, adapterAttempts: ["navigator.gpu unavailable"] };
          }

          const attempts = [
            ["high-performance", { powerPreference: "high-performance", forceFallbackAdapter: false }],
            ["default", { forceFallbackAdapter: false }],
            ["compatibility", { powerPreference: "high-performance", featureLevel: "compatibility", forceFallbackAdapter: false }]
          ];
          const adapterAttempts = [];

          for (const [name, options] of attempts) {
            try {
              const adapter = await navigator.gpu.requestAdapter(options);
              if (adapter) {
                return {
                  adapter: {
                    name,
                    features: Array.from(adapter.features || []),
                    limits: {
                      maxTextureDimension2D: adapter.limits?.maxTextureDimension2D ?? null,
                      maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize ?? null
                    },
                    info: adapter.info
                      ? {
                          vendor: adapter.info.vendor || "",
                          architecture: adapter.info.architecture || "",
                          device: adapter.info.device || "",
                          description: adapter.info.description || ""
                        }
                      : null
                  },
                  adapterAttempts
                };
              }
              adapterAttempts.push(`${name}: null`);
            } catch (error) {
              adapterAttempts.push(`${name}: ${error instanceof Error ? error.message : String(error)}`);
            }
          }

          return { adapter: null, adapterAttempts };
        }

        return {
          context: "page",
          url: location.href,
          secureContext: window.isSecureContext,
          crossOriginIsolated: window.crossOriginIsolated,
          navigatorGpu: Boolean(navigator.gpu),
          permissionsPolicy: featureChecks,
          ...(await probeAdapterInPage())
        };
      }
    });

    return result;
  } catch (error) {
    return {
      context: "page",
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

async function probeAdapter() {
  const attempts = [
    ["high-performance", { powerPreference: "high-performance", forceFallbackAdapter: false }],
    ["default", { forceFallbackAdapter: false }],
    ["compatibility", { powerPreference: "high-performance", featureLevel: "compatibility", forceFallbackAdapter: false }]
  ];
  const adapterAttempts = [];

  for (const [name, options] of attempts) {
    try {
      const adapter = await navigator.gpu.requestAdapter(options);
      if (adapter) {
        return {
          adapter: {
            name,
            features: Array.from(adapter.features || []),
            limits: {
              maxTextureDimension2D: adapter.limits?.maxTextureDimension2D ?? null,
              maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize ?? null
            },
            info: adapter.info
              ? {
                  vendor: adapter.info.vendor || "",
                  architecture: adapter.info.architecture || "",
                  device: adapter.info.device || "",
                  description: adapter.info.description || ""
                }
              : null
          },
          adapterAttempts
        };
      }
      adapterAttempts.push(`${name}: null`);
    } catch (error) {
      adapterAttempts.push(`${name}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  return { adapter: null, adapterAttempts };
}

function formatDiagnostics(extensionDiagnostics, pageDiagnostics) {
  return [
    formatSection("扩展上下文", extensionDiagnostics),
    formatSection("页面上下文", pageDiagnostics)
  ].join("\n\n");
}

function formatSection(title, data) {
  const lines = [title];
  if (data.error) {
    lines.push(`error: ${data.error}`);
    return lines.join("\n");
  }

  if (data.url) lines.push(`url: ${data.url}`);
  lines.push(`secureContext: ${String(data.secureContext)}`);
  lines.push(`crossOriginIsolated: ${String(data.crossOriginIsolated)}`);
  lines.push(`navigator.gpu: ${String(data.navigatorGpu)}`);

  if (data.permissionsPolicy) {
    lines.push(
      `permissionsPolicy: webgpu=${String(data.permissionsPolicy.webgpu)}, gpu=${String(data.permissionsPolicy.gpu)}`
    );
  }

  if (data.adapter) {
    lines.push(`adapter: ${data.adapter.name}`);
    if (data.adapter.info) {
      lines.push(
        `adapterInfo: ${[
          data.adapter.info.vendor,
          data.adapter.info.architecture,
          data.adapter.info.device,
          data.adapter.info.description
        ].filter(Boolean).join(" | ") || "n/a"}`
      );
    }
    lines.push(
      `limits: maxTexture2D=${String(data.adapter.limits.maxTextureDimension2D)}, maxStorageBuffer=${String(data.adapter.limits.maxStorageBufferBindingSize)}`
    );
    lines.push(`features: ${data.adapter.features.slice(0, 8).join(", ") || "none"}`);
  } else {
    lines.push("adapter: null");
  }

  if (data.adapterAttempts?.length) {
    lines.push(`attempts: ${data.adapterAttempts.join(" ; ")}`);
  }

  return lines.join("\n");
}
