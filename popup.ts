import type { Settings, EngineType } from "./src/upscaler/types";

const DEFAULT_SETTINGS: Settings = {
  enabled: false,
  scale: 1.5,
  sharpness: 0.65,
  mode: "balanced",
  overlayOpacity: 0.8,
  displayMode: "overlay",
  engine: "tiny-cnn",
  targetFps: "auto",
};

const VALID_ENGINES = new Set<EngineType>(["webgpu", "tiny-cnn", "ecbsr"]);

const els = {
  enabled: document.querySelector<HTMLInputElement>("#enabled")!,
  scale: document.querySelector<HTMLSelectElement>("#scale")!,
  sharpness: document.querySelector<HTMLInputElement>("#sharpness")!,
  sharpnessValue: document.querySelector<HTMLOutputElement>("#sharpnessValue")!,
  overlayOpacity: document.querySelector<HTMLInputElement>("#overlayOpacity")!,
  overlayOpacityValue: document.querySelector<HTMLOutputElement>("#overlayOpacityValue")!,
  displayMode: document.querySelector<HTMLSelectElement>("#displayMode")!,
  engine: document.querySelector<HTMLSelectElement>("#engine")!,
  mode: document.querySelector<HTMLSelectElement>("#mode")!,
  targetFps: document.querySelector<HTMLSelectElement>("#targetFps")!,
  status: document.querySelector<HTMLParagraphElement>("#status")!,
  rescan: document.querySelector<HTMLButtonElement>("#rescan")!,
  runDiagnostics: document.querySelector<HTMLButtonElement>("#runDiagnostics")!,
  diagnosticsOutput: document.querySelector<HTMLPreElement>("#diagnosticsOutput")!,
};

let activeTabId: number | null = null;
let settings: Settings = { ...DEFAULT_SETTINGS };

init();

async function init(): Promise<void> {
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
    applyState(
      state,
      settings.enabled ? "已连接，增强已启用" : "已连接，当前关闭",
    );
  } catch {
    setStatus("当前页面不支持注入");
  }
}

for (const key of [
  "enabled",
  "scale",
  "sharpness",
  "overlayOpacity",
  "displayMode",
  "engine",
  "mode",
  "targetFps",
] as const) {
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

function readSettings(): Settings {
  return normalizeSettings({
    enabled: els.enabled.checked,
    scale: Number(els.scale.value),
    sharpness: Number(els.sharpness.value),
    overlayOpacity: Number(els.overlayOpacity.value),
    displayMode: els.displayMode.value as Settings["displayMode"],
    engine: els.engine.value as EngineType,
    mode: els.mode.value as Settings["mode"],
    targetFps: els.targetFps.value as Settings["targetFps"],
  });
}

function normalizeSettings(value: Partial<Settings> & { engine?: string }): Settings {
  return {
    ...DEFAULT_SETTINGS,
    ...value,
    engine: VALID_ENGINES.has(value.engine as EngineType)
      ? (value.engine as EngineType)
      : DEFAULT_SETTINGS.engine,
  };
}

function renderSettings(): void {
  els.enabled.checked = settings.enabled;
  els.scale.value = String(settings.scale);
  els.sharpness.value = String(settings.sharpness);
  els.sharpnessValue.value = settings.sharpness.toFixed(2);
  els.overlayOpacity.value = String(settings.overlayOpacity);
  els.overlayOpacityValue.value = settings.overlayOpacity.toFixed(2);
  els.displayMode.value = settings.displayMode;
  els.engine.value = settings.engine;
  els.mode.value = settings.mode;
  els.targetFps.value = settings.targetFps;
}

async function ensureContentScript(): Promise<void> {
  try {
    await send({ type: "VSR_PING" });
  } catch {
    await chrome.scripting.insertCSS({
      target: { tabId: activeTabId! },
      files: ["styles/overlay.css"],
    });
    await chrome.scripting.executeScript({
      target: { tabId: activeTabId! },
      files: [new URL("content.js", import.meta.url).pathname],
    });
  }
}

function send(message: { type: string; settings?: Settings }): Promise<unknown> {
  return chrome.tabs.sendMessage(activeTabId!, message);
}

function setStatus(text: string): void {
  els.status.textContent = text;
}

function applyState(state: unknown, fallbackMessage: string): void {
  const s = state as Record<string, unknown> | null;
  if (s?.engine && s.engine !== settings.engine) {
    settings = { ...settings, engine: s.engine as EngineType };
    chrome.storage.sync.set({ engine: s.engine });
    renderSettings();
  }
  if (s?.displayMode && s.displayMode !== settings.displayMode) {
    settings = { ...settings, displayMode: s.displayMode as Settings["displayMode"] };
    chrome.storage.sync.set({ displayMode: s.displayMode });
    renderSettings();
  }
  setStatus(String(s?.message ?? fallbackMessage));
}

async function runDiagnostics(): Promise<void> {
  els.runDiagnostics.disabled = true;
  els.diagnosticsOutput.textContent = "检测中...";

  try {
    const extensionDiagnostics = await gatherExtensionDiagnostics();
    const pageDiagnostics = activeTabId
      ? await gatherPageDiagnostics(activeTabId)
      : { error: "没有可用标签页" };
    els.diagnosticsOutput.textContent = formatDiagnostics(
      extensionDiagnostics,
      pageDiagnostics,
    );
  } finally {
    els.runDiagnostics.disabled = false;
  }
}

async function gatherExtensionDiagnostics(): Promise<Record<string, unknown>> {
  const base: Record<string, unknown> = {
    context: "extension",
    secureContext: window.isSecureContext,
    crossOriginIsolated: window.crossOriginIsolated,
    navigatorGpu: Boolean(navigator.gpu),
    userAgent: navigator.userAgent,
  };

  if (!navigator.gpu) {
    return { ...base, adapter: null, adapterAttempts: ["navigator.gpu unavailable"] };
  }

  return {
    ...base,
    ...(await probeAdapter()),
  };
}

async function gatherPageDiagnostics(
  tabId: number,
): Promise<Record<string, unknown>> {
  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId },
      world: "MAIN",
      func: async () => {
        const policy = (document as unknown as Record<string, unknown>).permissionsPolicy || (document as unknown as Record<string, unknown>).featurePolicy;
        const featureChecks: Record<string, unknown> = {};
        for (const feature of ["webgpu", "gpu"]) {
          try {
            const policyObj = policy as Record<string, unknown> | null;
            featureChecks[feature] = typeof policyObj?.allowsFeature === "function"
              ? (policyObj.allowsFeature as (f: string) => boolean)(feature)
              : null;
          } catch {
            featureChecks[feature] = "error";
          }
        }

        async function probeAdapterInPage(): Promise<Record<string, unknown>> {
          if (!navigator.gpu) {
            return { adapter: null, adapterAttempts: ["navigator.gpu unavailable"] };
          }

          const attempts = [
            ["high-performance", { powerPreference: "high-performance", forceFallbackAdapter: false }],
            ["default", { forceFallbackAdapter: false }],
            ["compatibility", { powerPreference: "high-performance", featureLevel: "compatibility", forceFallbackAdapter: false }],
          ];
          const adapterAttempts: string[] = [];

          for (const [name, options] of attempts) {
            try {
              const adapter = await navigator.gpu.requestAdapter(options as GPURequestAdapterOptions);
              if (adapter) {
                return {
                  adapter: {
                    name,
                    features: Array.from(adapter.features || []),
                    limits: {
                      maxTextureDimension2D: adapter.limits?.maxTextureDimension2D ?? null,
                      maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize ?? null,
                    },
                    info: adapter.info
                      ? {
                          vendor: adapter.info.vendor || "",
                          architecture: adapter.info.architecture || "",
                          device: adapter.info.device || "",
                          description: adapter.info.description || "",
                        }
                      : null,
                  },
                  adapterAttempts,
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
          ...(await probeAdapterInPage()),
        };
      },
    });

    return result as Record<string, unknown>;
  } catch (error) {
    return {
      context: "page",
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

async function probeAdapter(): Promise<Record<string, unknown>> {
  const attempts = [
    ["high-performance", { powerPreference: "high-performance", forceFallbackAdapter: false }],
    ["default", { forceFallbackAdapter: false }],
    ["compatibility", { powerPreference: "high-performance", featureLevel: "compatibility", forceFallbackAdapter: false }],
  ];
  const adapterAttempts: string[] = [];

  for (const [name, options] of attempts) {
    try {
      const adapter = await navigator.gpu.requestAdapter(options as GPURequestAdapterOptions);
      if (adapter) {
        return {
          adapter: {
            name,
            features: Array.from(adapter.features || []),
            limits: {
              maxTextureDimension2D: adapter.limits?.maxTextureDimension2D ?? null,
              maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize ?? null,
            },
            info: adapter.info
              ? {
                  vendor: adapter.info.vendor || "",
                  architecture: adapter.info.architecture || "",
                  device: adapter.info.device || "",
                  description: adapter.info.description || "",
                }
              : null,
          },
          adapterAttempts,
        };
      }
      adapterAttempts.push(`${name}: null`);
    } catch (error) {
      adapterAttempts.push(`${name}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  return { adapter: null, adapterAttempts };
}

function formatDiagnostics(
  extensionDiagnostics: Record<string, unknown>,
  pageDiagnostics: Record<string, unknown>,
): string {
  return [
    formatSection("扩展上下文", extensionDiagnostics),
    formatSection("页面上下文", pageDiagnostics),
  ].join("\n\n");
}

function formatSection(
  title: string,
  data: Record<string, unknown>,
): string {
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
    const pp = data.permissionsPolicy as Record<string, unknown>;
    lines.push(
      `permissionsPolicy: webgpu=${String(pp.webgpu)}, gpu=${String(pp.gpu)}`,
    );
  }

  const adapter = data.adapter as Record<string, unknown> | null;
  if (adapter) {
    lines.push(`adapter: ${adapter.name}`);
    const info = adapter.info as Record<string, string> | null;
    if (info) {
      lines.push(
        `adapterInfo: ${[
          info.vendor,
          info.architecture,
          info.device,
          info.description,
        ]
          .filter(Boolean)
          .join(" | ") || "n/a"}`,
      );
    }
    const limits = adapter.limits as Record<string, unknown>;
    lines.push(
      `limits: maxTexture2D=${String(limits?.maxTextureDimension2D)}, maxStorageBuffer=${String(limits?.maxStorageBufferBindingSize)}`,
    );
    const features = adapter.features as string[];
    lines.push(`features: ${features?.slice(0, 8).join(", ") || "none"}`);
  } else {
    lines.push("adapter: null");
  }

  if ((data.adapterAttempts as string[])?.length) {
    lines.push(
      `attempts: ${(data.adapterAttempts as string[]).join(" ; ")}`,
    );
  }

  return lines.join("\n");
}
