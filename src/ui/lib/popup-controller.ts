import type { ControllerState, Settings } from "@src/upscaler/types";

import { createMessage } from "@src/shared/extension/messages";
import {
  DEFAULT_SETTINGS,
  normalizeSettings,
  ONNX_MODEL_OPTIONS,
} from "@src/shared/extension/settings";
import { loadSettings, saveSettings } from "@src/shared/extension/storage";

import {
  ensureTabReady,
  openExtensionOptions,
  queryActiveTab,
  sendMessageToTab,
} from "./chrome";

export type PopupModel = {
  statusText: string;
  connectionLabel: string;
  hasVideo: boolean;
  activeEngineLabel: string;
  settings: Settings;
};

export type PopupSession = {
  tabId: number | null;
  model: PopupModel;
};

const ENGINE_LABELS: Record<Settings["engine"], string> = {
  webgpu: "WebGPU",
  "tiny-cnn": "Tiny CNN",
  ecbsr: "ECBSR",
};

export const INITIAL_POPUP_MODEL: PopupModel = {
  statusText: "正在连接当前标签页",
  connectionLabel: "Connecting",
  hasVideo: false,
  activeEngineLabel: ENGINE_LABELS[DEFAULT_SETTINGS.engine],
  settings: DEFAULT_SETTINGS,
};

export async function bootstrapPopup(): Promise<PopupSession> {
  const settings = normalizeSettings(await loadSettings());
  const tab = await queryActiveTab();

  if (!tab?.id) {
    return {
      tabId: null,
      model: {
        ...INITIAL_POPUP_MODEL,
        settings,
        connectionLabel: "Unavailable",
        statusText: "未找到可用标签页",
        activeEngineLabel: getEngineLabel(settings.engine),
      },
    };
  }

  try {
    await ensureTabReady(tab.id);
    const state = (await sendMessageToTab(
      tab.id,
      createMessage("VSR_UPDATE", settings),
    )) as ControllerState;

    return {
      tabId: tab.id,
      model: mapPopupModel(settings, state, "Connected"),
    };
  } catch {
    return {
      tabId: tab.id,
      model: {
        settings,
        hasVideo: false,
        connectionLabel: "Unavailable",
        activeEngineLabel: getEngineLabel(settings.engine),
        statusText: "当前页面无法连接或不支持注入",
      },
    };
  }
}

export async function updatePopupSettings(
  tabId: number | null,
  patch: Partial<Settings>,
): Promise<PopupSession> {
  const settings = await saveSettings(patch);

  if (tabId == null) {
    return {
      tabId: null,
      model: {
        settings,
        hasVideo: false,
        connectionLabel: "Unavailable",
        activeEngineLabel: getEngineLabel(settings.engine),
        statusText: "设置已保存，但没有可用标签页",
      },
    };
  }

  try {
    const state = (await sendMessageToTab(
      tabId,
      createMessage("VSR_UPDATE", settings),
    )) as ControllerState;

    return {
      tabId,
      model: mapPopupModel(settings, state, "Connected"),
    };
  } catch {
    return {
      tabId,
      model: {
        settings,
        hasVideo: false,
        connectionLabel: "Unavailable",
        activeEngineLabel: getEngineLabel(settings.engine),
        statusText: "设置已保存，但当前页面未响应",
      },
    };
  }
}

export async function rescanPopup(tabId: number | null): Promise<PopupSession> {
  const settings = normalizeSettings(await loadSettings());

  if (tabId == null) {
    return {
      tabId: null,
      model: {
        ...INITIAL_POPUP_MODEL,
        settings,
        connectionLabel: "Unavailable",
        statusText: "没有可重新扫描的标签页",
        activeEngineLabel: getEngineLabel(settings.engine),
      },
    };
  }

  try {
    const state = (await sendMessageToTab(
      tabId,
      createMessage("VSR_RESCAN"),
    )) as ControllerState;

    return {
      tabId,
      model: mapPopupModel(settings, state, "Connected"),
    };
  } catch {
    return {
      tabId,
      model: {
        settings,
        hasVideo: false,
        connectionLabel: "Unavailable",
        activeEngineLabel: getEngineLabel(settings.engine),
        statusText: "重新扫描失败，当前页面未响应",
      },
    };
  }
}

export async function launchOptionsPage(): Promise<void> {
  await openExtensionOptions();
}

export function getModelLabel(modelId: string): string {
  return (
    ONNX_MODEL_OPTIONS.find((option) => option.id === modelId)?.label ??
    ONNX_MODEL_OPTIONS[0].label
  );
}

function mapPopupModel(
  settings: Settings,
  state: ControllerState | null | undefined,
  connectionLabel: string,
): PopupModel {
  return {
    settings,
    hasVideo: Boolean(state?.hasVideo),
    connectionLabel,
    statusText: state?.message ?? "已连接当前标签页",
    activeEngineLabel: getEngineLabel(
      (state?.engine as Settings["engine"] | undefined) ?? settings.engine,
    ),
  };
}

function getEngineLabel(engine: Settings["engine"]): string {
  return ENGINE_LABELS[engine] ?? engine;
}
