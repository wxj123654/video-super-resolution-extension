import type { ControllerState, Settings } from "@src/upscaler/types";

import { createMessage } from "@src/shared/extension/messages";
import {
  DEFAULT_SETTINGS,
  normalizeSettings,
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
    const syncedSettings = await syncSettingsFromState(settings, state);

    return {
      tabId: tab.id,
      model: mapPopupModel(syncedSettings, state, "Connected"),
    };
  } catch (error) {
    console.error("[VSR] bootstrapPopup failed:", error);
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
    const syncedSettings = await syncSettingsFromState(settings, state);

    return {
      tabId,
      model: mapPopupModel(syncedSettings, state, "Connected"),
    };
  } catch (error) {
    console.error("[VSR] updatePopupSettings failed:", error);
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
    const syncedSettings = await syncSettingsFromState(settings, state);

    return {
      tabId,
      model: mapPopupModel(syncedSettings, state, "Connected"),
    };
  } catch (error) {
    console.error("[VSR] rescanPopup failed:", error);
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

async function syncSettingsFromState(
  settings: Settings,
  state: ControllerState | null | undefined,
): Promise<Settings> {
  const nextSettings = normalizeSettings({
    ...settings,
    engine: state?.engine ?? settings.engine,
    modelId: state?.modelId ?? settings.modelId,
    displayMode:
      (state?.displayMode as Settings["displayMode"] | undefined) ??
      settings.displayMode,
  });

  if (
    nextSettings.engine === settings.engine &&
    nextSettings.modelId === settings.modelId &&
    nextSettings.displayMode === settings.displayMode
  ) {
    return settings;
  }

  return saveSettings({
    engine: nextSettings.engine,
    modelId: nextSettings.modelId,
    displayMode: nextSettings.displayMode,
  });
}

function getEngineLabel(engine: Settings["engine"]): string {
  return ENGINE_LABELS[engine] ?? engine;
}
