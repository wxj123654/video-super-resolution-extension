import type { VsrMessage } from "@src/upscaler/types";

import {
  CONTENT_SCRIPT_FILE,
  CONTENT_STYLE_FILE,
  ORT_RUNTIME_SCRIPT_FILES,
} from "@src/shared/extension/injection";

export async function queryActiveTab(): Promise<chrome.tabs.Tab | undefined> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

export async function sendMessageToTab(
  tabId: number,
  message: VsrMessage,
): Promise<unknown> {
  return chrome.tabs.sendMessage(tabId, message);
}

export async function ensureTabReady(tabId: number): Promise<void> {
  try {
    await sendMessageToTab(tabId, { type: "VSR_PING" });
  } catch {
    await chrome.scripting.insertCSS({
      target: { tabId },
      files: [CONTENT_STYLE_FILE],
    });

    await chrome.scripting.executeScript({
      target: { tabId },
      files: [...ORT_RUNTIME_SCRIPT_FILES, CONTENT_SCRIPT_FILE],
    });
  }
}

export async function openExtensionOptions(): Promise<void> {
  await chrome.runtime.openOptionsPage();
}
