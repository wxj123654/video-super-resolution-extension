import type { Settings } from "../../upscaler/types";
import { DEFAULT_SETTINGS } from "./defaults";
import { normalizeSettings } from "./settings";

export async function loadSettings(): Promise<Settings> {
  const stored = await chrome.storage.sync.get(DEFAULT_SETTINGS as unknown as Record<string, unknown>);
  return normalizeSettings(stored as unknown as Settings);
}

export async function saveSettings(patch: Partial<Settings>): Promise<Settings> {
  const normalized = normalizeSettings({
    ...(await loadSettings()),
    ...patch,
  });
  await chrome.storage.sync.set(normalized);
  return normalized;
}
