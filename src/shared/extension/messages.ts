import type { Settings, VsrMessage, VsrMessageType } from "../../upscaler/types";

export function createMessage(
  type: VsrMessageType,
  settings?: Settings,
): VsrMessage {
  return settings ? { type, settings } : { type };
}
