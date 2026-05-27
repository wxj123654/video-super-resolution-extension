import { Controller } from "./controller";
import { normalizeSettings } from "../shared/extension/settings";
import { createLogger, toLogDetails } from "./debug";

const logger = createLogger("content");

const GUARD_ATTR = "dataVgsrInjected";

function claimInjectionLock(): boolean {
  if (document.documentElement.dataset[GUARD_ATTR]) return false;
  document.documentElement.dataset[GUARD_ATTR] = "1";
  return true;
}

if (!claimInjectionLock()) {
  logger.warn("Duplicate content script initialization detected");
} else {
  logger.info("Initializing content script", {
    url: location.href,
    readyState: document.readyState,
  });

  const controller = new Controller();
  logger.info("Content script initialized", {
    url: location.href,
  });

  chrome.runtime.onMessage.addListener(
    (
      message: { type: string; settings?: unknown },
      _sender: chrome.runtime.MessageSender,
      sendResponse: (response?: unknown) => void,
    ) => {
      logger.debug("Received runtime message", {
        type: message?.type ?? "unknown",
      });

      if (message?.type === "VSR_PING") {
        const state = controller.getState("已连接");
        logger.debug("Responding to ping", toLogDetails(state));
        sendResponse(state);
        return;
      }

      if (message?.type === "VSR_UPDATE") {
        const validated = normalizeSettings(
          message.settings as Record<string, unknown>,
        );
        const state = controller.update(validated);
        logger.debug("Responding to update", toLogDetails(state));
        sendResponse(state);
        return;
      }

      if (message?.type === "VSR_RESCAN") {
        const state = controller.rescan();
        logger.debug("Responding to rescan", toLogDetails(state));
        sendResponse(state);
        return;
      }

      logger.warn("Received unsupported runtime message", {
        message: toLogDetails(message),
      });
    } catch (error) {
      logger.error("Message handler error", { type: message?.type, error });
      sendResponse({ ok: false, message: `处理消息失败: ${error instanceof Error ? error.message : String(error)}` });
    }
  },
);

