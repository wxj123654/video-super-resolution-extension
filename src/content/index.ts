import { Controller } from "./controller";
import { createLogger, toLogDetails } from "./debug";

const logger = createLogger("content");

if ((window as unknown as { __videoGpuSuperResolutionController?: unknown }).__videoGpuSuperResolutionController) {
  logger.warn("Duplicate content script initialization detected");
  throw new Error("Video GPU Super Resolution already initialized");
}

logger.info("Initializing content script", {
  url: location.href,
  readyState: document.readyState,
});

const controller = new Controller();
logger.info("Content script initialized", {
  url: location.href,
});

(window as unknown as { __videoGpuSuperResolutionController: unknown }).__videoGpuSuperResolutionController = controller;

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
      const state = controller.update(message.settings as Parameters<typeof controller.update>[0]);
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
  },
);
