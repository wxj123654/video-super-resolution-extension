import { Controller } from "./controller";

if ((window as unknown as { __videoGpuSuperResolutionController?: unknown }).__videoGpuSuperResolutionController) {
  throw new Error("Video GPU Super Resolution already initialized");
}

const controller = new Controller();

(window as unknown as { __videoGpuSuperResolutionController: unknown }).__videoGpuSuperResolutionController = controller;

chrome.runtime.onMessage.addListener(
  (
    message: { type: string; settings?: unknown },
    _sender: chrome.runtime.MessageSender,
    sendResponse: (response?: unknown) => void,
  ) => {
    if (message?.type === "VSR_PING") {
      sendResponse(controller.getState("已连接"));
      return;
    }

    if (message?.type === "VSR_UPDATE") {
      sendResponse(controller.update(message.settings as Parameters<typeof controller.update>[0]));
      return;
    }

    if (message?.type === "VSR_RESCAN") {
      sendResponse(controller.rescan());
    }
  },
);
