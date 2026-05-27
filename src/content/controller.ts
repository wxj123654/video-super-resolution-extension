import type { Settings, ControllerState, SiteProfile } from "../upscaler/types";
import { Upscaler } from "../upscaler/index";
import { getOnnxModelDefinition } from "../backends/onnx-models";
import { DEFAULT_SETTINGS } from "../shared/extension/defaults";
import { detectSiteProfile } from "./site-profiles";
import {
  collectVideos,
  scoreVideo,
  isRenderableVideo,
  getErrorMessage,
  getEngine,
  getOnnxModelId,
  getPipelineKey,
} from "./video-utils";
import { createLogger, toLogDetails } from "./debug";

export { DEFAULT_SETTINGS };

const logger = createLogger("controller");

export class Controller {
  settings: Settings = { ...DEFAULT_SETTINGS };
  private profile: SiteProfile;
  private video: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement;
  private canvasParent: HTMLElement | null = null;
  private canvasAnchor: Element | null = null;
  private upscaler: Upscaler | null = null;
  private upscalerKey = "";
  private failedEngine = "";
  private failedPipelineKey = "";
  private frame = 0;
  private pendingRescan = 0;
  private lastError = "";
  private failedEngineSkipLogged = false;
  private lastRenderTime = 0;
  private lastVideoTime = -1;
  private resizeObserver: ResizeObserver;
  private mutationObserver: MutationObserver;
  private handleScroll: () => void;
  private handleResize: () => void;
  private handleMediaChange: () => void;

  constructor() {
    this.profile = detectSiteProfile();
    logger.info("Controller created", {
      host: location.hostname,
      profile: {
        canvasClass: this.profile.canvasClass,
        localOverlay: this.profile.localOverlay,
        insertAfterVideo: this.profile.insertAfterVideo,
        hideSource: this.profile.hideSource,
        overlayRootSelector: this.profile.overlayRootSelector,
        videoSelectors: this.profile.videoSelectors,
      },
    });
    this.canvas = this.createCanvas();
    this.appendCanvasTo(document.documentElement, null);
    this.applyCanvasVisuals();
    this.handleScroll = () => this.syncCanvasBounds();
    this.handleResize = () => this.syncCanvasBounds();
    this.handleMediaChange = () => this.scheduleRescan();
    window.addEventListener("scroll", this.handleScroll, true);
    window.addEventListener("resize", this.handleResize);
    document.addEventListener("fullscreenchange", this.handleResize);
    document.addEventListener("loadedmetadata", this.handleMediaChange, true);
    document.addEventListener("loadeddata", this.handleMediaChange, true);
    document.addEventListener("playing", this.handleMediaChange, true);
    this.resizeObserver = new ResizeObserver(() => this.syncCanvasBounds());
    this.mutationObserver = new MutationObserver(() => this.scheduleRescan());
    this.mutationObserver.observe(document.documentElement, {
      childList: true,
      subtree: true,
      characterData: false,
    });
  }

  destroy(): void {
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    window.removeEventListener("scroll", this.handleScroll, true);
    window.removeEventListener("resize", this.handleResize);
    document.removeEventListener("fullscreenchange", this.handleResize);
    document.removeEventListener("loadedmetadata", this.handleMediaChange, true);
    document.removeEventListener("loadeddata", this.handleMediaChange, true);
    document.removeEventListener("playing", this.handleMediaChange, true);
    this.resizeObserver.disconnect();
    this.mutationObserver.disconnect();
    this.upscaler?.destroy();
    this.upscaler = null;
    this.canvas.remove();
    delete document.documentElement.dataset["dataVgsrInjected"];
  }

  update(settings: Partial<Settings>): ControllerState {
    logger.debug("Update requested", {
      incomingSettings: settings,
      currentSettings: this.settings,
    });
    const previousPipelineKey = getPipelineKey(this.settings);
    this.settings = { ...this.settings, ...settings };
    this.settings.modelId = getOnnxModelId(this.settings);
    if (previousPipelineKey !== getPipelineKey(this.settings)) {
      this.failedEngine = "";
      this.failedPipelineKey = "";
      this.failedEngineSkipLogged = false;
    }
    this.lastError = "";

    if (!this.settings.enabled) {
      this.stop();
      const state = this.getState("关闭");
      logger.info("Enhancement disabled", this.summarizeState(state));
      return state;
    }

    const previous = this.video;
    this.pickVideo();
    this.handleVideoChange(previous);
    if (!this.video) {
      this.stop(previous);
      const state = this.getState("没有检测到可播放视频");
      logger.warn("No playable video detected during update", {
        selectors: this.profile.videoSelectors,
        state: this.summarizeState(state),
      });
      return state;
    }

    this.start();
    const state = this.getState("运行中");
    logger.info("Enhancement started", this.summarizeState(state));
    return state;
  }

  rescan(): ControllerState {
    logger.debug("Rescan requested", {
      enabled: this.settings.enabled,
      currentVideo: this.summarizeVideo(this.video),
    });
    this.lastError = "";
    const previous = this.video;
    this.pickVideo();
    this.handleVideoChange(previous);
    if (this.settings.enabled && this.video) {
      this.start();
    } else {
      this.stop(previous);
    }
    const state = this.getState(
      this.video ? "已重新扫描" : "没有检测到可播放视频",
    );
    logger.info("Rescan completed", this.summarizeState(state));
    return state;
  }

  private pickVideo(): void {
    const videos = collectVideos(this.profile.videoSelectors);
    this.video =
      videos
        .map(({ video, priority }) => ({
          video,
          rect: video.getBoundingClientRect(),
          priority,
        }))
        .filter(({ rect }) => rect.width >= 120 && rect.height >= 90)
        .sort((a, b) => scoreVideo(b) - scoreVideo(a))[0]?.video ?? null;
    logger.debug("Video selection completed", {
      candidateCount: videos.length,
      selectedVideo: this.summarizeVideo(this.video),
    });
  }

  private start(): void {
    if (!this.video) return;
    const engine = getEngine(this.settings);
    const pipelineKey = getPipelineKey(this.settings);
    if (this.failedPipelineKey === pipelineKey) {
      this.lastError ||=
        "当前引擎已失败，请切换引擎或关闭后重新开启再试";
      this.canvas.hidden = true;
      this.setSourceHidden(false);
      if (!this.failedEngineSkipLogged) {
        logger.warn("Start skipped because engine is marked failed", {
          engine,
          modelId: this.settings.modelId,
          state: this.summarizeState(this.getState(this.lastError)),
        });
        this.failedEngineSkipLogged = true;
      }
      return;
    }
    if (!this.upscaler || this.upscalerKey !== pipelineKey) {
      logger.info("Creating upscaler", {
        engine,
        modelId: this.settings.modelId,
        previousKey: this.upscalerKey || null,
      });
      this.upscaler?.destroy();
      if (this.upscalerKey) {
        this.replaceCanvas();
      }
      this.upscaler = new Upscaler(this.canvas, {
        engine,
        modelId: this.settings.modelId,
      });
      this.upscalerKey = pipelineKey;
    }
    this.resizeObserver.observe(this.video);
    this.syncCanvasBounds();
    logger.debug("Upscaler ready", {
      engine,
      modelId: this.settings.modelId,
      video: this.summarizeVideo(this.video),
      canvas: {
        width: this.canvas.width,
        height: this.canvas.height,
      },
    });
    if (!this.frame) {
      this.loop();
    }
  }

  private stop(source: HTMLVideoElement | null = this.video): void {
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.canvas.hidden = true;
    source?.classList.remove("vgsr-hidden-source");
    this.resizeObserver.disconnect();
  }

  private loop(): void {
    if (!this.settings.enabled || !this.video) {
      this.frame = 0;
      return;
    }

    if (!isRenderableVideo(this.video)) {
      this.video.classList.remove("vgsr-hidden-source");
      this.canvas.hidden = true;
      this.frame = requestAnimationFrame(() => this.loop());
      return;
    }

    if (!this.shouldRenderFrame()) {
      this.frame = requestAnimationFrame(() => this.loop());
      return;
    }

    try {
      this.syncCanvasBounds();
      if (!this.upscaler) return;
      const rendered = this.upscaler.render(this.video, this.settings);
      if (rendered && !this.canvas.hidden) {
        this.setSourceHidden(true);
      } else {
        this.canvas.hidden = true;
        this.setSourceHidden(false);
      }
    } catch (error) {
      console.warn("[Video GPU Super Resolution]", error);
      this.lastError = getErrorMessage(error);
      this.failedEngine = getEngine(this.settings);
      this.failedPipelineKey = getPipelineKey(this.settings);
      this.failedEngineSkipLogged = false;
      logger.warn("Rendering loop failed", {
        engine: this.failedEngine,
        modelId: this.settings.modelId,
        error,
        userMessage: this.lastError,
        video: this.summarizeVideo(this.video),
      });
      this.stop();
      return;
    }

    this.frame = requestAnimationFrame(() => this.loop());
  }

  private shouldRenderFrame(): boolean {
    if (!this.video) return false;
    const targetFps = this.settings.targetFps;

    if (targetFps === "auto") {
      if (
        this.video.currentTime === this.lastVideoTime &&
        this.lastVideoTime >= 0
      ) {
        return false;
      }
      this.lastVideoTime = this.video.currentTime;
      return true;
    }

    const fps = Math.max(1, Number(targetFps) || 30);
    const now = performance.now();
    if (now - this.lastRenderTime < 1000 / fps) {
      return false;
    }
    this.lastRenderTime = now;
    return true;
  }

  private syncCanvasBounds(): void {
    if (!this.video) return;
    this.syncCanvasParent();
    const rect = this.video.getBoundingClientRect();
    const visible =
      rect.width > 0 &&
      rect.height > 0 &&
      rect.bottom > 0 &&
      rect.right > 0 &&
      rect.top < window.innerHeight &&
      rect.left < window.innerWidth;

    this.canvas.hidden =
      !visible || !this.settings.enabled || !isRenderableVideo(this.video);
    if (this.canvas.hidden) {
      this.setSourceHidden(false);
    }
    if (!visible) return;

    const scale = Math.max(1, Math.min(2, Number(this.settings.scale) || 1));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width * dpr * scale));
    const height = Math.max(1, Math.round(rect.height * dpr * scale));
    const parentRect =
      this.profile.localOverlay &&
      this.canvasParent &&
      this.canvasParent !== document.documentElement
        ? this.canvasParent.getBoundingClientRect()
        : { left: 0, top: 0 };

    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    this.canvas.style.left = `${rect.left - parentRect.left}px`;
    this.canvas.style.top = `${rect.top - parentRect.top}px`;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
  }

  getState(message: string): ControllerState {
    const status = this.lastError || message;
    return {
      ok: true,
      message: status,
      hasVideo: Boolean(this.video),
      engine: this.settings.engine,
      failedEngine: this.failedEngine,
      modelId: this.settings.engine === "ecbsr" ? this.settings.modelId : undefined,
      modelLabel:
        this.settings.engine === "ecbsr"
          ? getOnnxModelDefinition(this.settings.modelId).label
          : undefined,
      displayMode: this.getDisplayMode(),
      overlay: {
        hidden: this.canvas.hidden,
        opacity: this.canvas.style.opacity,
        width: this.canvas.width,
        height: this.canvas.height,
        cssWidth: this.canvas.style.width,
        cssHeight: this.canvas.style.height,
      },
      video: this.video
        ? {
            width: this.video.videoWidth,
            height: this.video.videoHeight,
            paused: this.video.paused,
          }
        : null,
    };
  }

  private handleVideoChange(
    previous: HTMLVideoElement | null,
  ): void {
    if (previous !== this.video) {
      previous?.classList.remove("vgsr-hidden-source");
      this.resizeObserver.disconnect();
    }
  }

  private scheduleRescan(): void {
    if (!this.settings.enabled || this.pendingRescan) return;
    if (this.failedPipelineKey === getPipelineKey(this.settings)) {
      logger.debug("Skipping scheduled rescan because engine is marked failed", {
        engine: this.failedEngine,
        modelId: this.settings.modelId,
      });
      return;
    }
    logger.debug("Scheduling rescan", {
      delayMs: 120,
      currentVideo: this.summarizeVideo(this.video),
    });
    this.pendingRescan = window.setTimeout(() => {
      this.pendingRescan = 0;
      const previous = this.video;
      this.pickVideo();
      this.handleVideoChange(previous);
      if (this.video) {
        this.start();
        logger.debug("Scheduled rescan found video", {
          video: this.summarizeVideo(this.video),
        });
      } else {
        logger.warn("Scheduled rescan did not find a playable video");
      }
    }, 120);
  }

  private syncCanvasParent(): void {
    const { root, anchor } = this.getCanvasPlacement();
    this.appendCanvasTo(root, anchor);
    this.canvas.classList.toggle(
      "vgsr-local-overlay",
      root !== document.documentElement,
    );
    this.applyCanvasVisuals();
  }

  private getCanvasPlacement(): {
    root: HTMLElement;
    anchor: Element | null;
  } {
    if (!this.profile.localOverlay || !this.video) {
      return { root: document.documentElement, anchor: null };
    }

    if (this.profile.insertAfterVideo && this.video.parentElement) {
      return { root: this.video.parentElement, anchor: this.video };
    }

    const fullscreen = document.fullscreenElement;
    if (fullscreen?.contains(this.video)) {
      const videoLayer = this.profile.overlayRootSelector
        ? this.video.closest(this.profile.overlayRootSelector)
        : null;
      const root =
        videoLayer instanceof HTMLElement && fullscreen.contains(videoLayer)
          ? videoLayer
          : fullscreen instanceof HTMLElement
            ? fullscreen
            : document.documentElement;
      return { root, anchor: null };
    }

    const overlayRoot = this.profile.overlayRootSelector
      ? this.video.closest(this.profile.overlayRootSelector)
      : null;
    return {
      root: overlayRoot instanceof HTMLElement ? overlayRoot : document.documentElement,
      anchor: null,
    };
  }

  private appendCanvasTo(
    parent: HTMLElement,
    anchor: Element | null,
  ): void {
    if (this.canvasParent === parent && this.canvasAnchor === anchor) return;
    if (anchor?.parentElement === parent) {
      anchor.after(this.canvas);
    } else {
      parent.append(this.canvas);
    }
    this.canvasParent = parent;
    this.canvasAnchor = anchor;
    if (
      parent !== document.documentElement &&
      getComputedStyle(parent).position === "static"
    ) {
      parent.style.position = "relative";
    }
  }

  private createCanvas(): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    canvas.className = ["vgsr-overlay", this.profile.canvasClass]
      .filter(Boolean)
      .join(" ");
    canvas.hidden = true;
    return canvas;
  }

  private replaceCanvas(): void {
    const previous = this.canvas;
    const next = this.createCanvas();
    next.hidden = previous.hidden;
    next.style.cssText = previous.style.cssText;
    previous.replaceWith(next);
    this.canvas = next;
    this.canvasParent = null;
    this.canvasAnchor = null;
    this.applyCanvasVisuals();
  }

  private applyCanvasVisuals(): void {
    this.canvas.dataset.vgsrEngine = this.settings.engine;
    this.canvas.dataset.vgsrModelId = this.settings.modelId;
    this.canvas.dataset.vgsrDisplayMode = this.getDisplayMode();
    const configuredOpacity = Number(this.settings.overlayOpacity);
    const opacity = this.shouldReplaceSource()
      ? 1
      : Number.isFinite(configuredOpacity)
        ? configuredOpacity
        : (this.profile.canvasOpacity ?? 1);
    this.canvas.style.opacity = String(
      Math.max(0, Math.min(1, opacity)),
    );
  }

  private setSourceHidden(hidden: boolean): void {
    if (!this.shouldReplaceSource()) {
      this.video?.classList.remove("vgsr-hidden-source");
      return;
    }

    this.video?.classList.toggle("vgsr-hidden-source", hidden);
  }

  private getDisplayMode(): Settings["displayMode"] {
    if (this.settings.displayMode === "replace") return "replace";
    return "overlay";
  }

  private shouldReplaceSource(): boolean {
    return this.profile.hideSource || this.getDisplayMode() === "replace";
  }

  private summarizeVideo(video: HTMLVideoElement | null): Record<string, unknown> | null {
    if (!video) return null;
    return {
      currentSrc: video.currentSrc || video.src || "",
      readyState: video.readyState,
      paused: video.paused,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      rect: video.getBoundingClientRect().toJSON(),
    };
  }

  private summarizeState(state: ControllerState): Record<string, unknown> {
    return toLogDetails({
      message: state.message,
      hasVideo: state.hasVideo,
      engine: state.engine,
      failedEngine: state.failedEngine,
      modelId: state.modelId,
      modelLabel: state.modelLabel,
      displayMode: state.displayMode,
      overlay: state.overlay,
      video: state.video,
    }) as Record<string, unknown>;
  }
}
