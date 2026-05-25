(function startVideoGpuSuperResolution() {
  if (window.__videoGpuSuperResolutionController) {
    return;
  }

  const DEFAULT_SETTINGS = {
    enabled: false,
    scale: 1.5,
    sharpness: 0.65,
    mode: "balanced",
    overlayOpacity: 0.8,
    displayMode: "overlay",
    engine: "tiny-cnn"
  };

  const SITE_PROFILES = {
    bilibili: {
      hostPattern: /(^|\.)bilibili\.com$/,
      canvasClass: "vgsr-site-bilibili",
      localOverlay: true,
      insertAfterVideo: true,
      hideSource: false,
      canvasOpacity: 0.8,
      videoSelectors: [
        ".bpx-player-video-wrap video",
        ".bilibili-player-video video",
        "#bilibili-player video",
        "video"
      ],
      overlayRootSelector: ".bpx-player-video-wrap, .bilibili-player-video, #bilibili-player"
    },
    default: {
      canvasClass: "",
      localOverlay: false,
      insertAfterVideo: false,
      hideSource: true,
      canvasOpacity: 1,
      videoSelectors: ["video"],
      overlayRootSelector: null
    }
  };

  class Controller {
    constructor() {
      this.settings = { ...DEFAULT_SETTINGS };
      this.profile = detectSiteProfile();
      this.video = null;
      this.canvas = this.createCanvas();
      this.canvasParent = null;
      this.canvasAnchor = null;
      this.upscaler = null;
      this.upscalerEngine = "";
      this.failedEngine = "";
      this.frame = 0;
      this.pendingRescan = 0;
      this.lastError = "";
      this.resizeObserver = new ResizeObserver(() => this.syncCanvasBounds());
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
      this.mutationObserver = new MutationObserver(() => this.scheduleRescan());
      this.mutationObserver.observe(document.documentElement, { childList: true, subtree: true });
    }

    update(settings) {
      const previousEngine = this.settings.engine;
      this.settings = { ...this.settings, ...settings };
      if (previousEngine !== this.settings.engine) {
        this.failedEngine = "";
      }
      this.lastError = "";

      if (!this.settings.enabled) {
        this.stop();
        return this.getState("关闭");
      }

      const previous = this.video;
      this.pickVideo();
      this.handleVideoChange(previous);
      if (!this.video) {
        this.stop(previous);
        return this.getState("没有检测到可播放视频");
      }

      this.start();
      return this.getState("运行中");
    }

    rescan() {
      this.lastError = "";
      const previous = this.video;
      this.pickVideo();
      this.handleVideoChange(previous);
      if (this.settings.enabled && this.video) {
        this.start();
      } else {
        this.stop(previous);
      }
      return this.getState(this.video ? "已重新扫描" : "没有检测到可播放视频");
    }

    pickVideo() {
      const videos = collectVideos(this.profile.videoSelectors);
      this.video = videos
        .map(({ video, priority }) => ({
          video,
          rect: video.getBoundingClientRect(),
          priority
        }))
        .filter(({ rect }) => rect.width >= 120 && rect.height >= 90)
        .sort((a, b) => scoreVideo(b) - scoreVideo(a))[0]?.video ?? null;
    }

    start() {
      if (!this.video) return;
      const engine = getEngine(this.settings);
      if (this.failedEngine === engine) {
        this.lastError ||= "当前引擎已失败，请切换引擎或关闭后重新开启再试";
        this.canvas.hidden = true;
        this.setSourceHidden(false);
        return;
      }
      if (!this.upscaler || this.upscalerEngine !== engine) {
        this.upscaler?.destroy();
        if (this.upscalerEngine) {
          this.replaceCanvas();
        }
        this.upscaler = new window.VideoGpuSuperResolution.Upscaler(this.canvas, { engine });
        this.upscalerEngine = engine;
      }
      this.resizeObserver.observe(this.video);
      this.syncCanvasBounds();
      if (!this.frame) {
        this.loop();
      }
    }

    stop(source = this.video) {
      cancelAnimationFrame(this.frame);
      this.frame = 0;
      this.canvas.hidden = true;
      source?.classList.remove("vgsr-hidden-source");
      this.resizeObserver.disconnect();
    }

    loop() {
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

      try {
        this.syncCanvasBounds();
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
        this.stop();
        return;
      }

      this.frame = requestAnimationFrame(() => this.loop());
    }

    syncCanvasBounds() {
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

      this.canvas.hidden = !visible || !this.settings.enabled || !isRenderableVideo(this.video);
      if (this.canvas.hidden) {
        this.setSourceHidden(false);
      }
      if (!visible) return;

      const scale = Math.max(1, Math.min(2, Number(this.settings.scale) || 1));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.max(1, Math.round(rect.width * dpr * scale));
      const height = Math.max(1, Math.round(rect.height * dpr * scale));
      const parentRect =
        this.profile.localOverlay && this.canvasParent && this.canvasParent !== document.documentElement
          ? this.canvasParent.getBoundingClientRect()
          : { left: 0, top: 0 };

      if (this.canvas.width !== width) this.canvas.width = width;
      if (this.canvas.height !== height) this.canvas.height = height;
      this.canvas.style.left = `${rect.left - parentRect.left}px`;
      this.canvas.style.top = `${rect.top - parentRect.top}px`;
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${rect.height}px`;
    }

    getState(message) {
      const status = this.lastError || message;
      return {
        ok: true,
        message: status,
        hasVideo: Boolean(this.video),
        engine: this.settings.engine,
        failedEngine: this.failedEngine,
        displayMode: this.getDisplayMode(),
        overlay: {
          hidden: this.canvas.hidden,
          opacity: this.canvas.style.opacity,
          width: this.canvas.width,
          height: this.canvas.height,
          cssWidth: this.canvas.style.width,
          cssHeight: this.canvas.style.height
        },
        video: this.video
          ? {
              width: this.video.videoWidth,
              height: this.video.videoHeight,
              paused: this.video.paused
            }
          : null
      };
    }

    handleVideoChange(previous) {
      if (previous !== this.video) {
        previous?.classList.remove("vgsr-hidden-source");
        this.resizeObserver.disconnect();
      }
    }

    scheduleRescan() {
      if (!this.settings.enabled || this.pendingRescan) return;
      this.pendingRescan = window.setTimeout(() => {
        this.pendingRescan = 0;
        const previous = this.video;
        this.pickVideo();
        this.handleVideoChange(previous);
        if (this.video) {
          this.start();
        }
      }, 120);
    }

    syncCanvasParent() {
      const { root, anchor } = this.getCanvasPlacement();
      this.appendCanvasTo(root, anchor);
      this.canvas.classList.toggle("vgsr-local-overlay", root !== document.documentElement);
      this.applyCanvasVisuals();
    }

    getCanvasPlacement() {
      if (!this.profile.localOverlay || !this.video) {
        return { root: document.documentElement, anchor: null };
      }

      if (this.profile.insertAfterVideo && this.video.parentElement) {
        return { root: this.video.parentElement, anchor: this.video };
      }

      const fullscreen = document.fullscreenElement;
      if (fullscreen?.contains(this.video)) {
        const videoLayer = this.video.closest(this.profile.overlayRootSelector);
        return { root: videoLayer && fullscreen.contains(videoLayer) ? videoLayer : fullscreen, anchor: null };
      }

      return { root: this.video.closest(this.profile.overlayRootSelector) ?? document.documentElement, anchor: null };
    }

    appendCanvasTo(parent, anchor) {
      if (this.canvasParent === parent && this.canvasAnchor === anchor) return;
      if (anchor?.parentElement === parent) {
        anchor.after(this.canvas);
      } else {
        parent.append(this.canvas);
      }
      this.canvasParent = parent;
      this.canvasAnchor = anchor;
      if (parent !== document.documentElement && getComputedStyle(parent).position === "static") {
        parent.style.position = "relative";
      }
    }

    createCanvas() {
      const canvas = document.createElement("canvas");
      canvas.className = ["vgsr-overlay", this.profile.canvasClass].filter(Boolean).join(" ");
      canvas.hidden = true;
      return canvas;
    }

    replaceCanvas() {
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

    applyCanvasVisuals() {
      this.canvas.dataset.vgsrEngine = this.settings.engine;
      this.canvas.dataset.vgsrDisplayMode = this.getDisplayMode();
      const configuredOpacity = Number(this.settings.overlayOpacity);
      const opacity = this.shouldReplaceSource()
        ? 1
        : Number.isFinite(configuredOpacity)
          ? configuredOpacity
          : this.profile.canvasOpacity ?? 1;
      this.canvas.style.opacity = String(Math.max(0, Math.min(1, opacity)));
    }

    setSourceHidden(hidden) {
      if (!this.shouldReplaceSource()) {
        this.video?.classList.remove("vgsr-hidden-source");
        return;
      }

      this.video?.classList.toggle("vgsr-hidden-source", hidden);
    }

    getDisplayMode() {
      if (this.settings.displayMode === "replace") return "replace";
      return "overlay";
    }

    shouldReplaceSource() {
      return this.profile.hideSource || this.getDisplayMode() === "replace";
    }

  }

  function detectSiteProfile() {
    const host = location.hostname;
    return Object.values(SITE_PROFILES).find((profile) => profile.hostPattern?.test(host)) ?? SITE_PROFILES.default;
  }

  function collectVideos(selectors) {
    const seen = new Set();
    const videos = [];

    selectors.forEach((selector, index) => {
      document.querySelectorAll(selector).forEach((video) => {
        if (seen.has(video)) return;
        seen.add(video);
        videos.push({ video, priority: selectors.length - index });
      });
    });

    return videos;
  }

  function scoreVideo({ video, rect, priority }) {
    const area = rect.width * rect.height;
    const readyBonus = isRenderableVideo(video) ? 1_000_000_000 : 0;
    const playingBonus = !video.paused && !video.ended ? 500_000_000 : 0;
    return priority * 10_000_000_000 + readyBonus + playingBonus + area;
  }

  function isRenderableVideo(video) {
    return (
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      video.videoWidth > 0 &&
      video.videoHeight > 0
    );
  }

  function getErrorMessage(error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/onnx|inferencesession|execution provider|ort\.|ecbsr/i.test(message)) {
      return `ECBSR 模型初始化失败：${message}`;
    }
    if (/swiftshader|fallback|hardware webgpu adapter/i.test(message)) {
      return "WebGPU 当前落到了软件适配器，已停止增强";
    }
    if (/webgpu|external texture|gpu canvas/i.test(message)) {
      return `WebGPU 初始化失败：${message}`;
    }
    if (/cross-origin|tainted|origin/i.test(message)) {
      return "当前视频受跨源限制，无法读取或上传视频帧";
    }
    return "渲染失败，已恢复原视频";
  }

  function getEngine(settings) {
    if (settings.engine === "webgpu") return "webgpu";
    if (settings.engine === "ecbsr") return "ecbsr";
    return "tiny-cnn";
  }

  const controller = new Controller();
  window.__videoGpuSuperResolutionController = controller;

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "VSR_PING") {
      sendResponse(controller.getState("已连接"));
      return;
    }

    if (message?.type === "VSR_UPDATE") {
      sendResponse(controller.update(message.settings));
      return;
    }

    if (message?.type === "VSR_RESCAN") {
      sendResponse(controller.rescan());
    }
  });
})();
