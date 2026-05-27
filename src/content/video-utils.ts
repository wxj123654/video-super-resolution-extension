import type { Settings, EngineType, SiteProfile } from "../upscaler/types";
import {
  DEFAULT_ONNX_MODEL_ID,
  getOnnxModelDefinition,
} from "../backends/onnx-models";

interface VideoCandidate {
  video: HTMLVideoElement;
  priority: number;
}

interface ScoredVideo {
  video: HTMLVideoElement;
  rect: DOMRect;
  priority: number;
}

export function collectVideos(selectors: string[]): VideoCandidate[] {
  const seen = new Set<HTMLVideoElement>();
  const videos: VideoCandidate[] = [];

  selectors.forEach((selector, index) => {
    document.querySelectorAll(selector).forEach((el) => {
      const video = el as HTMLVideoElement;
      if (seen.has(video)) return;
      seen.add(video);
      videos.push({ video, priority: selectors.length - index });
    });
  });

  return videos;
}

export function scoreVideo({ video, rect, priority }: ScoredVideo): number {
  const area = rect.width * rect.height;
  const readyBonus = isRenderableVideo(video) ? 1_000_000_000 : 0;
  const playingBonus = !video.paused && !video.ended ? 500_000_000 : 0;
  return priority * 10_000_000_000 + readyBonus + playingBonus + area;
}

export function isRenderableVideo(video: HTMLVideoElement): boolean {
  return (
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.videoWidth > 0 &&
    video.videoHeight > 0
  );
}

export function getErrorMessage(error: unknown): string {
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

export function getEngine(settings: Settings): EngineType {
  return settings.engine;
}

export function getOnnxModelId(settings: Settings): string {
  if (settings.engine !== "ecbsr") {
    return DEFAULT_ONNX_MODEL_ID;
  }
  return getOnnxModelDefinition(settings.modelId).id;
}

export function getPipelineKey(settings: Settings): string {
  if (settings.engine === "ecbsr") {
    return `ecbsr:${getOnnxModelId(settings)}`;
  }
  return getEngine(settings);
}
