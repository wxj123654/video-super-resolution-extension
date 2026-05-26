export type EngineType = "webgpu" | "tiny-cnn" | "ecbsr";

export interface Settings {
  enabled: boolean;
  scale: number;
  sharpness: number;
  mode: "balanced" | "quality" | "performance";
  overlayOpacity: number;
  displayMode: "overlay" | "replace";
  engine: EngineType;
  targetFps: "auto" | "60" | "30" | "24" | "15";
}

export interface UpscalerImpl {
  render(video: HTMLVideoElement, settings: Settings): boolean;
  destroy(): void;
}

export interface ControllerState {
  ok: boolean;
  message: string;
  hasVideo: boolean;
  engine: string;
  failedEngine: string;
  displayMode: string;
  overlay: {
    hidden: boolean;
    opacity: string;
    width: number;
    height: number;
    cssWidth: string;
    cssHeight: string;
  };
  video: {
    width: number;
    height: number;
    paused: boolean;
  } | null;
}

export type VsrMessageType = "VSR_PING" | "VSR_UPDATE" | "VSR_RESCAN";

export interface VsrMessage {
  type: VsrMessageType;
  settings?: Settings;
}

export interface SiteProfile {
  hostPattern?: RegExp;
  canvasClass: string;
  localOverlay: boolean;
  insertAfterVideo: boolean;
  hideSource: boolean;
  canvasOpacity: number;
  videoSelectors: string[];
  overlayRootSelector: string | null;
}

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

export interface WebGpuAdapterOptions {
  allowSoftware?: boolean;
  preferCompatibility?: boolean;
}
