export type EngineType = "webgpu" | "tiny-cnn" | "onnx";
export type OnnxInputLayout = "NCHW";
export type OnnxExecutionProvider = "webgpu";
export type OnnxModelCategory = "lightweight" | "balanced" | "quality";
export type OnnxQuantization = "fp32" | "fp16";

export type OnnxCompositeMode = "replace" | "luma_inject" | "overlay";

export interface OnnxSizePolicy {
  widthAlign: number;
  heightAlign: number;
  minInputWidth?: number;
  minInputHeight?: number;
  fixedScale?: number;
}

export interface OnnxModelSource {
  type: "bundled" | "download";
  downloadUrl?: string;
  fileSize?: number;
  sha256?: string;
}

export interface OnnxModelInput {
  channels: 1 | 3;
  colorWeights?: [number, number, number];
  normalization: { scale: number; bias: number };
}

export interface OnnxModelOutput {
  channels: 1 | 3;
  denormalization?: { scale: number; bias: number };
}

export interface OnnxModelComposite {
  mode: OnnxCompositeMode;
  params?: {
    lumaClampMin?: number;
    lumaClampMax?: number;
    colorDeviation?: number;
    blendStrength?: number;
  };
}

export interface OnnxModelDefinition {
  id: string;
  label: string;
  modelPath: string;
  scale: number;
  inputLayout: OnnxInputLayout;
  sizePolicy: OnnxSizePolicy;
  executionProvider: OnnxExecutionProvider;
  description?: string;
  source?: OnnxModelSource;
  quantization?: OnnxQuantization;
  category?: OnnxModelCategory;
  tileSize?: number;
  tilePoolSize?: number;
  precisionAlternatives?: { fp16?: string; fp32?: string };

  input: OnnxModelInput;
  output: OnnxModelOutput;
  composite: OnnxModelComposite;
}

export interface Settings {
  enabled: boolean;
  scale: number;
  sharpness: number;
  mode: "balanced" | "quality" | "performance";
  overlayOpacity: number;
  displayMode: "overlay" | "replace";
  engine: EngineType;
  modelId: string;
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
  modelId?: string;
  modelLabel?: string;
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

export type VsrMessageType = "VSR_PING" | "VSR_UPDATE" | "VSR_RESCAN" | "VSR_MODEL_STATUS";

export interface VsrMessage {
  type: VsrMessageType;
  settings?: Settings;
  modelStatus?: {
    modelId: string;
    state: "cached" | "downloading" | "ready" | "error";
    progress?: number;
    total?: number;
    error?: string;
  };
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
