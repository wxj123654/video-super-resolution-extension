export type EngineType = "webgpu" | "tiny-cnn" | "onnx";
export type OnnxInputLayout = "NCHW";
export type OnnxInputColorSpace = "y-only" | "rgb";
export type OnnxCompositorKind = "luma_replace" | "rgb_replace" | "rgb_overlay";
export type OnnxExecutionProvider = "webgpu";
export type OnnxInputPacking = "luma_f32_planar" | "rgb_f32_planar";
export type OnnxOutputPacking = "luma_f32_planar" | "rgb_f32_planar";
export type OnnxCompositeMode = "luma_replace" | "rgb_replace" | "rgb_overlay";
export type OnnxModelCategory = "lightweight" | "balanced" | "quality";
export type OnnxPreprocessing = "range_01" | "range_neg1_1";
export type OnnxQuantization = "fp32" | "fp16";

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

export interface OnnxModelDefinition {
  id: string;
  label: string;
  modelPath: string;
  scale: number;
  inputLayout: OnnxInputLayout;
  inputChannels: 1 | 3;
  outputChannels: 1 | 3;
  inputColorSpace: OnnxInputColorSpace;
  sizePolicy: OnnxSizePolicy;
  executionProvider: OnnxExecutionProvider;
  inputPacking: OnnxInputPacking;
  outputPacking: OnnxOutputPacking;
  compositeMode: OnnxCompositeMode;
  compositor: OnnxCompositorKind;
  description?: string;
  source?: OnnxModelSource;
  quantization?: OnnxQuantization;
  category?: OnnxModelCategory;
  preprocessing?: OnnxPreprocessing;
  tileSize?: number;
  precisionAlternatives?: { fp16?: string; fp32?: string };
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
