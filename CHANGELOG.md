# Changelog

## v0.4.0 — 多模型 ONNX GPU 管线

- **多模型 ONNX 管线**：ONNX 推理路径从单一 ECBSR 硬编码扩展为数据驱动的多模型架构。新增 `OnnxModelDefinition` 类型系统，通过 `src/backends/onnx-models.ts` 注册模型元数据（输入/输出通道、色彩空间、尺寸对齐策略、合成模式），运行时根据 `modelId` 自动选择管线。
- **RGB 三通道模型支持**：新增 `RGB Bicubic x2` 基线模型（`rgb_bicubic_x2.onnx`），验证 RGB 三通道 ONNX 全通路。配套实现 GPU compute shader 打包（`onnx-pack-rgb.wgsl`，`textureLoad` → NCHW planar storage buffer）和解包（`onnx-rgb-unpack.wgsl`，storage buffer → `texture_2d` render）。
- **模型选择 UI**：弹窗新增"模型"下拉框，仅在 ONNX 引擎激活时显示。用户可切换 ECBSR Y-only x2 和 RGB Bicubic x2 模型，选择通过 `chrome.storage.sync` 持久化。
- **管线键（pipeline key）驱动重建**：控制器用 `engine + modelId` 组合键判断管线变更，避免仅切换模型时未重建 upscaler 的问题。
- **多 session 缓存**：`ORT_STATE.sessionPromises` 从单 session 改为 `Map<modelId, SessionBundle>`，支持多个 ONNX 模型 session 共存复用。
- **GPU 输入打包管线**：新增 `InputPacker` 抽象，根据模型 `inputPacking` 配置（`luma_f32_planar` / `rgb_f32_planar`）选择对应的 compute shader 将视频帧打包为 NCHW planar GPU buffer。luma 路径复用现有 `onnx-luma.wgsl`，RGB 路径使用新 `onnx-pack-rgb.wgsl`。
- **多合成器架构**：`onnx-upscaler.ts` 重构为三种合成器——`WebGpuLumaCompositor`（Y-only → luma_replace）、`WebGpuRgbCompositor`（RGB → 全 GPU 合成）、`RgbCanvasCompositor`（RGB → 2D canvas CPU fallback）。根据模型 `compositor` 字段自动选择。
- **全 GPU RGB 合成路径**：`WebGpuRgbCompositor` 使用 `onnx-present.wgsl`（`texture_2d` 采样 → canvas render）或 `onnx-rgb-unpack.wgsl`（storage buffer 直接 → render）实现 RGB 模型的全 GPU 输出。
- **CPU fallback 双路径**：Y-only 和 RGB 模型均保留 CPU 回退路径（`extractLumaCpu` / `extractRgbCpu`），在 GPU buffer 不可用时自动降级。
- **结构化调试日志系统**：新增 `src/content/debug.ts`，提供 `createLogger(scope)` 工厂。所有模块（controller、content script、popup）统一使用 `[VSR][scope]` 前缀的结构化日志，通过 `__VSR_DEBUG__` 编译开关控制（`npm run build:debug` 启用）。
- **`build:debug` 模式**：Vite 配置改为函数式，新增 `--mode debug` 支持，关闭 minify、开启 sourcemap、注入 `__VSR_DEBUG__` 全局常量。
- **WebGPU adapter 请求策略增强**：将 adapter 请求逻辑抽取为 `src/backends/onnx-webgpu-utilities.ts`，依次尝试 high-performance → default → compatibility adapter，增强软件适配器检测（`isFallbackAdapter` + SwiftShader 关键字匹配）。
- **`onnx-video-copy.wgsl`**：新增视频帧 → `texture_2d` 复制 shader，用于 RGB 路径的 GPU 输入准备。
- **ONNX 模型文档**：新增 `docs/onnx-models.md`，记录内置模型目录和添加新模型的步骤。
- **弹窗错误增强**：popup 所有异步操作路径增加结构化错误日志，`setStatus` 增加 `reason` 标识便于排查。

## v0.3.0 — TypeScript + Vite 构建迁移

- **TypeScript 全量迁移**：所有源文件从纯 JS 转为 TypeScript strict 模式，添加 `@types/chrome`、`@webgpu/types` 类型覆盖。
- **Vite + @crxjs/vite-plugin 构建系统**：引入 Vite 作为构建工具，CRXJS 插件处理 Manifest V3 扩展打包，支持 HMR 开发模式。
- **ES Module 架构**：移除 IIFE 全局命名空间模式，改用 ES Module 导入导出。内容脚本由 10 个独立文件按序注入合并为单个 bundled chunk。
- **着色器文件独立管理**：GLSL/WGSL 着色器从 JS 模板字符串提取为独立 `.glsl`/`.wgsl` 文件，通过 Vite `?raw` 导入，获得语法高亮和 LSP 支持。
- **onnxruntime-web npm 集成**：移除手动管理的 vendor 文件，改用 `onnxruntime-web` npm 包，消除 `window.ort` 全局依赖，由 bundler 自动处理 WASM 输出。
- **模块化拆分**：`content.js`（464 行）拆分为 `controller.ts`、`site-profiles.ts`、`video-utils.ts`、`index.ts`；`upscaler-core.js` 拆分为 `webgl-utilities.ts` 和 `webgpu-utilities.ts`。
- **运行时行为完全不变**：三个引擎（WebGL tiny-cnn、WebGPU shader、ECBSR ONNX）、弹窗交互、消息通信、站点适配等均保持原有行为。

## v0.2.0 — ECBSR 全 GPU 管线

**性能：推理延迟从 ~23ms 降至 ~0.8ms（29x 加速）**

- **全 GPU 零拷贝推理**：用 `ort.Tensor.fromGpuBuffer` 将 luma GPU buffer 零上传送入 ONNX，推理结果写入 pre-allocated GPU output buffer（`COPY_SRC | COPY_DST | STORAGE`），全程无 CPU 参与。
- **WebGPU compositor 替代 WebGL compositor**：canvas 改用 WebGPU 上下文，合成 shader 通过 `importExternalTexture` 直接采样视频帧 + R32F luma 纹理，消除 GPU→CPU→GPU 回环。
- **`textureLoad` 手动双线性插值**：R32Float 纹理的 sampleType 为 `UnfilterableFloat`，无法使用 `textureSample`；改用 `textureLoad` + 4 次采样 + `mix` 实现等价双线性插值。
- **输入宽度对齐到 32px**：满足 `copyBufferToTexture` 的 `bytesPerRow` 256 字节对齐要求。
- **GPU luma 提取管线**：用 `texture_external` + render pipeline 提取亮度到 R32F 纹理，再 `copyTextureToBuffer` 到 GPU buffer，替代 CPU `getImageData` 路径。
- **可配置目标帧率**：新增"目标帧率"设置，默认跟随视频帧率（仅在 `video.currentTime` 变化时渲染），可选 60/30/24/15 fps。避免 GPU 被高刷新率 rAF 吃满。
