# Changelog

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
