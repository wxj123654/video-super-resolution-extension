# Video GPU Super Resolution

一个基于 TypeScript + Vite 的 Chrome MV3 扩展，用 WebGL2 / WebGPU / ONNX Runtime Web 对当前页面 HTML5 视频做实时超分和增强。

## 加载方式

1. 打开 `chrome://extensions/`
2. 开启"开发者模式"
3. 点击"加载已解压的扩展程序"
4. 选择 `dist/` 目录

## 使用方式

1. 打开含 HTML5 `<video>` 的页面。
2. 点击扩展图标，打开"视频超分"开关。
3. 选择引擎（WebGPU 超分 / 轻量模型 / ONNX 超分）。
4. 若使用 ONNX 引擎，可在"模型"下拉框中选择模型（ECBSR Y-only x2 / RGB Bicubic x2）。
5. 调整放大倍率、锐化、叠加强度、模式和目标帧率。
6. 页面动态切换视频后，点击"重新扫描视频"。

## 引擎与模型

| 引擎 | 模型 | 输入 | 合成 | 说明 |
|------|------|------|------|------|
| 轻量模型 (WebGL) | 内置固定权重 | RGBA | WebGL2 合成 | 三段卷积管线，偏向稳定增强 |
| WebGPU 超分 | — | texture_external | WebGPU shader | 基于 texture_external 的 shader 增强链 |
| ONNX 超分 | ECBSR Y-only x2 | Y 通道 | luma_replace (WebGPU) | ECBSR 移动版亮度超分，全 GPU 零拷贝 |
| ONNX 超分 | RGB Bicubic x2 | RGB 三通道 | rgb_replace (WebGPU / 2D) | RGB 三通道基线模型，验证全通路 |

ONNX 引擎支持多模型切换，通过弹窗"模型"下拉框选择。模型定义在 `src/backends/onnx-models.ts` 中注册，详见 `docs/onnx-models.md`。

## 架构概览

```
popup.ts                    扩展弹窗：设置管理 + 内容脚本注入
src/content/
  index.ts                  内容脚本入口，消息分发
  controller.ts             视频扫描、canvas 管理、渲染循环
  debug.ts                  结构化调试日志系统
  video-utils.ts            视频/引擎工具函数
  site-profiles.ts          站点适配（Bilibili 等）
src/upscaler/
  index.ts                  Upscaler 工厂，按引擎创建实例
  types.ts                  共享类型（Settings, ControllerState, OnnxModelDefinition 等）
src/backends/
  onnx-upscaler.ts          ONNX 多模型推理管线
  onnx-models.ts            模型定义注册表
  onnx-webgpu-utilities.ts  WebGPU adapter 请求策略
  webgl-upscaler.ts         轻量模型 WebGL 执行器
  webgpu-upscaler.ts        WebGPU shader 增强路径
src/shaders/
  tiny-cnn-*.glsl           WebGL 卷积着色器
  webgpu.wgsl               WebGPU 增强着色器
  onnx-luma.wgsl            Y 通道 GPU 打包
  onnx-pack-rgb.wgsl        RGB 三通道 GPU 打包
  onnx-composite.wgsl       luma_replace 合成
  onnx-rgb-unpack.wgsl      RGB storage buffer → texture 解包
  onnx-present.wgsl         texture_2d → canvas 输出
  onnx-video-copy.wgsl      视频帧 → texture_2d 复制
public/models/              ONNX 模型文件
```

### ONNX 管线流程

**Y-only 模型（ECBSR）：**
`texture_external` → GPU luma 提取 → `ort.Tensor.fromGpuBuffer` 零上传 → ONNX WebGPU 推理 → `copyBufferToTexture` → luma_replace 合成 shader → canvas

**RGB 模型：**
`texture_external` → GPU RGB 打包（compute shader → NCHW planar buffer）→ ONNX WebGPU 推理 → `copyBufferToTexture` → RGB 解包 shader（或 2D canvas fallback）→ canvas

## 构建

```bash
pnpm install           # 安装依赖
pnpm run dev           # 开发模式 (HMR)
pnpm run build         # 生产构建 → dist/
pnpm run build:debug   # Debug 构建 → dist/，保留 sourcemap 和详细日志
pnpm run typecheck     # 类型检查
```

在 `chrome://extensions/` 加载 `dist/` 目录即可使用。

## 限制

- 轻量模型是内置固定权重的小型 WebGL2 卷积管线，目标是实时增强；它不是 Real-ESRGAN/BasicVSR++ 这类大模型。
- `WebGPU 超分`会拒绝 SwiftShader/fallback adapter，只在拿到硬件 WebGPU adapter 时工作。
- ONNX 引擎当前内置 ECBSR Y-only x2（移动版亮度超分）和 RGB Bicubic x2（基线验证模型）。它们偏向单帧超分而非时序视频超分，对严重压缩伪影和时序细节不会像大视频模型那样重建。
- DRM 视频、跨源受限视频或某些站点自定义播放器可能禁止把视频帧上传到 GPU texture。
- 扩展只能增强普通网页里的 HTML5 视频，不能增强浏览器 UI、原生播放器或受保护媒体路径。
- ONNX 全 GPU 管线要求浏览器支持 WebGPU 且 `ort.Tensor.fromGpuBuffer` 可用；不满足时自动降级到 CPU 路径。
- RGB Bicubic x2 是验证用基线模型，效果等同于双三次上采样，不提供实质性超分增益。

## 后续可扩展方向

- 接入更多 ONNX 超分模型（如 Real-ESRGAN 精简版、SwinIR-light），利用多模型注册表热插拔。
- 评估 RealBasicVSR、BasicVSR++、FRVSR 等时序视频 SR 模型的浏览器内可行性。
- 增加站点适配器，精确处理 YouTube 等播放器的全屏和 theater mode。
- 优化渲染调度，降低主线程占用。

## Changelog

详见 [CHANGELOG.md](./CHANGELOG.md)。
