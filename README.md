# Video GPU Super Resolution

一个无构建步骤的 Chrome MV3 扩展原型，用 WebGL2 / WebGPU / ONNX Runtime Web 对当前页面 HTML5 视频做实时超分和增强。

## 加载方式

1. 打开 `chrome://extensions/`
2. 开启“开发者模式”
3. 点击“加载已解压的扩展程序”
4. 选择本目录：`/home/wxj/document/video-super-resolution-extension`

## 使用方式

1. 打开含 HTML5 `<video>` 的页面。
2. 点击扩展图标，打开“视频超分”开关。
3. 选择引擎和显示方式，并调整放大倍率、锐化、叠加强度和模式。
4. 页面动态切换视频后，点击“重新扫描视频”。

## 当前实现

- `src/content.js`：扫描当前页面中面积最大的可播放视频，创建覆盖 canvas。
- 已内置 Bilibili 适配：优先选择 B 站播放器视频层，支持动态加载/SPA 切换，并使用保留原视频的叠加模式，减少对控制条、弹幕和字幕层的遮挡。
- 如果叠加模式下效果不明显，可以临时切到“替换原视频”确认增强 canvas 是否正确显示。
- `src/upscaler-webgl.js`：仅保留 `轻量模型(WebGL)` 的三段卷积执行器。
- `src/upscaler-webgpu.js`：提供基于 `texture_external` 的 WebGPU 内容增强路径，优先请求硬件高性能 adapter。
- `src/upscaler-onnx.js`：提供 `ECBSR 模型(ONNX/WebGPU)` 的运行时，使用 `onnxruntime-web` 在浏览器内执行官方移动版模型。
- `src/upscaler-post.js` / `src/shaders/ecbsr-post.js`：负责把 ECBSR 的亮度输出交给 WebGL2 做实时合成，避免 JS 逐像素 `mergeLuma`。
- `src/shaders/tiny-cnn.js`：内置固定权重的小型 WebGL2 卷积管线。
- `src/shaders/webgpu.js`：当前 WebGPU 的增强 shader。
- `src/models/ecbsr_x2_m4c8_y.onnx`：由官方 ECBSR mobile checkpoint 导出的 ONNX 模型，输入为 Y 通道，倍率固定为 2x。
- `popup.js`：注入内容脚本，保存设置，并把设置发送到当前标签页。

## 限制

- 轻量模型是内置固定权重的小型 WebGL2 卷积管线，目标是实时增强；它不是 Real-ESRGAN/BasicVSR++ 这类大模型。
- `WebGPU 超分`会拒绝 `SwiftShader`/fallback adapter，只在拿到硬件 WebGPU adapter 时工作。
- `ECBSR 模型(ONNX/WebGPU)` 当前接入的是官方移动版 `x2 / m4c8 / Y-only` 模型。它更像“单帧亮度重建”而不是完整时序视频超分，对严重压缩伪影和时序细节不会像大视频模型那样重建。
- DRM 视频、跨源受限视频或某些站点自定义播放器可能禁止把视频帧上传到 GPU texture。
- 扩展只能增强普通网页里的 HTML5 视频，不能增强浏览器 UI、原生播放器或受保护媒体路径。
- 目前保留的三个引擎里，`轻量模型(WebGL)` 偏向稳定增强，`WebGPU 超分` 是 shader 增强链，`ECBSR 模型(ONNX/WebGPU)` 是真正的小模型单帧超分路径。

## 后续可扩展方向

- 如需接入真正的视频 SR 模型，建议优先评估 `RealBasicVSR`、`BasicVSR++`、`FRVSR` 这类开源路线，再决定是做离线推理还是浏览器内 WebGPU/ONNX 精简版。
- 增加站点适配器，精确处理 Bilibili、YouTube 等播放器的全屏和 theater mode。
- 优化渲染调度，降低主线程占用。
