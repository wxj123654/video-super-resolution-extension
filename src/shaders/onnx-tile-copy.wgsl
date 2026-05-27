struct CopyParams {
  dstWidth: u32,
  dstHeight: u32,
  channels: u32,
  dstX: u32,
  dstY: u32,
  tileW: u32,
  tileH: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> srcBuffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> dstBuffer: array<f32>;
@group(0) @binding(2) var<uniform> params: CopyParams;

@compute @workgroup_size(8, 8, 1)
fn computeMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.tileW || gid.y >= params.tileH) {
    return;
  }

  let srcPlaneSize = params.tileW * params.tileH;
  let dstPlaneSize = params.dstWidth * params.dstHeight;
  let dstAbsX = params.dstX + gid.x;
  let dstAbsY = params.dstY + gid.y;
  let dstPixelIdx = dstAbsY * params.dstWidth + dstAbsX;
  let srcPixelIdx = gid.y * params.tileW + gid.x;

  for (var c = 0u; c < params.channels; c++) {
    dstBuffer[c * dstPlaneSize + dstPixelIdx] =
      srcBuffer[c * srcPlaneSize + srcPixelIdx];
  }
}
