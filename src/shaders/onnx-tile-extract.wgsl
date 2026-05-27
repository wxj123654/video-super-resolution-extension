struct ExtractParams {
  srcWidth: u32,
  srcHeight: u32,
  srcChannels: u32,
  tileX: u32,
  tileY: u32,
  tileW: u32,
  tileH: u32,
};

@group(0) @binding(0) var<storage, read> srcBuffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> dstBuffer: array<f32>;
@group(0) @binding(2) var<uniform> params: ExtractParams;

@compute @workgroup_size(8, 8, 1)
fn computeMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.tileW || gid.y >= params.tileH) {
    return;
  }

  let srcPlaneSize = params.srcWidth * params.srcHeight;
  let dstPlaneSize = params.tileW * params.tileH;
  let srcX = params.tileX + gid.x;
  let srcY = params.tileY + gid.y;
  let srcPixelIdx = srcY * params.srcWidth + srcX;
  let dstPixelIdx = gid.y * params.tileW + gid.x;

  for (var c = 0u; c < params.srcChannels; c++) {
    dstBuffer[c * dstPlaneSize + dstPixelIdx] =
      srcBuffer[c * srcPlaneSize + srcPixelIdx];
  }
}
