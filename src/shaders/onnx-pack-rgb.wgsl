struct PackParams {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dstBuffer: array<f32>;
@group(0) @binding(2) var<uniform> params: PackParams;

@compute @workgroup_size(8, 8, 1)
fn computeMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let color = textureLoad(srcTex, vec2i(gid.xy), 0);
  let planeSize = params.width * params.height;
  let pixelIndex = gid.y * params.width + gid.x;
  dstBuffer[pixelIndex] = color.r;
  dstBuffer[planeSize + pixelIndex] = color.g;
  dstBuffer[planeSize * 2u + pixelIndex] = color.b;
}
