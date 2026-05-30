struct PackParams {
  width: u32,
  height: u32,
  channels: u32,
  normScale: f32,
  normBias: f32,
  colorWeightR: f32,
  colorWeightG: f32,
  colorWeightB: f32,
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

  if (params.channels == 1u) {
    let luma = dot(color.rgb, vec3f(params.colorWeightR, params.colorWeightG, params.colorWeightB));
    dstBuffer[pixelIndex] = luma * params.normScale + params.normBias;
  } else {
    dstBuffer[pixelIndex] = color.r * params.normScale + params.normBias;
    dstBuffer[planeSize + pixelIndex] = color.g * params.normScale + params.normBias;
    dstBuffer[planeSize * 2u + pixelIndex] = color.b * params.normScale + params.normBias;
  }
}
