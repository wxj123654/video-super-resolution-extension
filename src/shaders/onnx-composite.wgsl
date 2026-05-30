struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct CompositeParams {
  blendMode: u32,
  lumaClampMin: f32,
  lumaClampMax: f32,
  colorDeviation: f32,
  blendStrength: f32,
};

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  var uvs = array<vec2f, 3>(
    vec2f(0.0, 1.0),
    vec2f(2.0, 1.0),
    vec2f(0.0, -1.0)
  );

  var output: VertexOutput;
  output.position = vec4f(positions[index], 0.0, 1.0);
  output.uv = uvs[index];
  return output;
}

@group(0) @binding(0) var videoTex: texture_external;
@group(0) @binding(1) var videoSampler: sampler;
@group(0) @binding(2) var modelTex: texture_2d<f32>;
@group(0) @binding(3) var modelSampler: sampler;
@group(0) @binding(4) var<uniform> params: CompositeParams;

fn lumaFromRgb(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

fn sampleLumaBilinear(uv: vec2f) -> f32 {
  let dims = textureDimensions(modelTex);
  let pixCoord = uv * vec2f(dims) - 0.5;
  let tl = vec2i(floor(pixCoord));
  let f = fract(pixCoord);
  let d1 = vec2i(dims) - 1;
  let c00 = textureLoad(modelTex, clamp(tl, vec2i(0), d1), 0).r;
  let c10 = textureLoad(modelTex, clamp(tl + vec2i(1, 0), vec2i(0), d1), 0).r;
  let c01 = textureLoad(modelTex, clamp(tl + vec2i(0, 1), vec2i(0), d1), 0).r;
  let c11 = textureLoad(modelTex, clamp(tl + vec2i(1, 1), vec2i(0), d1), 0).r;
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let uv = clamp(input.uv, vec2f(0.0), vec2f(1.0));
  let base = textureSampleBaseClampToEdge(videoTex, videoSampler, uv).rgb;

  // blendMode 0: replace — direct model output
  if (params.blendMode == 0u) {
    let model = textureSample(modelTex, modelSampler, uv).rgb;
    return vec4f(model, 1.0);
  }

  // blendMode 1: luma_inject — replace luma, keep original chroma
  if (params.blendMode == 1u) {
    let baseY = lumaFromRgb(base);
    let modelY = clamp(sampleLumaBilinear(uv), 0.0, 1.0);
    let gain = modelY / max(baseY, 0.0001);
    var color = base * clamp(gain, params.lumaClampMin, params.lumaClampMax);
    let minColor = max(base - vec3f(params.colorDeviation), vec3f(0.0));
    let maxColor = min(base + vec3f(params.colorDeviation), vec3f(1.0));
    return vec4f(clamp(color, minColor, maxColor), 1.0);
  }

  // blendMode 2: overlay — alpha-blend model output with original
  let model = textureSample(modelTex, modelSampler, uv).rgb;
  let strength = params.blendStrength;
  return vec4f(mix(base, model, strength), 1.0);
}
