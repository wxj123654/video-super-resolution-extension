struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
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
@group(0) @binding(2) var lumaTex: texture_2d<f32>;

fn sampleLumaBilinear(uv: vec2f) -> f32 {
  let dims = textureDimensions(lumaTex);
  let pixCoord = uv * vec2f(dims) - 0.5;
  let tl = vec2i(floor(pixCoord));
  let f = fract(pixCoord);
  let d1 = vec2i(dims) - 1;
  let c00 = textureLoad(lumaTex, clamp(tl, vec2i(0), d1), 0).r;
  let c10 = textureLoad(lumaTex, clamp(tl + vec2i(1, 0), vec2i(0), d1), 0).r;
  let c01 = textureLoad(lumaTex, clamp(tl + vec2i(0, 1), vec2i(0), d1), 0).r;
  let c11 = textureLoad(lumaTex, clamp(tl + vec2i(1, 1), vec2i(0), d1), 0).r;
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}


fn lumaFromRgb(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let uv = clamp(input.uv, vec2f(0.0), vec2f(1.0));
  let base = textureSampleBaseClampToEdge(videoTex, videoSampler, uv).rgb;
  let baseY = lumaFromRgb(base);
  let modelY = clamp(sampleLumaBilinear(uv), 0.0, 1.0);
  let y = mix(baseY, modelY, 1.0);
  let gain = y / max(baseY, 0.0001);
  var color = base * clamp(gain, 0.55, 1.8);
  let minColor = max(base - vec3f(0.12), vec3f(0.0));
  let maxColor = min(base + vec3f(0.12), vec3f(1.0));
  return vec4f(clamp(color, minColor, maxColor), 1.0);
}
