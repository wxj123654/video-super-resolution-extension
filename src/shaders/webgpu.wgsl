struct Params {
  texel: vec2f,
  sharpness: f32,
  mode: f32,
};

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

@group(0) @binding(0) var videoFrame: texture_external;
@group(0) @binding(1) var videoSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

fn luma(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

fn sampleVideo(uv: vec2f) -> vec3f {
  return textureSampleBaseClampToEdge(videoFrame, videoSampler, clamp(uv, vec2f(0.0), vec2f(1.0))).rgb;
}

fn edgeAdaptiveUpscale(uv: vec2f, texel: vec2f) -> vec3f {
  let c = sampleVideo(uv);
  let n = sampleVideo(uv + vec2f(0.0, -texel.y));
  let s = sampleVideo(uv + vec2f(0.0, texel.y));
  let e = sampleVideo(uv + vec2f(texel.x, 0.0));
  let w = sampleVideo(uv + vec2f(-texel.x, 0.0));
  let ne = sampleVideo(uv + texel * vec2f(1.0, -1.0));
  let nw = sampleVideo(uv + texel * vec2f(-1.0, -1.0));
  let se = sampleVideo(uv + texel * vec2f(1.0, 1.0));
  let sw = sampleVideo(uv + texel * vec2f(-1.0, 1.0));

  let gx = luma(e) - luma(w);
  let gy = luma(s) - luma(n);
  let edge = smoothstep(0.02, 0.16, length(vec2f(gx, gy)));
  let axisBlend = select((e + w) * 0.5, (n + s) * 0.5, abs(gx) > abs(gy));
  let diagBlend = select((nw + se) * 0.5, (ne + sw) * 0.5, abs(luma(se) - luma(nw)) > abs(luma(sw) - luma(ne)));
  let directional = mix(axisBlend, diagBlend, 0.33);
  let detail = c - directional;
  let diagonal = (ne + nw + se + sw) * 0.125 + (n + s + e + w) * 0.125 + c * 0.5;
  return mix(diagonal, c + detail * 1.35, edge);
}

fn contrastAdaptiveSharpen(color: vec3f, uv: vec2f, texel: vec2f, strength: f32) -> vec3f {
  let n = sampleVideo(uv + vec2f(0.0, -texel.y));
  let s = sampleVideo(uv + vec2f(0.0, texel.y));
  let e = sampleVideo(uv + vec2f(texel.x, 0.0));
  let w = sampleVideo(uv + vec2f(-texel.x, 0.0));
  let blur = (n + s + e + w + color * 4.0) * 0.125;
  let contrast = max(abs(luma(e) - luma(w)), abs(luma(n) - luma(s)));
  let limiter = smoothstep(0.015, 0.18, contrast);
  return color + (color - blur) * strength * (0.38 + limiter * 0.72);
}

fn antiRing(color: vec3f, uv: vec2f, texel: vec2f) -> vec3f {
  let c = sampleVideo(uv);
  let n = sampleVideo(uv + vec2f(0.0, -texel.y));
  let s = sampleVideo(uv + vec2f(0.0, texel.y));
  let e = sampleVideo(uv + vec2f(texel.x, 0.0));
  let w = sampleVideo(uv + vec2f(-texel.x, 0.0));
  let minColor = min(c, min(min(n, s), min(e, w)));
  let maxColor = max(c, max(max(n, s), max(e, w)));
  return clamp(color, minColor - vec3f(0.03), maxColor + vec3f(0.03));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let uv = clamp(input.uv, vec2f(0.0), vec2f(1.0));
  let texel = params.texel;
  let strength = clamp(params.sharpness, 0.0, 1.5);
  let modeGain = select(select(0.82, 0.55, params.mode >= 1.5), 1.05, params.mode > 0.5 && params.mode < 1.5);
  var color = edgeAdaptiveUpscale(uv, texel);
  color = contrastAdaptiveSharpen(color, uv, texel, strength * modeGain);
  color = antiRing(color, uv, texel);
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
