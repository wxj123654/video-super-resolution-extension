(function attachTinyCnnShaders() {
  const ns = window.VideoGpuSuperResolutionInternal ||= {};

  const tinyCnnFeatureShader = `#version 300 es
    precision highp float;

    uniform sampler2D u_video;
    uniform vec2 u_textureSize;
    in vec2 v_uv;
    out vec4 outColor;

    float luma(vec3 color) { return dot(color, vec3(0.299, 0.587, 0.114)); }
    vec3 sampleVideo(vec2 uv) { return texture(u_video, clamp(uv, vec2(0.0), vec2(1.0))).rgb; }
    float encodeSigned(float value) { return clamp(value * 2.5 + 0.5, 0.0, 1.0); }

    void main() {
      vec2 texel = 1.0 / max(u_textureSize, vec2(1.0));
      vec3 c = sampleVideo(v_uv);
      vec3 n = sampleVideo(v_uv + vec2(0.0, -texel.y));
      vec3 s = sampleVideo(v_uv + vec2(0.0, texel.y));
      vec3 e = sampleVideo(v_uv + vec2(texel.x, 0.0));
      vec3 w = sampleVideo(v_uv + vec2(-texel.x, 0.0));
      vec3 ne = sampleVideo(v_uv + texel * vec2(1.0, -1.0));
      vec3 nw = sampleVideo(v_uv + texel * vec2(-1.0, -1.0));
      vec3 se = sampleVideo(v_uv + texel * vec2(1.0, 1.0));
      vec3 sw = sampleVideo(v_uv + texel * vec2(-1.0, 1.0));
      float lc = luma(c);
      float blur = luma(n + s + e + w + c * 4.0) * 0.125;
      float horizontal = luma(e) - luma(w);
      float vertical = luma(s) - luma(n);
      float diagonal = (luma(se) - luma(nw) + luma(sw) - luma(ne)) * 0.5;
      float edge = smoothstep(0.02, 0.18, max(length(vec2(horizontal, vertical)), abs(diagonal)));
      outColor = vec4(encodeSigned(lc - blur), encodeSigned(horizontal), encodeSigned(vertical), edge);
    }
  `;

  const tinyCnnRefineShader = `#version 300 es
    precision highp float;

    uniform sampler2D u_features;
    uniform vec2 u_targetSize;
    uniform int u_mode;
    in vec2 v_uv;
    out vec4 outColor;

    float decodeSigned(float value) { return (value - 0.5) / 2.5; }
    float encodeSigned(float value) { return clamp(value * 2.5 + 0.5, 0.0, 1.0); }
    vec4 sampleFeature(vec2 uv) { return texture(u_features, clamp(uv, vec2(0.0), vec2(1.0))); }

    void main() {
      vec2 texel = 1.0 / max(u_targetSize, vec2(1.0));
      vec4 c = sampleFeature(v_uv);
      vec4 n = sampleFeature(v_uv + vec2(0.0, -texel.y));
      vec4 s = sampleFeature(v_uv + vec2(0.0, texel.y));
      vec4 e = sampleFeature(v_uv + vec2(texel.x, 0.0));
      vec4 w = sampleFeature(v_uv + vec2(-texel.x, 0.0));
      vec4 ne = sampleFeature(v_uv + texel * vec2(1.0, -1.0));
      vec4 nw = sampleFeature(v_uv + texel * vec2(-1.0, -1.0));
      vec4 se = sampleFeature(v_uv + texel * vec2(1.0, 1.0));
      vec4 sw = sampleFeature(v_uv + texel * vec2(-1.0, 1.0));
      float detail = decodeSigned(c.r) * 1.65 - (decodeSigned(n.r) + decodeSigned(s.r) + decodeSigned(e.r) + decodeSigned(w.r)) * 0.14 - (decodeSigned(ne.r) + decodeSigned(nw.r) + decodeSigned(se.r) + decodeSigned(sw.r)) * 0.045;
      float h = decodeSigned(c.g) * 1.35 + (decodeSigned(e.g) + decodeSigned(w.g)) * 0.16 - (decodeSigned(n.g) + decodeSigned(s.g)) * 0.08;
      float v = decodeSigned(c.b) * 1.35 + (decodeSigned(n.b) + decodeSigned(s.b)) * 0.16 - (decodeSigned(e.b) + decodeSigned(w.b)) * 0.08;
      float qualityBoost = u_mode == 1 ? 1.25 : (u_mode == 2 ? 0.75 : 1.0);
      float edge = max(c.a, max(max(n.a, s.a), max(e.a, w.a)));
      outColor = vec4(encodeSigned(detail * qualityBoost), encodeSigned(h * qualityBoost), encodeSigned(v * qualityBoost), edge);
    }
  `;

  const tinyCnnReconstructShader = `#version 300 es
    precision highp float;

    uniform sampler2D u_video;
    uniform sampler2D u_features;
    uniform vec2 u_textureSize;
    uniform vec2 u_targetSize;
    uniform float u_sharpness;
    uniform int u_mode;
    in vec2 v_uv;
    out vec4 outColor;

    float luma(vec3 color) { return dot(color, vec3(0.299, 0.587, 0.114)); }
    float decodeSigned(float value) { return (value - 0.5) / 2.5; }
    vec3 sampleVideo(vec2 uv) { return texture(u_video, clamp(uv, vec2(0.0), vec2(1.0))).rgb; }

    vec3 catmullRom(vec2 uv) {
      vec2 texel = 1.0 / max(u_textureSize, vec2(1.0));
      vec2 xy = uv * u_textureSize - 0.5;
      vec2 base = floor(xy);
      vec2 f = xy - base;
      vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
      vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
      vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
      vec2 w3 = f * f * (-0.5 + 0.5 * f);
      vec3 color = vec3(0.0);
      for (int j = 0; j < 4; j++) {
        float wy = j == 0 ? w0.y : (j == 1 ? w1.y : (j == 2 ? w2.y : w3.y));
        for (int i = 0; i < 4; i++) {
          float wx = i == 0 ? w0.x : (i == 1 ? w1.x : (i == 2 ? w2.x : w3.x));
          color += sampleVideo((base + vec2(float(i - 1), float(j - 1)) + 0.5) * texel) * wx * wy;
        }
      }
      return color;
    }

    void main() {
      vec2 sourceTexel = 1.0 / max(u_textureSize, vec2(1.0));
      vec2 targetTexel = 1.0 / max(u_targetSize, vec2(1.0));
      vec3 source = sampleVideo(v_uv);
      vec3 base = u_mode == 2 ? source : catmullRom(v_uv);
      vec4 f = texture(u_features, clamp(v_uv, vec2(0.0), vec2(1.0)));
      vec4 fn = texture(u_features, clamp(v_uv + vec2(0.0, -targetTexel.y), vec2(0.0), vec2(1.0)));
      vec4 fs = texture(u_features, clamp(v_uv + vec2(0.0, targetTexel.y), vec2(0.0), vec2(1.0)));
      vec4 fe = texture(u_features, clamp(v_uv + vec2(targetTexel.x, 0.0), vec2(0.0), vec2(1.0)));
      vec4 fw = texture(u_features, clamp(v_uv + vec2(-targetTexel.x, 0.0), vec2(0.0), vec2(1.0)));
      float detail = decodeSigned(f.r);
      float h = decodeSigned(f.g);
      float v = decodeSigned(f.b);
      float localConsistency = 1.0 - min(1.0, abs(detail - decodeSigned(fn.r)) + abs(detail - decodeSigned(fs.r)) + abs(detail - decodeSigned(fe.r)) + abs(detail - decodeSigned(fw.r)));
      float modelGain = (u_mode == 1 ? 1.45 : (u_mode == 2 ? 0.7 : 1.05)) * clamp(u_sharpness, 0.0, 1.4);
      float lineDetail = (detail * (0.65 + f.a * 1.55) + (abs(h) + abs(v)) * sign(detail) * 0.18) * mix(0.55, 1.0, localConsistency);
      vec3 n = sampleVideo(v_uv + vec2(0.0, -sourceTexel.y));
      vec3 s = sampleVideo(v_uv + vec2(0.0, sourceTexel.y));
      vec3 e = sampleVideo(v_uv + vec2(sourceTexel.x, 0.0));
      vec3 w = sampleVideo(v_uv + vec2(-sourceTexel.x, 0.0));
      vec3 blur = (n + s + e + w + source * 4.0) * 0.125;
      vec3 chroma = base - vec3(luma(base));
      float enhancedLuma = luma(base) + lineDetail * modelGain + luma(base - blur) * clamp(u_sharpness, 0.0, 1.4) * 0.24;
      vec3 color = vec3(enhancedLuma) + chroma;
      vec3 minColor = min(source, min(min(n, s), min(e, w)));
      vec3 maxColor = max(source, max(max(n, s), max(e, w)));
      outColor = vec4(clamp(color, minColor - 0.045, maxColor + 0.045), 1.0);
    }
  `;

  Object.assign(ns, { tinyCnnFeatureShader, tinyCnnRefineShader, tinyCnnReconstructShader });
})();
