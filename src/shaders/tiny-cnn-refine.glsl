#version 300 es
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
