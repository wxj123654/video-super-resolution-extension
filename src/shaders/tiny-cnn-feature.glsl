#version 300 es
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
