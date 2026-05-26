#version 300 es
    precision highp float;

    uniform sampler2D u_video;
    uniform sampler2D u_luma;
    uniform float u_mix;
    in vec2 v_uv;
    out vec4 outColor;

    float luma(vec3 color) {
      return dot(color, vec3(0.299, 0.587, 0.114));
    }

    void main() {
      vec3 base = texture(u_video, clamp(v_uv, vec2(0.0), vec2(1.0))).rgb;
      float baseY = luma(base);
      float modelY = clamp(texture(u_luma, clamp(v_uv, vec2(0.0), vec2(1.0))).r, 0.0, 1.0);
      float y = mix(baseY, modelY, clamp(u_mix, 0.0, 1.0));
      float gain = y / max(baseY, 0.0001);
      vec3 color = base * clamp(gain, 0.55, 1.8);

      vec3 minColor = max(base - vec3(0.12), vec3(0.0));
      vec3 maxColor = min(base + vec3(0.12), vec3(1.0));
      outColor = vec4(clamp(color, minColor, maxColor), 1.0);
    }
