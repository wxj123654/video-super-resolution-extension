struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct OutputParams {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
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

@group(0) @binding(0) var<storage, read> srcBuffer: array<f32>;
@group(0) @binding(1) var<uniform> params: OutputParams;

fn readChannel(channel: u32, x: u32, y: u32) -> f32 {
  let planeSize = params.width * params.height;
  let pixelIndex = y * params.width + x;
  return srcBuffer[channel * planeSize + pixelIndex];
}

@fragment
fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let maxX = max(params.width, 1u) - 1u;
  let maxY = max(params.height, 1u) - 1u;
  let x = min(u32(position.x), maxX);
  let y = min(u32(position.y), maxY);
  let color = vec3f(
    readChannel(0u, x, y),
    readChannel(1u, x, y),
    readChannel(2u, x, y)
  );
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
