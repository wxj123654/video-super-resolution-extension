struct BlendParams {
  outWidth: u32,
  outHeight: u32,
  tileCount: u32,
  overlap: u32,
};

struct TileDesc {
  dstX: u32,
  dstY: u32,
  tileW: u32,
  tileH: u32,
};

@group(0) @binding(0) var<storage, read_write> dstBuffer: array<f32>;
@group(0) @binding(1) var<storage, read> tilesBuffer: array<f32>;
@group(0) @binding(2) var<storage, read> tileDescs: array<TileDesc>;
@group(0) @binding(3) var<uniform> params: BlendParams;

fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
  let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(8, 8, 1)
fn computeMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.outWidth || gid.y >= params.outHeight) {
    return;
  }

  let channels = 3u;
  let outPlaneSize = params.outWidth * params.outHeight;
  let outIdx = gid.y * params.outWidth + gid.x;

  var accumR: f32 = 0.0;
  var accumG: f32 = 0.0;
  var accumB: f32 = 0.0;
  var totalWeight: f32 = 0.0;

  for (var t = 0u; t < params.tileCount; t++) {
    let desc = tileDescs[t];
    let localX = i32(gid.x) - i32(desc.dstX);
    let localY = i32(gid.y) - i32(desc.dstY);

    if (localX < 0 || localX >= i32(desc.tileW) || localY < 0 || localY >= i32(desc.tileH)) {
      continue;
    }

    let overlap = params.overlap;
    var wx: f32 = 1.0;
    if (u32(localX) < overlap) {
      wx = smoothstep_f32(0.0, f32(overlap), f32(localX));
    }
    let rightDist = i32(desc.tileW) - 1 - localX;
    if (u32(rightDist) < overlap) {
      wx = wx * smoothstep_f32(0.0, f32(overlap), f32(rightDist));
    }

    var wy: f32 = 1.0;
    if (u32(localY) < overlap) {
      wy = smoothstep_f32(0.0, f32(overlap), f32(localY));
    }
    let bottomDist = i32(desc.tileH) - 1 - localY;
    if (u32(bottomDist) < overlap) {
      wy = wy * smoothstep_f32(0.0, f32(overlap), f32(bottomDist));
    }

    let weight = wx * wy;
    let tilePlaneSize = desc.tileW * desc.tileH;
    let tileIdx = u32(localY) * desc.tileW + u32(localX);

    let tileBase = t * tilePlaneSize * channels;
    accumR += tilesBuffer[tileBase + tileIdx] * weight;
    accumG += tilesBuffer[tileBase + tilePlaneSize + tileIdx] * weight;
    accumB += tilesBuffer[tileBase + tilePlaneSize * 2u + tileIdx] * weight;
    totalWeight += weight;
  }

  if (totalWeight > 0.0) {
    let inv = 1.0 / totalWeight;
    dstBuffer[outIdx] = accumR * inv;
    dstBuffer[outPlaneSize + outIdx] = accumG * inv;
    dstBuffer[outPlaneSize * 2u + outIdx] = accumB * inv;
  }
}
