export interface TileExtractParams {
  srcWidth: number;
  srcHeight: number;
  channels: number;
  tileX: number;
  tileY: number;
  tileW: number;
  tileH: number;
}

export interface TileCopyParams {
  dstWidth: number;
  dstHeight: number;
  channels: number;
  dstX: number;
  dstY: number;
  tileW: number;
  tileH: number;
}

export interface TileBlendParams {
  outWidth: number;
  outHeight: number;
  channels: number;
  tileCount: number;
  overlap: number;
}

export interface TileDescData {
  dstX: number;
  dstY: number;
  tileW: number;
  tileH: number;
}
