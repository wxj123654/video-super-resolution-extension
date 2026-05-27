import type { OnnxModelDefinition } from "../upscaler/types";

const CACHE_NAME = "vsr-models";

function cacheKey(modelId: string): string {
  return `/vsr-models/${modelId}`;
}

export async function getCachedModel(
  modelId: string,
): Promise<ArrayBuffer | null> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(cacheKey(modelId));
    if (!response) return null;
    return response.arrayBuffer();
  } catch {
    return null;
  }
}

export async function cacheModel(
  modelId: string,
  data: ArrayBuffer,
): Promise<void> {
  const cache = await caches.open(CACHE_NAME);
  const response = new Response(data, {
    headers: { "Content-Type": "application/octet-stream" },
  });
  await cache.put(cacheKey(modelId), response);
}

export async function isModelCached(modelId: string): Promise<boolean> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(cacheKey(modelId));
    return response !== undefined;
  } catch {
    return false;
  }
}

export async function removeCachedModel(modelId: string): Promise<void> {
  try {
    const cache = await caches.open(CACHE_NAME);
    await cache.delete(cacheKey(modelId));
  } catch {
    // ignore
  }
}

export async function resolveModelBuffer(
  model: OnnxModelDefinition,
  onProgress?: (loaded: number, total: number) => void,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  // Bundled models: load from extension assets
  if (!model.source || model.source.type === "bundled") {
    const url = chrome.runtime.getURL(model.modelPath);
    const response = await fetch(url);
    return response.arrayBuffer();
  }

  // Downloadable models: check cache first
  const cached = await getCachedModel(model.id);
  if (cached) return cached;

  // Download from CDN
  if (!model.source.downloadUrl) {
    throw new Error(`Model ${model.id} has no download URL`);
  }

  const response = await fetch(model.source.downloadUrl, { signal });
  if (!response.ok) {
    throw new Error(`Failed to download model: ${response.status} ${response.statusText}`);
  }

  const contentLength = Number(response.headers.get("Content-Length")) || 0;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    await cacheModel(model.id, buffer);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, contentLength || loaded);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  const result = buffer.buffer as ArrayBuffer;
  await cacheModel(model.id, result);
  return result;
}
