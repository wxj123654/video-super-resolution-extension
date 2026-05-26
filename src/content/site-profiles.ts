import type { SiteProfile } from "../upscaler/types";

export const SITE_PROFILES: Record<string, SiteProfile> = {
  bilibili: {
    hostPattern: /(^|\.)bilibili\.com$/,
    canvasClass: "vgsr-site-bilibili",
    localOverlay: true,
    insertAfterVideo: true,
    hideSource: false,
    canvasOpacity: 0.8,
    videoSelectors: [
      ".bpx-player-video-wrap video",
      ".bilibili-player-video video",
      "#bilibili-player video",
      "video",
    ],
    overlayRootSelector:
      ".bpx-player-video-wrap, .bilibili-player-video, #bilibili-player",
  },
  default: {
    canvasClass: "",
    localOverlay: false,
    insertAfterVideo: false,
    hideSource: true,
    canvasOpacity: 1,
    videoSelectors: ["video"],
    overlayRootSelector: null,
  },
};

export function detectSiteProfile(): SiteProfile {
  const host = location.hostname;
  return (
    Object.values(SITE_PROFILES).find((profile) =>
      profile.hostPattern?.test(host),
    ) ?? SITE_PROFILES.default
  );
}
