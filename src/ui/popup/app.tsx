import React, { useEffect, useState } from "react";

import { ONNX_MODEL_OPTIONS } from "@src/shared/extension/settings";
import { Badge } from "@src/ui/components/ui/badge";
import { Button } from "@src/ui/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@src/ui/components/ui/card";
import { Field, RangeField } from "@src/ui/components/ui/field";
import { Select } from "@src/ui/components/ui/select";
import { Switch } from "@src/ui/components/ui/switch";
import {
  bootstrapPopup,
  INITIAL_POPUP_MODEL,
  launchOptionsPage,
  type ModelDownloadStatus,
  type PopupModel,
  rescanPopup,
  updatePopupSettings,
} from "@src/ui/lib/popup-controller";
import type { Settings } from "@src/upscaler/types";

const CATEGORY_LABELS: Record<string, string> = {
  lightweight: "轻量级",
  balanced: "平衡",
  quality: "质量优先",
};

function formatFileSize(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
}

function DownloadProgress({ status }: { status: ModelDownloadStatus }) {
  if (status.state === "idle") return null;

  if (status.state === "downloading") {
    const percent = status.total > 0 ? Math.round((status.progress / status.total) * 100) : 0;
    const loadedMB = (status.progress / 1024 / 1024).toFixed(1);
    const totalMB = status.total > 0 ? (status.total / 1024 / 1024).toFixed(1) : "?";
    return (
      <div className="download-progress">
        <span className="download-progress__text">
          正在下载模型... {loadedMB}/{totalMB} MB ({percent}%)
        </span>
        <progress className="download-progress__bar" max={100} value={percent} />
      </div>
    );
  }

  if (status.state === "error") {
    return (
      <div className="download-progress download-progress--error">
        <span className="download-progress__text">下载失败: {status.error}</span>
      </div>
    );
  }

  return null;
}

type PopupAppProps = {
  model: PopupModel;
  onSettingsChange: (patch: Partial<Settings>) => void;
  onRescan: () => void;
  onOpenOptions: () => void;
};

export function PopupApp({
  model,
  onSettingsChange,
  onRescan,
  onOpenOptions,
}: PopupAppProps) {
  const { settings } = model;
  const statusTone = model.connectionLabel === "Connected" ? "ok" : "warn";

  return (
    <main className="popup-shell">
      <Card>
        <CardHeader className="popup-header">
          <div>
            <p className="popup-eyebrow">当前标签页控制</p>
            <CardTitle>视频 GPU 超分辨率</CardTitle>
            <p className="popup-subtitle">{model.statusText}</p>
          </div>
          <Switch
            aria-label="Enable enhancement"
            checked={settings.enabled}
            onChange={(event) =>
              onSettingsChange({ enabled: event.currentTarget.checked })
            }
          />
        </CardHeader>

        <CardContent className="popup-stack">
          <section className="status-grid" aria-label="状态">
            <div className="status-item">
              <span className="status-item__label">连接</span>
              <Badge className={statusTone === "warn" ? "ui-badge--warn" : undefined}>
                {model.connectionLabel}
              </Badge>
            </div>
            <div className="status-item">
              <span className="status-item__label">视频</span>
              <strong>{model.hasVideo ? "已检测" : "未检测"}</strong>
            </div>
            <div className="status-item">
              <span className="status-item__label">引擎</span>
              <strong>{model.activeEngineLabel}</strong>
            </div>
          </section>

          <section className="control-section" aria-label="核心控制">
            <Field label="引擎">
              <Select
                aria-label="引擎"
                value={settings.engine}
                onChange={(event) =>
                  onSettingsChange({
                    engine: event.currentTarget.value as Settings["engine"],
                  })
                }
              >
                <option value="webgpu">WebGPU 超分辨率</option>
                <option value="tiny-cnn">Tiny CNN (WebGL)</option>
                <option value="onnx">神经网络超分 (ONNX/WebGPU)</option>
              </Select>
            </Field>

            {settings.engine === "onnx" ? (
              <>
                <Field label="模型">
                  <Select
                    aria-label="模型"
                    value={settings.modelId}
                    onChange={(event) =>
                      onSettingsChange({ modelId: event.currentTarget.value })
                    }
                  >
                    {(() => {
                      const groups = new Map<string, typeof ONNX_MODEL_OPTIONS[number][]>();
                      for (const opt of ONNX_MODEL_OPTIONS) {
                        const cat = opt.category ?? "lightweight";
                        if (!groups.has(cat)) groups.set(cat, []);
                        groups.get(cat)!.push(opt);
                      }
                      const elements: React.ReactElement[] = [];
                      for (const [category, items] of groups) {
                        const groupElements = items.map((opt) => {
                          const sizeTag = opt.source?.fileSize
                            ? ` (${formatFileSize(opt.source.fileSize)})`
                            : "";
                          return (
                            <option key={opt.id} value={opt.id}>
                              {opt.label}{sizeTag}
                            </option>
                          );
                        });
                        if (groups.size > 1) {
                          elements.push(
                            <optgroup key={category} label={CATEGORY_LABELS[category] ?? category}>
                              {groupElements}
                            </optgroup>
                          );
                        } else {
                          elements.push(...groupElements);
                        }
                      }
                      return elements;
                    })()}
                  </Select>
                </Field>
                <DownloadProgress status={model.modelDownload} />
              </>
            ) : null}

            <Field label="显示模式">
              <Select
                aria-label="显示模式"
                value={settings.displayMode}
                onChange={(event) =>
                  onSettingsChange({
                    displayMode: event.currentTarget
                      .value as Settings["displayMode"],
                  })
                }
              >
                <option value="overlay">叠加</option>
                <option value="replace">替换源</option>
              </Select>
            </Field>

            <Field label="缩放倍数">
              <Select
                aria-label="缩放倍数"
                value={String(settings.scale)}
                onChange={(event) =>
                  onSettingsChange({ scale: Number(event.currentTarget.value) })
                }
              >
                <option value="1">1x</option>
                <option value="1.5">1.5x</option>
                <option value="2">2x</option>
              </Select>
            </Field>
          </section>

          <section className="control-section" aria-label="精细调整">
            <RangeField
              label="锐度"
              value={settings.sharpness}
              min={0}
              max={1.4}
              step={0.05}
              onChange={(value) => onSettingsChange({ sharpness: value })}
            />
            <RangeField
              label="叠加透明度"
              value={settings.overlayOpacity}
              min={0}
              max={1}
              step={0.05}
              onChange={(value) => onSettingsChange({ overlayOpacity: value })}
            />

            <Field label="模式">
              <Select
                aria-label="模式"
                value={settings.mode}
                onChange={(event) =>
                  onSettingsChange({
                    mode: event.currentTarget.value as Settings["mode"],
                  })
                }
              >
                <option value="balanced">平衡</option>
                <option value="quality">Quality first</option>
                <option value="performance">Performance first</option>
              </Select>
            </Field>

            <Field label="目标帧率">
              <Select
                aria-label="目标帧率"
                value={settings.targetFps}
                onChange={(event) =>
                  onSettingsChange({
                    targetFps: event.currentTarget.value as Settings["targetFps"],
                  })
                }
              >
                <option value="auto">自动</option>
                <option value="60">60 fps</option>
                <option value="30">30 fps</option>
                <option value="24">24 fps</option>
                <option value="15">15 fps</option>
              </Select>
            </Field>
          </section>

          <section className="action-row">
            <Button onClick={onRescan}>重新扫描视频</Button>
            <Button variant="secondary" onClick={onOpenOptions}>
              完整设置
            </Button>
          </section>
        </CardContent>
      </Card>
    </main>
  );
}

export function PopupRoot() {
  const [tabId, setTabId] = useState<number | null>(null);
  const [model, setModel] = useState<PopupModel>(INITIAL_POPUP_MODEL);

  useEffect(() => {
    void bootstrapPopup().then((session) => {
      setTabId(session.tabId);
      setModel(session.model);
    });
  }, []);

  useEffect(() => {
    const listener = (message: { type: string; modelStatus?: { state: string; progress?: number; total?: number; error?: string } }) => {
      if (message.type === "VSR_MODEL_STATUS" && message.modelStatus) {
        const { state, progress, total, error } = message.modelStatus;
        if (state === "downloading" && progress !== undefined && total !== undefined) {
          setModel((prev) => ({ ...prev, modelDownload: { state: "downloading", progress, total } }));
        } else if (state === "ready") {
          setModel((prev) => ({ ...prev, modelDownload: { state: "ready" } }));
        } else if (state === "error") {
          setModel((prev) => ({ ...prev, modelDownload: { state: "error", error: error ?? "未知错误" } }));
        }
      }
    };
    chrome.runtime.onMessage.addListener(listener as Parameters<typeof chrome.runtime.onMessage.addListener>[0]);
    return () => {
      chrome.runtime.onMessage.removeListener(listener as Parameters<typeof chrome.runtime.onMessage.removeListener>[0]);
    };
  }, []);

  return (
    <PopupApp
      model={model}
      onSettingsChange={(patch) => {
        void updatePopupSettings(tabId, patch).then((session) => {
          setTabId(session.tabId);
          setModel((prev) => ({ ...session.model, modelDownload: prev.modelDownload }));
        });
      }}
      onRescan={() => {
        void rescanPopup(tabId).then((session) => {
          setTabId(session.tabId);
          setModel((prev) => ({ ...session.model, modelDownload: prev.modelDownload }));
        });
      }}
      onOpenOptions={() => {
        void launchOptionsPage();
      }}
    />
  );
}
