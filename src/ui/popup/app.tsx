import { useEffect, useState } from "react";

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
  type PopupModel,
  rescanPopup,
  updatePopupSettings,
} from "@src/ui/lib/popup-controller";
import type { Settings } from "@src/upscaler/types";

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
                <option value="ecbsr">ECBSR (ONNX/WebGPU)</option>
              </Select>
            </Field>

            {settings.engine === "ecbsr" ? (
              <Field label="模型">
                <Select
                  aria-label="模型"
                  value={settings.modelId}
                  onChange={(event) =>
                    onSettingsChange({ modelId: event.currentTarget.value })
                  }
                >
                  {ONNX_MODEL_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              </Field>
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

  return (
    <PopupApp
      model={model}
      onSettingsChange={(patch) => {
        void updatePopupSettings(tabId, patch).then((session) => {
          setTabId(session.tabId);
          setModel(session.model);
        });
      }}
      onRescan={() => {
        void rescanPopup(tabId).then((session) => {
          setTabId(session.tabId);
          setModel(session.model);
        });
      }}
      onOpenOptions={() => {
        void launchOptionsPage();
      }}
    />
  );
}
