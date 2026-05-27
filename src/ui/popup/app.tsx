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
            <p className="popup-eyebrow">Current tab controls</p>
            <CardTitle>Video GPU Super Resolution</CardTitle>
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
          <section className="status-grid" aria-label="Status">
            <div className="status-item">
              <span className="status-item__label">Connection</span>
              <Badge className={statusTone === "warn" ? "ui-badge--warn" : undefined}>
                {model.connectionLabel}
              </Badge>
            </div>
            <div className="status-item">
              <span className="status-item__label">Video</span>
              <strong>{model.hasVideo ? "Detected" : "Not detected"}</strong>
            </div>
            <div className="status-item">
              <span className="status-item__label">Engine</span>
              <strong>{model.activeEngineLabel}</strong>
            </div>
          </section>

          <section className="control-section" aria-label="Core controls">
            <Field label="Engine">
              <Select
                aria-label="Engine"
                value={settings.engine}
                onChange={(event) =>
                  onSettingsChange({
                    engine: event.currentTarget.value as Settings["engine"],
                  })
                }
              >
                <option value="webgpu">WebGPU Super Resolution</option>
                <option value="tiny-cnn">Tiny CNN (WebGL)</option>
                <option value="ecbsr">ECBSR (ONNX/WebGPU)</option>
              </Select>
            </Field>

            {settings.engine === "ecbsr" ? (
              <Field label="Model">
                <Select
                  aria-label="Model"
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

            <Field label="Display mode">
              <Select
                aria-label="Display mode"
                value={settings.displayMode}
                onChange={(event) =>
                  onSettingsChange({
                    displayMode: event.currentTarget
                      .value as Settings["displayMode"],
                  })
                }
              >
                <option value="overlay">Overlay</option>
                <option value="replace">Replace source</option>
              </Select>
            </Field>

            <Field label="Scale">
              <Select
                aria-label="Scale"
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

          <section className="control-section" aria-label="Fine tuning">
            <RangeField
              label="Sharpness"
              value={settings.sharpness}
              min={0}
              max={1.4}
              step={0.05}
              onChange={(value) => onSettingsChange({ sharpness: value })}
            />
            <RangeField
              label="Overlay opacity"
              value={settings.overlayOpacity}
              min={0}
              max={1}
              step={0.05}
              onChange={(value) => onSettingsChange({ overlayOpacity: value })}
            />

            <Field label="Mode">
              <Select
                aria-label="Mode"
                value={settings.mode}
                onChange={(event) =>
                  onSettingsChange({
                    mode: event.currentTarget.value as Settings["mode"],
                  })
                }
              >
                <option value="balanced">Balanced</option>
                <option value="quality">Quality first</option>
                <option value="performance">Performance first</option>
              </Select>
            </Field>

            <Field label="Target FPS">
              <Select
                aria-label="Target FPS"
                value={settings.targetFps}
                onChange={(event) =>
                  onSettingsChange({
                    targetFps: event.currentTarget.value as Settings["targetFps"],
                  })
                }
              >
                <option value="auto">Auto</option>
                <option value="60">60 fps</option>
                <option value="30">30 fps</option>
                <option value="24">24 fps</option>
                <option value="15">15 fps</option>
              </Select>
            </Field>
          </section>

          <section className="action-row">
            <Button onClick={onRescan}>Rescan videos</Button>
            <Button variant="secondary" onClick={onOpenOptions}>
              Full settings
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
