import { useEffect, useState } from "react";

import { DEFAULT_SETTINGS, ONNX_MODEL_OPTIONS } from "@src/shared/extension/settings";
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
  loadOptions,
  type OptionsModel,
  persistOptionsSettings,
  runDiagnostics,
} from "@src/ui/lib/options-controller";
import type { Settings } from "@src/upscaler/types";

type OptionsAppProps = {
  model: OptionsModel;
  extensionVersion: string;
  onSettingsChange: (patch: Partial<Settings>) => void;
  onRunDiagnostics: () => void;
};

export function OptionsApp({
  model,
  extensionVersion,
  onSettingsChange,
  onRunDiagnostics,
}: OptionsAppProps) {
  const { settings } = model;

  return (
    <main className="mx-auto max-w-5xl p-8">
      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>常规设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Field label="启用增强">
                <Switch
                  aria-label="启用增强"
                  checked={settings.enabled}
                  onChange={(event) =>
                    onSettingsChange({
                      enabled: event.currentTarget.checked,
                    })
                  }
                />
              </Field>

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
                    onSettingsChange({
                      scale: Number(event.currentTarget.value),
                    })
                  }
                >
                  <option value="1">1x</option>
                  <option value="1.5">1.5x</option>
                  <option value="2">2x</option>
                </Select>
              </Field>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>精细调整</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
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
                onChange={(value) =>
                  onSettingsChange({ overlayOpacity: value })
                }
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
                  <option value="quality">质量优先</option>
                  <option value="performance">性能优先</option>
                </Select>
              </Field>

              <Field label="目标帧率">
                <Select
                  aria-label="目标帧率"
                  value={settings.targetFps}
                  onChange={(event) =>
                    onSettingsChange({
                      targetFps: event.currentTarget
                        .value as Settings["targetFps"],
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
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>诊断</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button onClick={onRunDiagnostics}>运行诊断</Button>
              <pre className="max-h-96 overflow-auto rounded-lg bg-[hsl(var(--muted))] p-4 text-xs">
                {model.diagnosticsText}
              </pre>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>关于</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-[hsl(var(--muted-foreground))]">
              <p>Video GPU Super Resolution</p>
              <p>版本: {extensionVersion}</p>
              <p>
                实时 WebGL/WebGPU 视频超分辨率扩展，为 HTML5
                视频提供画质增强。
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  );
}

export function OptionsRoot() {
  const [model, setModel] = useState<OptionsModel>({
    settings: DEFAULT_SETTINGS,
    diagnosticsText: "正在加载...",
  });

  useEffect(() => {
    loadOptions()
      .then(setModel)
      .catch((err: unknown) => {
        setModel({
          settings: DEFAULT_SETTINGS,
          diagnosticsText: "加载失败",
        });
        console.error("[VSR] Failed to load options:", err);
      });
  }, []);

  return (
    <OptionsApp
      model={model}
      extensionVersion={chrome.runtime.getManifest().version}
      onSettingsChange={(patch) => {
        persistOptionsSettings(patch)
          .then(() => loadOptions().then(setModel))
          .catch((err: unknown) => {
            console.error("[VSR] Failed to save settings:", err);
          });
      }}
      onRunDiagnostics={() => {
        runDiagnostics()
          .then((diagnosticsText) => {
            setModel((prev) => ({ ...prev, diagnosticsText }));
          })
          .catch((err: unknown) => {
            setModel((prev) => ({
              ...prev,
              diagnosticsText: `诊断失败: ${err instanceof Error ? err.message : String(err)}`,
            }));
          });
      }}
    />
  );
}
