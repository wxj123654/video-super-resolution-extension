import type { InputHTMLAttributes } from "react";

import { cn } from "@src/ui/lib/utils";

type SwitchProps = Omit<InputHTMLAttributes<HTMLInputElement>, "type"> & {
  label?: string;
};

export function Switch({ className, label, ...props }: SwitchProps) {
  return (
    <label className={cn("ui-switch", className)}>
      <input className="ui-switch__input" type="checkbox" {...props} />
      <span className="ui-switch__track" aria-hidden="true">
        <span className="ui-switch__thumb" />
      </span>
      {label ? <span className="ui-switch__text">{label}</span> : null}
    </label>
  );
}
