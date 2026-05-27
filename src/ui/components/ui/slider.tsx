import type { InputHTMLAttributes } from "react";

import { cn } from "@src/ui/lib/utils";

type SliderProps = Omit<InputHTMLAttributes<HTMLInputElement>, "type">;

export function Slider({ className, ...props }: SliderProps) {
  return (
    <input
      type="range"
      className={cn("ui-slider", className)}
      {...props}
    />
  );
}
