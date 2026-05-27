import type { ButtonHTMLAttributes } from "react";

import { cn } from "@src/ui/lib/utils";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
};

export function Button({
  className,
  variant = "primary",
  type = "button",
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cn("ui-button", `ui-button--${variant}`, className)}
      {...props}
    />
  );
}
