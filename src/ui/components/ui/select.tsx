import type { SelectHTMLAttributes } from "react";

import { cn } from "@src/ui/lib/utils";

export function Select({
  className,
  children,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className="ui-select">
      <select className={cn("ui-select__control", className)} {...props}>
        {children}
      </select>
    </div>
  );
}
