import type { HTMLAttributes } from "react";

import { cn } from "@src/ui/lib/utils";

export function Card({ className, ...props }: HTMLAttributes<HTMLElement>) {
  return <section className={cn("ui-card", className)} {...props} />;
}

export function CardHeader({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("ui-card__header", className)} {...props} />;
}

export function CardTitle({
  className,
  ...props
}: HTMLAttributes<HTMLHeadingElement>) {
  return <h2 className={cn("ui-card__title", className)} {...props} />;
}

export function CardContent({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("ui-card__content", className)} {...props} />;
}
