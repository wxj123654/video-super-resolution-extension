const DEBUG_ENABLED = __VSR_DEBUG__;

type LogMethod = "debug" | "info" | "warn" | "error";

export function createLogger(scope: string) {
  return {
    enabled: DEBUG_ENABLED,
    debug(message: string, details?: unknown): void {
      emit("debug", scope, message, details);
    },
    info(message: string, details?: unknown): void {
      emit("info", scope, message, details);
    },
    warn(message: string, details?: unknown): void {
      emit("warn", scope, message, details);
    },
    error(message: string, details?: unknown): void {
      emit("error", scope, message, details);
    },
  };
}

export function toLogDetails(value: unknown): unknown {
  if (value instanceof Error) {
    const cause =
      "cause" in value
        ? toLogDetails((value as Error & { cause?: unknown }).cause)
        : undefined;
    return {
      name: value.name,
      message: value.message,
      stack: value.stack,
      cause,
    };
  }

  if (Array.isArray(value)) {
    return value.map((entry) => toLogDetails(entry));
  }

  if (value && typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(
      value as Record<string, unknown>,
    )) {
      result[key] = toLogDetails(entry);
    }
    return result;
  }

  return value;
}

function emit(
  method: LogMethod,
  scope: string,
  message: string,
  details?: unknown,
): void {
  if (!DEBUG_ENABLED) return;
  const text = `[VSR][${scope}] ${message}`;
  if (details === undefined) {
    console[method](text);
    return;
  }
  console[method](text, toLogDetails(details));
}
