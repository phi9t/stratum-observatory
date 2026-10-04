import { useCallback, useSyncExternalStore } from "react";

// All playable state lives in the URL hash (#layer=12&op=router), so any view
// can be linked and survives a reload. One shared store; each control owns a key.

const listeners = new Set<() => void>();
let installed = false;

function read(): URLSearchParams {
  if (typeof window === "undefined") {
    return new URLSearchParams();
  }
  return new URLSearchParams(window.location.hash.replace(/^#/, ""));
}

function install(): void {
  if (installed || typeof window === "undefined") {
    return;
  }
  installed = true;
  window.addEventListener("hashchange", () => listeners.forEach((listener) => listener()));
}

function subscribe(listener: () => void): () => void {
  install();
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function write(key: string, value: string | null): void {
  const params = read();
  if (value === null) {
    params.delete(key);
  } else {
    params.set(key, value);
  }
  const hash = params.toString();
  window.history.replaceState(null, "", hash === "" ? window.location.pathname + window.location.search : `#${hash}`);
  listeners.forEach((listener) => listener());
}

/** Integer state in the URL hash, clamped to [min, max]. */
export function useUrlNumber(
  key: string,
  fallback: number,
  min: number,
  max: number,
): readonly [number, (next: number | ((current: number) => number)) => void] {
  const raw = useSyncExternalStore(
    subscribe,
    () => read().get(key),
    () => null,
  );
  const clamp = (value: number): number =>
    Number.isFinite(value) ? Math.min(max, Math.max(min, Math.round(value))) : fallback;
  const value = raw === null ? fallback : clamp(Number(raw));
  const set = useCallback(
    (next: number | ((current: number) => number)) => {
      const stored = read().get(key);
      const current = stored === null ? fallback : clamp(Number(stored));
      const target = clamp(typeof next === "function" ? next(current) : next);
      write(key, target === fallback ? null : String(target));
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [key, fallback, min, max],
  );
  return [value, set];
}

/** String state in the URL hash, restricted to the allowed values. */
export function useUrlChoice<T extends string>(
  key: string,
  fallback: T,
  allowed: readonly T[],
): readonly [T, (next: T) => void] {
  const raw = useSyncExternalStore(
    subscribe,
    () => read().get(key),
    () => null,
  );
  const value = raw !== null && (allowed as readonly string[]).includes(raw) ? (raw as T) : fallback;
  const set = useCallback((next: T) => write(key, next === fallback ? null : next), [key, fallback]);
  return [value, set];
}
