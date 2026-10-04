// Every number a page shows is a Fact: read from the pinned config, read from
// pinned source code, or derived from other facts. Nothing is typed into a
// section. `kit/lib/verify.ts` checks cfg and code facts against the cache.

export type Locator = Readonly<{
  /** Source id from sources.json. */
  source: string;
  /** Class or function name that must be defined in that file. */
  symbol: string;
  /** Short literal that must appear inside that definition. */
  contains?: string;
}>;

export type Fact<T = unknown> = Readonly<{
  kind: "cfg" | "code" | "derived";
  value: T;
  /** Dotted path into config.json, for cfg facts. */
  path?: string;
  locator?: Locator;
  deps?: readonly Fact[];
}>;

export function cfg<T>(path: string, value: T): Fact<T> {
  return { kind: "cfg", value, path };
}

export function code<T>(value: T, locator: Locator): Fact<T> {
  return { kind: "code", value, locator };
}

export function derive<T>(fn: () => T, deps: readonly Fact[]): Fact<T> {
  if (deps.length === 0) {
    throw new Error("derive: a derived fact must name the facts it is computed from");
  }
  return { kind: "derived", value: fn(), deps };
}

/** Reader-facing number: plain digits, never thousands separators. */
export function n(value: number): string {
  if (!Number.isFinite(value)) {
    throw new RangeError(`n: not a finite number: ${String(value)}`);
  }
  return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(4)));
}

/** Compact magnitude for prose and bars: 1.41e9 -> "1.41B". Exact value stays in tables. */
export function approx(value: number): string {
  const units: readonly (readonly [number, string])[] = [
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "K"],
  ];
  for (const [size, unit] of units) {
    if (Math.abs(value) >= size) {
      return `${String(Number((value / size).toPrecision(3)))}${unit}`;
    }
  }
  return n(value);
}

/** Tensor shape string: shape("B", "S", 7168) -> "[B, S, 7168]". */
export function shape(...dims: readonly (string | number)[]): string {
  return `[${dims.map((dim) => (typeof dim === "number" ? n(dim) : dim)).join(", ")}]`;
}
