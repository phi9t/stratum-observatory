import { useId } from "react";
import { n } from "../../kit";
import * as f from "./facts";
import { rotaryFrequency } from "./reference/ops";
// Layout constants are pixels, never checkpoint quantities.
const L = { left: 24, pitch: 48, bar: 32, base: 160, full: 110, sharedX: 470, width: 700, height: 220 };
export function ToyRouter({ scores, chosen, weights }: { scores: number[]; chosen: number[]; weights: number[] }) {
  const title = useId();
  const peak = Math.max(...scores);
  return (
    <svg
      className="ms-fig"
      role="img"
      aria-labelledby={title}
      width={L.width}
      height={L.height}
      viewBox={`0 0 ${L.width} ${L.height}`}
    >
      <title id={title}>Computed toy router scores and independent shared expert</title>
      <text className="ms-fig__column" x={L.left} y={18}>
        toy scores · {n(scores.length)} routed experts
      </text>
      {scores.map((score, i) => (
        <g key={i}>
          <rect
            className="ms-router__bar"
            data-chosen={chosen.includes(i) ? "true" : undefined}
            x={L.left + i * L.pitch}
            y={L.base - (score / peak) * L.full}
            width={L.bar}
            height={(score / peak) * L.full}
          />
          <text className="ms-fig__note" x={L.left + i * L.pitch + L.bar / 2} y={L.base + 20} textAnchor="middle">
            {n(i)}
          </text>
        </g>
      ))}
      <text className="ms-fig__note" x={L.left} y={L.base + 44}>
        weights: {weights.map((v) => n(v)).join(" · ")}
      </text>
      <text className="ms-fig__column" x={L.sharedX} y={18}>
        shared · always on
      </text>
      <rect
        className="ms-router__bar"
        data-shared="true"
        x={L.sharedX}
        y={L.base - L.full}
        width={L.bar}
        height={L.full}
      />
      <text className="ms-fig__note" x={L.sharedX} y={L.base + 20}>
        no router logit
      </text>
    </svg>
  );
}
const D = { r: 34, pitch: 156, cx: 60, cy: 64, width: 520, height: 150 };
export function RotaryDials({ position, compressed }: { position: number; compressed: boolean }) {
  const title = useId();
  const pairs = [0, 1, f.ropeDim.value / 2 - 1];
  return (
    <svg
      className="ms-fig ms-dial"
      role="img"
      aria-labelledby={title}
      width={D.width}
      height={D.height}
      viewBox={`0 0 ${D.width} ${D.height}`}
    >
      <title id={title}>{compressed ? "Compressed-layer YaRN" : "Window-only base"} tail rotations</title>
      {pairs.map((pair, i) => {
        const frequency = rotaryFrequency(
          pair,
          f.ropeDim.value,
          compressed ? f.compressedBase.value : f.ropeBase.value,
          compressed ? f.original.value : 0,
          f.factor.value,
          f.fast.value,
          f.slow.value,
        );
        const angle = position * frequency,
          cx = D.cx + i * D.pitch;
        return (
          <g key={pair}>
            <circle cx={cx} cy={D.cy} r={D.r} />
            <line data-ghost="true" x1={cx} x2={cx + D.r} y1={D.cy} y2={D.cy} />
            <line x1={cx} x2={cx + D.r * Math.cos(angle)} y1={D.cy} y2={D.cy - D.r * Math.sin(angle)} />
            <text className="ms-fig__note" textAnchor="middle" x={cx} y={D.cy + D.r + 20}>
              pair {n(pair)}
            </text>
            <text className="ms-node__line" textAnchor="middle" x={cx} y={D.cy + D.r + 36}>
              {n(angle % (2 * Math.PI))} rad
            </text>
          </g>
        );
      })}
    </svg>
  );
}
