import { n, type Routing } from "../../kit";
const LAYOUT = { width: 600, height: 230, left: 28, base: 152, bar: 38, pitch: 74, full: 104, shared: 390 };
export function ToyRouter({ routing }: { routing: Routing }) {
  const L = LAYOUT;
  return (
    <svg
      className="ms-fig"
      width={L.width}
      height={L.height}
      viewBox={`0 0 ${L.width} ${L.height}`}
      role="img"
      aria-label="Computed toy expert scores and independent shared expert"
    >
      <text className="ms-fig__column" x={L.left} y={22}>
        routed · sigmoid scores
      </text>
      {routing.scores.map((score, i) => {
        const height = score * L.full;
        return (
          <g key={i}>
            <rect
              className="ms-router__bar"
              data-chosen={routing.chosen.includes(i) ? "true" : undefined}
              x={L.left + i * L.pitch}
              y={L.base - height}
              width={L.bar}
              height={height}
            />
            <text className="ms-node__line" x={L.left + i * L.pitch} y={L.base + 22}>
              {n(i)}
            </text>
            <text className="ms-node__line" x={L.left + i * L.pitch} y={L.base + 42}>
              {score.toFixed(3)}
            </text>
          </g>
        );
      })}
      <text className="ms-fig__column" x={L.shared} y={22}>
        shared · always on
      </text>
      <rect
        className="ms-router__bar"
        data-shared="true"
        x={L.shared}
        y={L.base - L.full}
        width={L.bar}
        height={L.full}
      />
      <text className="ms-fig__note" x={L.shared} y={L.base + 22}>
        no router logit
      </text>
      <text className="ms-fig__note" x={L.left} y={L.height - 10}>
        Selected bars are teal; score height uses a fixed unit scale.
      </text>
    </svg>
  );
}
