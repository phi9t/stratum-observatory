import { useId } from "react";
import { n, type Routing } from "../../../kit";
import * as f from "../facts";
const LAYOUT = { left: 24, pitch: 54, base: 170, full: 110, shared: 520, width: 720, height: 244 };

/** A small expert bank needs wider bars than the catalogue's large-bank widget.
 * Keep the kit's router tokens and selection attributes; never compress labels. */
export function ToyRouter({ routing }: { routing: Routing }) {
  const titleId = useId();
  const per = f.TOY.experts / f.TOY.groups;
  const peak = Math.max(...routing.scores);
  return (
    <svg
      className="ms-fig"
      role="img"
      aria-labelledby={titleId}
      width={LAYOUT.width}
      height={LAYOUT.height}
      viewBox={`0 0 ${LAYOUT.width} ${LAYOUT.height}`}
    >
      <title id={titleId}>Computed toy grouped router; original scores and selected experts</title>
      <text className="ms-fig__column" x={LAYOUT.left} y={18}>
        Router scores · {n(f.TOY.experts)} experts
      </text>
      {Array.from({ length: f.TOY.groups }, (_, g) => (
        <g key={g}>
          <text className="ms-fig__note" x={LAYOUT.left + g * per * LAYOUT.pitch} y={42}>
            group {n(g)}
            {routing.chosenGroups.includes(g) ? " · kept" : ""}
          </text>
          <rect
            className="ms-router__group"
            data-chosen={routing.chosenGroups.includes(g) ? "true" : undefined}
            x={LAYOUT.left + g * per * LAYOUT.pitch}
            y={LAYOUT.base - LAYOUT.full - 6}
            width={per * LAYOUT.pitch - 4}
            height={LAYOUT.full + 12}
          />
        </g>
      ))}
      {routing.scores.map((score, e) => {
        const height = (score / peak) * LAYOUT.full;
        const selected = routing.chosen.indexOf(e);
        return (
          <g key={e}>
            <rect
              className="ms-router__bar"
              data-chosen={selected >= 0 ? "true" : undefined}
              x={LAYOUT.left + e * LAYOUT.pitch + 8}
              y={LAYOUT.base - height}
              width={30}
              height={height}
            />
            <text className="ms-fig__note" x={LAYOUT.left + e * LAYOUT.pitch + 8} y={195}>
              e{n(e)}
            </text>
            <text className="ms-fig__note" x={LAYOUT.left + e * LAYOUT.pitch + 8} y={218}>
              {selected >= 0 ? routing.weights[selected].toFixed(f.DEMO.precision) : "—"}
            </text>
          </g>
        );
      })}
      <text className="ms-fig__column" x={LAYOUT.shared} y={18}>
        Shared · always on
      </text>
      <rect
        className="ms-router__bar"
        data-shared="true"
        x={LAYOUT.shared}
        y={LAYOUT.base - LAYOUT.full}
        width={30}
        height={LAYOUT.full}
      />
      <text className="ms-fig__note" x={LAYOUT.shared} y={195}>
        no router logit
      </text>
      <text className="ms-fig__note" x={LAYOUT.left} y={238}>
        Weights below selected experts; dash means not routed.
      </text>
    </svg>
  );
}
