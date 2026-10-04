import { useId, type ReactNode } from "react";

import type { Lane } from "./shell";

/** Native range input with a label and a mono value pill. */
export function Slider({
  label,
  value,
  min,
  max,
  onChange,
  display,
  valueText,
  lane,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (next: number) => void;
  /** Text in the value pill; defaults to the number. */
  display?: string;
  /** Screen-reader description of the current value. */
  valueText?: string;
  lane?: Lane;
}) {
  const id = useId();
  return (
    <div className="ms-control">
      <label htmlFor={id}>{label}</label>
      <input
        aria-valuetext={valueText ?? display ?? String(value)}
        id={id}
        max={max}
        min={min}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        step={1}
        type="range"
        value={value}
      />
      <span aria-hidden="true" className="ms-control__value" data-lane={lane}>
        {display ?? String(value)}
      </span>
    </div>
  );
}

/** Radio group drawn as pills. Use for 2–7 mutually exclusive views or instances. */
export function Segments<T extends string>({
  legend,
  options,
  value,
  onChange,
}: {
  legend: string;
  options: readonly Readonly<{ id: T; label: string }>[];
  value: T;
  onChange: (next: T) => void;
}) {
  const name = useId();
  return (
    <fieldset className="ms-segments">
      <legend>{legend}</legend>
      {options.map((option) => (
        <label key={option.id}>
          <input
            checked={option.id === value}
            name={name}
            onChange={() => onChange(option.id)}
            type="radio"
            value={option.id}
          />
          <span>{option.label}</span>
        </label>
      ))}
    </fieldset>
  );
}

/** Previous / scrub / Next. Pass functional updates so rapid presses do not collapse. */
export function Stepper({
  label,
  value,
  max,
  onChange,
  valueText,
  unit = "step",
}: {
  label: string;
  value: number;
  max: number;
  onChange: (next: number | ((current: number) => number)) => void;
  valueText: string;
  unit?: string;
}) {
  const id = useId();
  const pad = (v: number): string => String(v).padStart(String(max).length, "0");
  return (
    <div aria-label={`${label} stepper`} className="ms-stepper" role="group">
      <button disabled={value === 0} onClick={() => onChange((current) => current - 1)} type="button">
        Previous {unit}
      </button>
      <div className="ms-stepper__scrub">
        <label htmlFor={id}>{label}</label>
        <input
          aria-valuetext={valueText}
          id={id}
          max={max}
          min={0}
          onChange={(event) => onChange(Number(event.currentTarget.value))}
          step={1}
          type="range"
          value={value}
        />
        <span aria-hidden="true" className="ms-stepper__value">
          {pad(value)} / {pad(max)}
        </span>
      </div>
      <button disabled={value === max} onClick={() => onChange((current) => current + 1)} type="button">
        Next {unit}
      </button>
    </div>
  );
}

/** The serif sentence that restates the current state in words. One per interactive figure. */
export function Sentence({ children }: { children: ReactNode }) {
  return (
    <p aria-live="polite" className="ms-sentence">
      {children}
    </p>
  );
}

/** Row of exact values under a figure. `note` is the grey qualifier after the value. */
export function Readout({
  label,
  items,
}: {
  label: string;
  items: readonly Readonly<{ term: string; value: ReactNode; note?: ReactNode; text?: boolean }>[];
}) {
  return (
    <dl aria-label={label} className="ms-readout">
      {items.map((item) => (
        <div key={item.term}>
          <dt>{item.term}</dt>
          <dd className={item.text === true ? "ms-readout__text" : undefined}>
            {item.value}
            {item.note === undefined ? null : <span> {item.note}</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function Button({
  children,
  onClick,
  disabled,
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button className="ms-button" disabled={disabled} onClick={onClick} type="button">
      {children}
    </button>
  );
}
