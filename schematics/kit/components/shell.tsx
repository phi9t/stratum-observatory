import { useEffect, useState, type ReactNode } from "react";

import { SourceRef, type Ref } from "../lib/sources";

export type Lane = "high" | "low" | "instance" | "skip" | "merged";

export type SectionSpec = Readonly<{ id: string; label: string }>;

function useActiveSection(ids: readonly string[]): string {
  const [active, setActive] = useState(ids[0] ?? "");
  useEffect(() => {
    if (typeof IntersectionObserver !== "function") {
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActive(entry.target.id.replace(/^ms-/, ""));
          }
        }
      },
      { rootMargin: "-30% 0px -60% 0px" },
    );
    for (const id of ids) {
      const element = document.getElementById(`ms-${id}`);
      if (element !== null) {
        observer.observe(element);
      }
    }
    return () => observer.disconnect();
  }, [ids]);
  return active;
}

function SectionNav({ mark, sections }: { mark: string; sections: readonly SectionSpec[] }) {
  const active = useActiveSection(sections.map((section) => section.id));
  // Buttons, not anchors: the URL hash is reserved for playable state.
  const jump = (id: string): void => {
    const target = document.getElementById(`ms-${id}`);
    if (target === null || typeof target.scrollIntoView !== "function") {
      return;
    }
    const reduced =
      typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    target.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  };
  return (
    <nav aria-label="Schematic sections" className="ms-nav">
      <span className="ms-nav__mark">{mark}</span>
      <ul>
        {sections.map((section) => (
          <li key={section.id}>
            <button
              aria-current={section.id === active ? "true" : undefined}
              onClick={() => jump(section.id)}
              type="button"
            >
              {section.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/**
 * Page root. `spec` is the at-a-glance strip under the title: a handful of
 * value/label pairs taken from facts. `authority` names what the page is drawn from.
 */
export function Schematic({
  name,
  eyebrow,
  tagline,
  spec,
  authority,
  sections,
  children,
}: {
  name: string;
  eyebrow: string;
  tagline: string;
  spec: readonly Readonly<{ value: string; label: string }>[];
  authority: ReactNode;
  sections: readonly SectionSpec[];
  children: ReactNode;
}) {
  return (
    <div className="ms" data-standalone="true">
      <SectionNav mark={name} sections={sections} />
      <header className="ms-hero">
        <div className="ms-hero__eyebrow">{eyebrow}</div>
        <h1 className="ms-hero__title">{name}</h1>
        <div className="ms-hero__tagline">{tagline}</div>
        <ul aria-label="Model at a glance" className="ms-hero__spec">
          {spec.map((item) => (
            <li key={item.label}>
              <b>{item.value}</b>
              {item.label}
            </li>
          ))}
        </ul>
        <div className="ms-titleblock">{authority}</div>
      </header>
      {children}
    </div>
  );
}

/**
 * One section. `source` names the code being drawn and renders as the eyebrow;
 * pass one or more refs so the eyebrow links to the pinned definitions.
 */
export function Section({
  id,
  title,
  source,
  lede,
  children,
}: {
  id: string;
  title: string;
  source: readonly Ref[];
  lede: ReactNode;
  children: ReactNode;
}) {
  return (
    <section aria-labelledby={`ms-${id}-heading`} className="ms-section" id={`ms-${id}`}>
      <header className="ms-section__head">
        <div className="ms-section__source">
          {source.map((ref, index) => (
            <span key={`${ref.source}:${ref.symbol}`}>
              {index === 0 ? "" : " — "}
              <SourceRef {...ref} />
            </span>
          ))}
        </div>
        <h2 id={`ms-${id}-heading`}>{title}</h2>
        <div className="ms-section__lede">{lede}</div>
      </header>
      {children}
    </section>
  );
}

export function Sub({ children }: { children: ReactNode }) {
  return <h3 className="ms-sub">{children}</h3>;
}

export function Prose({ children }: { children: ReactNode }) {
  return <p className="ms-prose">{children}</p>;
}

/** Rounded white panel holding one fixed-size SVG; scrolls sideways when narrow. */
export function Figure({ label, caption, children }: { label: string; caption?: ReactNode; children: ReactNode }) {
  return (
    <figure className="ms-figure">
      <div
        aria-label={`${label}, scrolls horizontally on narrow screens`}
        className="ms-scroll"
        role="group"
        tabIndex={0}
      >
        {children}
      </div>
      {caption === undefined ? null : <figcaption>{caption}</figcaption>}
    </figure>
  );
}

/** A named tensor with its shape, e.g. `hidden_states [B, S, 7168]`. */
export function Tensor({ name, shape }: { name: string; shape: string }) {
  return (
    <span aria-label={`${name}: shape ${shape}`} className="ms-tensor">
      <code>{name}</code>
      <code className="ms-tensor__shape">{shape}</code>
    </span>
  );
}

export type PipeItem = Readonly<{ kind: "tensor" | "op"; text: string; sub?: string }>;
export const tensorItem = (text: string): PipeItem => ({ kind: "tensor", text });
export const opItem = (text: string, sub?: string): PipeItem => ({ kind: "op", text, sub });

/** Rows of a linear pipeline: tinted chips are tensors, outlined pills are operations. */
export function Pipes({ children }: { children: ReactNode }) {
  return <div className="ms-pipes">{children}</div>;
}

export function Pipe({ lane, label, items }: { lane: Lane; label: string; items: readonly PipeItem[] }) {
  return (
    <div className="ms-pipe" data-lane={lane}>
      <div className="ms-pipe__label">{label}</div>
      <ol aria-label={`${label} steps`}>
        {items.map((item, index) => (
          <li data-kind={item.kind} key={`${String(index)}:${item.text}`}>
            <span>{item.text}</span>
            {item.sub === undefined ? null : <small>{item.sub}</small>}
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Short hairline-separated facts: `<FactList items={[["Ingress", "…"], …]} />`. */
export function FactList({ items }: { items: readonly (readonly [string, ReactNode])[] }) {
  return (
    <ul className="ms-facts">
      {items.map(([title, body]) => (
        <li key={title}>
          <b>{title}.</b> {body}
        </li>
      ))}
    </ul>
  );
}

/** The closing note that states what the page does not cover. */
export function Boundary({ children }: { children: ReactNode }) {
  return <p className="ms-boundary">{children}</p>;
}

/** Hairline table. `dense` uses mono and tight rows for long numeric tables. */
export function Table({
  caption,
  head,
  rows,
  dense = false,
  current,
}: {
  caption: string;
  head: readonly string[];
  rows: readonly (readonly ReactNode[])[];
  dense?: boolean;
  /** Row index to mark as the current step. */
  current?: number;
}) {
  return (
    <div className="ms-table-scroll">
      <table className={dense ? "ms-table ms-table--dense" : "ms-table"}>
        <caption>{caption}</caption>
        <thead>
          <tr>
            {head.map((cell) => (
              <th key={cell} scope="col">
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr aria-current={index === current ? "step" : undefined} key={index}>
              {row.map((cell, column) =>
                column === 0 ? (
                  <th key={column} scope="row">
                    {cell}
                  </th>
                ) : (
                  <td key={column}>{cell}</td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Two to four sentences a reader should leave the section with. One per section. */
export function Takeaways({ items }: { items: readonly ReactNode[] }) {
  return (
    <aside aria-label="What to remember" className="ms-takeaways">
      <div className="ms-takeaways__label">What to remember</div>
      <ul>
        {items.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </aside>
  );
}

/** A self-check question with a hidden answer. Ask what the figure lets the reader work out. */
export function Check({ question, children }: { question: string; children: ReactNode }) {
  return (
    <details className="ms-check">
      <summary>
        <span className="ms-check__label">Check yourself</span>
        {question}
      </summary>
      <div className="ms-check__answer">{children}</div>
    </details>
  );
}

/** Terms the page introduces, each defined in one sentence, in order of first use. */
export function Glossary({ terms }: { terms: readonly (readonly [string, ReactNode])[] }) {
  return (
    <dl className="ms-glossary">
      {terms.map(([term, definition]) => (
        <div key={term}>
          <dt>{term}</dt>
          <dd>{definition}</dd>
        </div>
      ))}
    </dl>
  );
}
