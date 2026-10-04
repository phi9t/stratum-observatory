// Definition extents of a source file: the one place that decides where a class
// or function starts and ends. `tools/fetch-sources.mjs` writes the result to
// models/<slug>/symbols.json and `verify.ts` checks `code` facts inside the same
// lines. Keep this file free of imports and of non-erasable TypeScript: the
// tools import it directly through Node's type stripping.

/** symbol -> [firstLine, lastLine], 1-based and inclusive. */
export type FileSymbols = Record<string, [number, number]>;

type Definition = { name: string; indent: number; first: number; last: number; key: string };

const DEFINITION =
  /^([ \t]*)(?:(?:export|default|async|static|inline|public|private|protected|abstract|final)\s+)*(?:class|def|function\*?|struct)\s+([A-Za-z_$][\w$]*)/;

/** Sources that hold definitions; config, papers and notes are not indexed. */
export function isCodePath(path: string): boolean {
  return !/\.(json|pdf|md)$/i.test(path);
}

/**
 * Every `class`/`def`/`function` at any indentation. A definition ends just
 * before the next definition at the same or lower indentation (and before that
 * definition's decorators), with trailing blank lines trimmed. Nested definitions are keyed `Outer.inner`, and also by
 * their bare name when that name is defined only once in the file.
 */
export function indexSymbols(text: string): FileSymbols {
  const lines = text.split(/\r?\n/);
  const definitions: Definition[] = [];
  lines.forEach((line, index) => {
    const match = DEFINITION.exec(line);
    if (match) {
      definitions.push({ name: match[2], indent: match[1].length, first: index + 1, last: 0, key: "" });
    }
  });

  const stack: Definition[] = [];
  definitions.forEach((definition, position) => {
    const next = definitions.find(
      (other, otherPosition) => otherPosition > position && other.indent <= definition.indent,
    );
    let last = next ? next.first - 1 : lines.length;
    if (next) {
      // Decorators directly above the next definition belong to it, not to this one.
      const marker = `${" ".repeat(next.indent)}@`;
      let cut = last;
      for (let line = last; line > definition.first && lines[line - 1].trim() !== ""; line -= 1) {
        if (lines[line - 1].replace(/\t/g, " ").startsWith(marker)) {
          cut = line - 1;
        }
      }
      last = cut;
    }
    while (last > definition.first && lines[last - 1].trim() === "") {
      last -= 1;
    }
    definition.last = last;
    while (stack.length > 0 && stack[stack.length - 1].indent >= definition.indent) {
      stack.pop();
    }
    definition.key = [...stack.map((outer) => outer.name), definition.name].join(".");
    stack.push(definition);
  });

  const symbols: FileSymbols = {};
  for (const definition of definitions) {
    if (!(definition.key in symbols)) {
      symbols[definition.key] = [definition.first, definition.last];
    }
  }
  const uses = new Map<string, number>();
  for (const definition of definitions) {
    uses.set(definition.name, (uses.get(definition.name) ?? 0) + 1);
  }
  for (const definition of definitions) {
    if (uses.get(definition.name) === 1 && !(definition.name in symbols)) {
      symbols[definition.name] = [definition.first, definition.last];
    }
  }
  return symbols;
}

/** The text of lines `first`..`last` (1-based, inclusive). */
export function sliceLines(text: string, range: readonly [number, number]): string {
  return text
    .split(/\r?\n/)
    .slice(range[0] - 1, range[1])
    .join("\n");
}

/** Keys sorted at both levels, so symbols.json is stable. */
export function sortedIndex(index: Record<string, FileSymbols>): Record<string, FileSymbols> {
  const sorted: Record<string, FileSymbols> = {};
  for (const id of Object.keys(index).sort()) {
    const symbols: FileSymbols = {};
    for (const name of Object.keys(index[id]).sort()) {
      symbols[name] = index[id][name];
    }
    sorted[id] = symbols;
  }
  return sorted;
}
