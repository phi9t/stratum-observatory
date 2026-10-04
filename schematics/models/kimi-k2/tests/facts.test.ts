import { describe, expect, it } from "vitest";

import { verifyCoverage, verifyFacts, verifyParameters, verifyPins } from "../../../kit/lib/verify";
import * as model from "../facts";
import { sectionIds } from "../Page";

const SLUG = "kimi-k2";

describe(`${SLUG} against its pinned sources`, () => {
  it("pins every source to a commit and has it cached", () => {
    expect(verifyPins(SLUG)).toEqual([]);
  });

  it("states only facts the pinned config and code hold", () => {
    expect(verifyFacts(SLUG, model.facts)).toEqual([]);
  });

  it("maps every reference class to a section or excludes it with a reason", () => {
    expect(verifyCoverage(SLUG, sectionIds)).toEqual([]);
  });

  it("derives the parameter total the hub reports", () => {
    const note = (model as { parameterNote?: string }).parameterNote;
    const result = verifyParameters(SLUG, model.parameterTotal, note);
    expect(result.status, result.detail).not.toBe("fail");
  });
});
