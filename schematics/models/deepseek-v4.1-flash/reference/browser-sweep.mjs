// Additional model-local verification; uses the same Chrome driver as check/shoot.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { launchChrome, serveDirectory } from "../../../tools/lib/chrome.mjs";
const artifact = "dist/deepseek-v4.1-flash.html";
const server = await serveDirectory("dist");
const browser = await launchChrome();
try {
  const { page } = browser;
  await page.setView("desktop-light");
  await page.open("about:blank");
  await page.open(`${server.base}deepseek-v4.1-flash.html`);
  const layers = await page.evaluate(`(async () => {
    const tick = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const count = document.querySelectorAll('.ms-layermap__tiles button').length;
    const initial = Number(document.querySelector('.ms-layermap__tiles button[aria-pressed="true"]').textContent);
    for (let layer = 0; layer < count; layer++) {
      document.querySelectorAll('.ms-layermap__tiles button')[layer].click();
      await tick();
      if (document.querySelectorAll('.ms-layermap__tiles button')[layer].getAttribute('aria-pressed') !== 'true') throw Error('selection '+layer);
      if (!document.querySelector('#ms-block').textContent.includes('Layer '+layer+':')) throw Error('block sentence '+layer);
      if (Number(new URLSearchParams(location.hash.slice(1)).get('layer') ?? initial) !== layer) throw Error('URL '+layer);
      if (document.querySelector('#ms-run').textContent.includes('NaN')) throw Error('nonfinite '+layer);
    }
    return count;
  })()`);
  await page.open("about:blank");
  await page.open(`${server.base}deepseek-v4.1-flash.html`);
  const operations = await page.evaluate(`(async () => {
    const tick = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const graphs = document.querySelectorAll('.ms-opgraph');
    let count = 0;
    for (let gi = 0; gi < graphs.length; gi++) {
      const total = graphs[gi].querySelectorAll('g[data-op]').length;
      for (let oi = 0; oi < total; oi++) {
        const graph = document.querySelectorAll('.ms-opgraph')[gi];
        const op = graph.querySelectorAll('g[data-op]')[oi];
        op.dispatchEvent(new MouseEvent('click',{bubbles:true}));
        await tick();
        const updated = document.querySelectorAll('.ms-opgraph')[gi];
        if (updated.querySelectorAll('g[data-op]')[oi].getAttribute('aria-pressed') !== 'true') throw Error('op selection '+gi+'/'+oi);
        const inspector = updated.querySelector('.ms-inspector');
        if (!inspector.querySelector('.ms-inspector__formula') || !inspector.querySelector('a')) throw Error('op inspector '+gi+'/'+oi);
        count++;
      }
    }
    return count;
  })()`);
  const ranges = await page.evaluate(`(async () => {
    const tick = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const count = document.querySelectorAll('input[type="range"]').length;
    for (let i = 0; i < count; i++) {
      const input = document.querySelectorAll('input[type="range"]')[i];
      const before = document.querySelector('.ms').outerHTML;
      const value = Number(input.value) === Number(input.max) ? input.min : input.max;
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, value);
      input.dispatchEvent(new Event('input',{bubbles:true}));
      input.dispatchEvent(new Event('change',{bubbles:true}));
      await tick();
      if (document.querySelector('.ms').outerHTML === before) throw Error('range unchanged '+i);
    }
    return count;
  })()`);
  const restoredState = `JSON.stringify({ranges:[...document.querySelectorAll('input[type="range"]')].map(x=>x.value), ops:[...document.querySelectorAll('g[data-op][aria-pressed="true"]')].map(x=>x.getAttribute('data-op')), layer:document.querySelector('.ms-layermap__tiles button[aria-pressed="true"]').textContent})`;
  const state = await page.evaluate(restoredState);
  const hash = await page.evaluate("location.hash");
  await page.open("about:blank");
  await page.open(`${server.base}deepseek-v4.1-flash.html${hash}`);
  assert.equal(await page.evaluate("location.hash"), hash);
  assert.equal(await page.evaluate(restoredState), state);
  assert.deepEqual(page.errors, []);
  console.log(JSON.stringify({ status: "pass", layers, operations, ranges, reload: hash, artifact, sha256: createHash("sha256").update(readFileSync(artifact)).digest("hex") }, null, 2));
} finally {
  await browser.close();
  await server.close();
}
