// Headless Chrome over the DevTools protocol, plus a static file server.
// No dependencies: Node's built-in WebSocket and http.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, resolve, sep } from "node:path";

export const VIEWS = {
  "desktop-light": { width: 1440, height: 1000, scale: 1, mobile: false, scheme: "light" },
  "desktop-dark": { width: 1440, height: 1000, scale: 1, mobile: false, scheme: "dark" },
  "mobile-light": { width: 390, height: 844, scale: 2, mobile: true, scheme: "light" },
  "mobile-dark": { width: 390, height: 844, scale: 2, mobile: true, scheme: "dark" },
};

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

export const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

/** Path of the Chrome binary: `CHROME`, else the usual Linux locations; null if none exists. */
export function findChrome() {
  const candidates = process.env.CHROME
    ? [process.env.CHROME]
    : [
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
      ];
  return candidates.find((candidate) => existsSync(candidate)) ?? null;
}

/** Serves `directory` on 127.0.0.1 at a free port. */
export async function serveDirectory(directory) {
  const root = resolve(directory);
  const server = createServer((request, response) => {
    const path = decodeURIComponent(new URL(request.url ?? "/", "http://x").pathname);
    if (path === "/favicon.ico") {
      response.statusCode = 204;
      response.end();
      return;
    }
    const file = resolve(join(root, path));
    if (!(file === root || file.startsWith(root + sep)) || !existsSync(file) || !statSync(file).isFile()) {
      response.statusCode = 404;
      response.end();
      return;
    }
    response.setHeader("content-type", MIME[extname(file)] ?? "application/octet-stream");
    response.setHeader("cache-control", "no-store");
    response.end(readFileSync(file));
  });
  await new Promise((done) => server.listen(0, "127.0.0.1", done));
  return {
    base: `http://127.0.0.1:${server.address().port}/`,
    close: () => new Promise((done) => server.close(done)),
  };
}

/** Starts headless Chrome and opens one page. Throws if Chrome cannot start. */
export async function launchChrome(chromePath = findChrome()) {
  if (!chromePath) {
    throw new Error("Chrome not found (set CHROME to the binary)");
  }
  const profile = mkdtempSync(join(tmpdir(), "schematics-chrome-"));
  const chrome = spawn(
    chromePath,
    [
      "--headless=new",
      "--no-sandbox",
      "--disable-gpu",
      "--hide-scrollbars",
      "--no-first-run",
      "--remote-debugging-port=0",
      `--user-data-dir=${profile}`,
      "about:blank",
    ],
    { stdio: ["ignore", "ignore", "pipe"] },
  );
  const cleanup = () => {
    chrome.kill("SIGKILL");
    rmSync(profile, { recursive: true, force: true, maxRetries: 3 });
  };

  let wsUrl;
  try {
    wsUrl = await new Promise((done, reject) => {
      let stderr = "";
      const timer = setTimeout(() => reject(new Error("Chrome did not open a DevTools port in 20s")), 20_000);
      chrome.stderr.on("data", (chunk) => {
        stderr += chunk;
        const match = /DevTools listening on (ws:\/\/\S+)/.exec(stderr);
        if (match) {
          clearTimeout(timer);
          done(match[1]);
        }
      });
      chrome.on("error", (error) => reject(new Error(`cannot start ${chromePath}: ${error.message}`)));
      chrome.on("exit", (code) =>
        reject(new Error(`${chromePath} exited with code ${code}: ${stderr.trim().split("\n").at(-1) ?? ""}`)),
      );
    });
  } catch (error) {
    cleanup();
    throw error;
  }

  const socket = new WebSocket(wsUrl);
  await new Promise((done, reject) => {
    socket.addEventListener("open", done, { once: true });
    socket.addEventListener("error", () => reject(new Error("cannot connect to Chrome DevTools")), { once: true });
  });

  let nextId = 1;
  const pending = new Map();
  const listeners = new Set();
  const errors = [];
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id !== undefined) {
      pending.get(message.id)?.(message);
      pending.delete(message.id);
      return;
    }
    if (message.method === "Runtime.exceptionThrown") {
      const details = message.params.exceptionDetails;
      errors.push(details.exception?.description?.split("\n")[0] ?? details.text);
    } else if (message.method === "Log.entryAdded" && message.params.entry.level === "error") {
      errors.push(message.params.entry.text);
    }
    for (const listener of listeners) {
      listener(message);
    }
  });

  const send = (method, params = {}, sessionId) =>
    new Promise((done, reject) => {
      const id = nextId++;
      pending.set(id, (message) =>
        message.error ? reject(new Error(`${method}: ${message.error.message}`)) : done(message.result),
      );
      socket.send(JSON.stringify({ id, method, params, sessionId }));
    });

  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  await send("Runtime.enable", {}, sessionId);
  await send("Log.enable", {}, sessionId);
  await send("Page.enable", {}, sessionId);

  const waitFor = (method, timeoutMs = 20_000) =>
    new Promise((done, reject) => {
      const timer = setTimeout(() => {
        listeners.delete(listener);
        reject(new Error(`timed out waiting for ${method}`));
      }, timeoutMs);
      const listener = (message) => {
        if (message.method === method && message.sessionId === sessionId) {
          clearTimeout(timer);
          listeners.delete(listener);
          done(message.params);
        }
      };
      listeners.add(listener);
    });

  const page = {
    /** Page errors (uncaught exceptions and console errors) seen so far. */
    errors,
    send: (method, params = {}) => send(method, params, sessionId),
    /** Evaluates an expression in the page; promises are awaited; the value is returned by value. */
    async evaluate(expression) {
      const result = await send(
        "Runtime.evaluate",
        { expression, returnByValue: true, awaitPromise: true },
        sessionId,
      );
      if (result.exceptionDetails) {
        const details = result.exceptionDetails;
        throw new Error(`page script failed: ${details.exception?.description ?? details.text}`);
      }
      return result.result.value;
    },
    async setView(name) {
      const view = VIEWS[name];
      if (!view) {
        throw new Error(`unknown view "${name}" (known: ${Object.keys(VIEWS).join(", ")})`);
      }
      await send(
        "Emulation.setEmulatedMedia",
        {
          features: [
            { name: "prefers-color-scheme", value: view.scheme },
            { name: "prefers-reduced-motion", value: "reduce" },
          ],
        },
        sessionId,
      );
      await send(
        "Emulation.setDeviceMetricsOverride",
        { width: view.width, height: view.height, deviceScaleFactor: view.scale, mobile: view.mobile },
        sessionId,
      );
      return view;
    },
    /** Navigates and waits for the load event, fonts and two frames. */
    async open(url) {
      const loaded = waitFor("Page.loadEventFired");
      const result = await send("Page.navigate", { url }, sessionId);
      if (result.errorText) {
        loaded.catch(() => {});
        throw new Error(`cannot open ${url}: ${result.errorText}`);
      }
      await loaded;
      await page.settle();
    },
    async reload() {
      const loaded = waitFor("Page.loadEventFired");
      await send("Page.reload", { ignoreCache: true }, sessionId);
      await loaded;
      await page.settle();
    },
    async settle(ms = 350) {
      await page.evaluate(
        `(async () => { if (document.fonts && document.fonts.ready) { await document.fonts.ready; } await new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))); })()`,
      );
      await sleep(ms);
    },
    /** PNG of a document-coordinate rectangle. */
    async screenshot(clip) {
      const shot = await send(
        "Page.captureScreenshot",
        { format: "png", captureBeyondViewport: true, clip: { ...clip, scale: 1 } },
        sessionId,
      );
      return Buffer.from(shot.data, "base64");
    },
  };

  return {
    page,
    async close() {
      try {
        socket.close();
      } finally {
        cleanup();
      }
    },
  };
}
