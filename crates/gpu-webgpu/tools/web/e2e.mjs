// Serves the web_triangle build with COOP/COEP, loads it in headless Chromium with WebGPU
// enabled, waits for the in-wasm readback verdict, then screenshots the canvas and checks the
// probe pixels of the screenshot as well.
//
// usage: PW_DIR=<dir with node_modules/playwright> node e2e.mjs <dir with web_triangle.{js,wasm}> <screenshot.png>
import { createServer } from 'node:http';
import { readFileSync, existsSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const [buildDir, shotPath] = process.argv.slice(2);
const require = createRequire(join(process.env.PW_DIR ?? '.', 'noop.js'));
const { chromium } = require('playwright');
const here = dirname(fileURLToPath(import.meta.url));

const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const name = req.url.split('?')[0].replace(/^\/+/, '') || 'index.html';
  const file = name === 'index.html' ? join(here, name) : join(buildDir, name);
  if (name.includes('..') || !existsSync(file)) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, {
    'Content-Type': types[extname(file)] ?? 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cache-Control': 'no-cache',
  });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/`;

// No GPU (Chromium 131): Dawn's Vulkan backend on the SwiftShader ICD bundled with Chromium,
// in the full browser binary (`channel: 'chromium'`; headless_shell exposes no adapter).
// Headless mode renders the frames (the in-wasm readback passes) but never composites a
// worker-owned OffscreenCanvas, so screenshots hang; run.sh therefore uses a headed browser
// on Xvfb. E2E_HEADLESS=1 forces headless mode (screenshot check skipped).
const headless = !!process.env.E2E_HEADLESS;
const args = (process.env.CHROME_ARGS ??
  '--enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=swiftshader --use-webgpu-adapter=swiftshader')
  .split(/\s+/).filter(Boolean);
const browser = await chromium.launch({ channel: 'chromium', args, headless });
const page = await browser.newPage({ viewport: { width: 256, height: 256 } });
page.on('pageerror', (e) => console.log('PAGEERROR', e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('404')) console.log(`console.${m.type()}:`, m.text()); });
await page.goto(url);

let verdict = null;
const deadline = Date.now() + 90000;
while (Date.now() < deadline && !verdict) {
  verdict = await page.evaluate(() => window.__log.find((l) => l.startsWith('WEBGPU_E2E_RESULT')) ?? null);
  if (!verdict) await new Promise((r) => setTimeout(r, 250));
}
// A few more frames so the screenshot shows steady state.
await new Promise((r) => setTimeout(r, 500));
for (const l of await page.evaluate(() => window.__log)) console.log('page:', l);

let screenshotOk = headless;
if (!headless) {
  const png = await page.locator('#canvas').screenshot();
  writeFileSync(shotPath, png);
  // Decode the screenshot in the page and sample the same probes the wasm side checked.
  const probes = [[0.0625, 0.0625, [51, 102, 153]], [0.25, 0.25, [255, 0, 0]], [0.75, 0.25, [0, 255, 0]],
    [0.25, 0.75, [0, 0, 255]], [0.75, 0.75, [255, 255, 0]], [0.9375, 0.9375, [51, 102, 153]]];
  const sampled = await page.evaluate(async ({ b64, probes }) => {
    const img = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
    const g = new OffscreenCanvas(img.width, img.height).getContext('2d', { willReadFrequently: true });
    g.drawImage(img, 0, 0);
    return probes.map(([fx, fy]) => {
      const x = Math.floor(fx * img.width), y = Math.floor(fy * img.height);
      return [x, y, [...g.getImageData(x, y, 1, 1).data.slice(0, 3)]];
    });
  }, { b64: png.toString('base64'), probes });
  screenshotOk = true;
  sampled.forEach(([x, y, got], i) => {
    const want = probes[i][2];
    const ok = got.every((v, k) => Math.abs(v - want[k]) <= 3);
    screenshotOk &&= ok;
    console.log(`screenshot (${x},${y}) = ${got} want ${want} ${ok ? 'ok' : 'MISMATCH'}`);
  });
}
await browser.close();
server.close();

const wasmOk = verdict === 'WEBGPU_E2E_RESULT ok';
console.log(`wasm readback: ${verdict ?? 'no verdict (timeout)'}`);
console.log(headless ? 'screenshot: skipped (headless)' : `screenshot: ${screenshotOk ? 'ok' : 'FAIL'} (${shotPath})`);
process.exit(wasmOk && screenshotOk ? 0 : 1);
