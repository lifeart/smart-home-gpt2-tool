import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

// --- Base path -------------------------------------------------------------
// GitHub Pages serves this repo under a sub-path
// (https://lifeart.github.io/smart-home-gpt2-tool/) while the Hugging Face
// static Space and local dev serve from the root. The base is therefore
// configurable:
//   - dev / preview                : '/'              (default)
//   - GitHub Pages build           : '/smart-home-gpt2-tool/'
//   - HF Space build               : '/'
// Set it explicitly with the SITE_BASE env var, e.g.
//   SITE_BASE=/smart-home-gpt2-tool/ npm run build
const SITE_BASE = process.env.SITE_BASE || '/';

// --- Local model cache (dev only) -----------------------------------------
// `web/public/models/` is an ~8.7 GB local checkpoint cache (gitignored). It
// is a handy dev cache — transformers.js probes the local path first — but it
// must NOT be copied into the deployed `dist/`. Vite copies everything under
// `publicDir` verbatim, so for the build we point `publicDir` at a models-free
// directory (`public-build/`, populated below) and serve the real `public/`
// only during dev. The result: a few-MB `dist/` that fetches weights from the
// HF Hub at runtime, exactly like a fresh clone with no local cache.
const PUBLIC_DIR = path.resolve(__dirname, 'public');
const PUBLIC_BUILD_DIR = path.resolve(__dirname, 'public-build');

// Mirror `public/` into `public-build/` minus `models/`. Cheap: the only large
// thing under `public/` is `models/`; everything else (eval/ fixtures, the
// service worker, etc.) is a few hundred KB. Done with hard-link-free copies
// so it works on any FS.
function syncPublicBuildDir() {
  fs.rmSync(PUBLIC_BUILD_DIR, { recursive: true, force: true });
  fs.mkdirSync(PUBLIC_BUILD_DIR, { recursive: true });
  if (!fs.existsSync(PUBLIC_DIR)) return;
  for (const entry of fs.readdirSync(PUBLIC_DIR)) {
    if (entry === 'models') continue; // never ship the local checkpoint cache
    fs.cpSync(path.join(PUBLIC_DIR, entry), path.join(PUBLIC_BUILD_DIR, entry), {
      recursive: true,
    });
  }
}

// Without this, Vite's SPA fallback answers a missing `/models/...` request
// with index.html + HTTP 200. transformers.js then probes the local path,
// gets HTML, tries to JSON.parse it and crashes. With a real 404 it does the
// right thing on its own: try local, and if absent fall back to the HF Hub.
// Matches `/models/` under any base prefix.
function models404() {
  const make = (rootGetter) => (req, res, next) => {
    const url = (req.url || '').split('?')[0];
    const mi = url.indexOf('/models/');
    if (mi !== -1) {
      const rel = url.slice(mi); // strip any base prefix → '/models/...'
      const fp = path.join(rootGetter(), decodeURIComponent(rel));
      if (!fs.existsSync(fp)) {
        res.statusCode = 404;
        res.end('Not found');
        return;
      }
    }
    next();
  };
  return {
    name: 'models-404',
    configureServer(server) {
      server.middlewares.use(make(() => server.config.publicDir));
    },
    configurePreviewServer(server) {
      const outDir = path.resolve(server.config.root, server.config.build.outDir);
      server.middlewares.use(make(() => outDir));
    },
  };
}

export default defineConfig(({ command }) => {
  const isBuild = command === 'build';
  if (isBuild) syncPublicBuildDir();
  return {
    base: SITE_BASE,
    // Dev: serve the real public/ (so the local model cache is usable).
    // Build: serve the models-free mirror so dist/ stays small.
    publicDir: isBuild ? PUBLIC_BUILD_DIR : PUBLIC_DIR,
    plugins: [models404()],
    server: {
      headers: {
        // Required for SharedArrayBuffer (WASM multi-threading)
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      },
    },
    preview: {
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      },
    },
    optimizeDeps: {
      exclude: ['@huggingface/transformers'],
    },
  };
});
