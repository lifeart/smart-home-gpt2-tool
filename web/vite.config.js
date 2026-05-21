import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

// Without this, Vite's SPA fallback answers a missing `/models/...` request
// with index.html + HTTP 200. transformers.js then probes the local path,
// gets HTML, tries to JSON.parse it and crashes. With a real 404 it does the
// right thing on its own: try local, and if absent fall back to the HF Hub.
function models404() {
  const make = (rootGetter) => (req, res, next) => {
    const url = (req.url || '').split('?')[0];
    if (url.startsWith('/models/')) {
      const fp = path.join(rootGetter(), decodeURIComponent(url));
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

export default defineConfig({
  plugins: [models404()],
  server: {
    headers: {
      // Required for SharedArrayBuffer (WASM multi-threading)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
});
