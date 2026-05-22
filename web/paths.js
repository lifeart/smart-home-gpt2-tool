// Base-path-aware URL helpers.
//
// The demo is deployed under three different base paths:
//   - local dev / preview          → '/'
//   - GitHub Pages                 → '/smart-home-gpt2-tool/'
//   - Hugging Face static Space    → '/'
//
// Vite injects the build-time base into `import.meta.env.BASE_URL` (always
// ends with a trailing slash). Every reference to a same-origin asset
// (`/eval/...` fixtures, the `/models/` local checkpoint cache, the
// coi-serviceworker) must go through these helpers so it resolves correctly
// under whichever base the bundle was built with.

// `import.meta.env.BASE_URL` always ends with '/'.
export const BASE_URL = import.meta.env.BASE_URL || '/';

/**
 * Resolve an app-relative path against the deployment base.
 * @param {string} p  e.g. 'eval/tool_registry.json' or '/eval/tool_registry.json'
 * @returns {string}  e.g. '/smart-home-gpt2-tool/eval/tool_registry.json'
 */
export function asset(p) {
  return BASE_URL + String(p).replace(/^\/+/, '');
}

// transformers.js `env.localModelPath` — where it probes for a local model
// cache before falling back to the HF Hub. Must include the base prefix.
export const LOCAL_MODEL_PATH = asset('models/');
