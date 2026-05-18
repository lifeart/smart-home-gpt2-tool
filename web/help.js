// Per-model and per-toggle help copy for the demo UI.
//
// Numbers are pulled straight from PLAN.md so the page matches the latest
// shipped bench. If you re-bench, update the numbers here too.

export const MODEL_CARDS = {
  'local:smart-home-gpt2': {
    title: 'v1 SFT (local) — first browser-shippable checkpoint',
    body: [
      'Trained on 1200 smart-home items (from-scratch HF Trainer SFT on GPT-2 124M).',
      'Multi-tool acc (n=300): 55.7%. Voice E2E (RU→Whisper EN, n=30): 46.7% on curated cand-sets.',
      'On-disk: ~1.7 GB (fp32 622 MB + fp16 311 MB + q8 380 MB ONNX shards).',
    ],
  },
  'local:smart-home-gpt2-v2': {
    title: 'v2 SFT (local) — data-expansion checkpoint',
    body: [
      'Trained on 8651 items: 5990 SH×5 upsample + 2510 xLAM irrelevance + 151 Hermes IoT.',
      'Multi-tool acc (n=300): 80.3% (+24.6 pp vs v1). Light domain 25.0 → 65.6%.',
      'On-disk: ~1.7 GB if downloaded. Not bundled by default — falls back to HF Hub.',
    ],
  },
  'local:smart-home-gpt2-v3': {
    title: 'v3 SFT (local) — current best ship config',
    body: [
      'Trained on 12134 items: 5990 SH×5 + 2.5k irrelevance + 151 IoT + 3493 Whisper-noisy aug.',
      'Multi-tool n=300: name 82.33% / args 54.67% / exact 49.67% baseline; con+typed+wide ON: 83.0 / 58.0 / 55.0.',
      'Voice E2E (n=30, ret_con K=5 + alias): name 53.3% / exact 23.3% / recall@5 93.3%. On-disk ~1.7 GB.',
    ],
  },
  'local:smart-home-gpt2-v4': {
    title: 'v4 SFT (local) — twin-confusion + numeric coverage [Iter 11, may not be shipped yet]',
    body: [
      'Trained on 18532 items: v3 mix + 2953 xLAM + 369 Hermes + 131 twin-pair + 940 numeric.',
      'Target: lift `light` domain (twin polarity) and numeric-arg correctness. Iter 11 running.',
      'On-disk: ~1.7 GB once exported. Falls back to v3 if not downloaded.',
    ],
  },
  'lifeart/smart-home-gpt2': {
    title: 'v1 SFT (HF Hub) — streamed download',
    body: [
      'Same weights as `local:smart-home-gpt2`, fetched from huggingface.co/lifeart/smart-home-gpt2.',
      'First load: ~622 MB fp32 over the wire (or ~311 MB fp16 / ~380 MB q8 if selected). Cached by the browser.',
      'Use this if you do not have the local copy at /web/public/models/.',
    ],
  },
  'lifeart/smart-home-gpt2-v2': {
    title: 'v2 SFT (HF Hub) — data-expansion checkpoint',
    body: [
      'Same weights as v2 local. Multi-tool 80.3%. Streams from huggingface.co/lifeart/smart-home-gpt2-v2.',
      '~622 MB fp32 / ~311 MB fp16 / ~380 MB q8 ONNX over the wire on first load.',
    ],
  },
  'lifeart/smart-home-gpt2-v3': {
    title: 'v3 SFT (HF Hub) — current best',
    body: [
      'Same weights as v3 local. Multi-tool con+typed+wide: 83.0 / 58.0 / 55.0 (n=300).',
      'Streams from huggingface.co/lifeart/smart-home-gpt2-v3. ~622 MB fp32 / 311 MB fp16 / 380 MB q8.',
    ],
  },
  'Xenova/gpt2': {
    title: 'GPT-2 base (Xenova) — no fine-tuning',
    body: [
      'The 124M pre-trained GPT-2 from openai-community/gpt2, repackaged for transformers.js.',
      'Has no smart-home tool-calling priors — useful as an ablation baseline (expect 0% exact).',
      '~500 MB fp32 over the wire on first load.',
    ],
  },
};

// One-line + when-to-toggle + n=300 bench effect, per toggle.
// Pulled from PLAN.md Iter 9.3 / 10.3 best-ship-config tables.
export const TOGGLE_HELP = {
  constrained: {
    label: 'Constrained decoding (JSON schema)',
    what:
      'At every generation step, mask logits to only tokens compatible with a JSON-schema FSM built from the candidate list. Refuses any token that would derail valid {"name":..., "arguments":...} output.',
    when:
      'ON: weaker models (v1/v2) or any time you need 100% JSON-valid output. OFF: v3+ where SFT already gives 99.7% JSON valid and the schema masks empty-args edge cases (clean domain has irrelevance gold).',
    effect:
      'n=300 v3: baseline 82.33 / 54.67 / 49.67 → con+typed+wide 83.0 / 58.0 / 55.0 (+0.67 name, +3.33 args, +5.33 exact). Overhead ~1.3 ms/step.',
  },
  typedargs: {
    label: 'Typed args (enum / numeric / boolean)',
    what:
      'Within constrained mode, enforce per-arg-key types: enum values get masked to allowed strings, numeric keys disallow quotes, booleans only emit true/false.',
    when:
      'ON: pair with constrained. Cleans the wrongStr=37 / wrongNum=9 failure classes from Iter 7.1. OFF: only useful to A/B against pure name-masking.',
    effect:
      'n=50 con: +2 pp args, +2 pp exact vs typedArgs OFF, no negative flips. (Iter 7.2.)',
  },
  widenames: {
    label: 'Wide names (registry ∪ prompt)',
    what:
      'In constrained mode, widen the name-token allowlist from prompt-extracted candidates to (prompt ∪ all 123 registry names). Recovers gold names truncated out of the ~4 kB SYSTEM tool list at dataset prep time.',
    when:
      'ON (default since Iter 9.3): the only way to beat baseline on all 3 metrics. OFF: only if you trust the prompt cand list to be complete.',
    effect:
      'n=300 v3 con typedArgs ON: exact 50.67 → 55.00 (+4.33), name 77.67 → 83.00 (+5.33). Targeted: 19 regression items 0/19 → 19/19 name-correct.',
  },
  retrieval: {
    label: 'Retrieval pre-rank (MiniLM, top-K)',
    what:
      'At query time, encode the USER command with `Xenova/all-MiniLM-L6-v2` and cosine-rank against a 123-vector index of (name + description + example queries). Top-K replaces the prompt`s candidate list.',
    when:
      'ON: voice / curated cand-set demos — recovers +23.3 pp on voice (n=30). OFF: multi-tool n=300 (test set ships the right candidates already; retrieval ties or loses).',
    effect:
      'Voice n=30 v3: baseline 23.3% → ret_con 46.7% (+23.34 pp), recall@3 70%. Multi-tool n=300: ~neutral to -2 pp because misc domain (irrelevance gold) gets forced into a function call.',
  },
  topk: {
    label: 'K (retrieval top-K)',
    what:
      'How many candidates retrieval keeps. Lower K = stricter mask; higher K = better recall but more confusable peers.',
    when:
      'K=3 for multi-tool (matches Phase 4 default). K=5 for voice / ASR-noisy queries (Iter 6.1: voice acc 46.7 → 50.0%, recall 70 → 83.3%).',
    effect:
      'Voice n=30 v3 ret_con: K=3 acc 46.67% / recall 70%; K=5 acc 50.00% / recall 83.3%. With alias expansion K=5: 53.33% / 93.3%.',
  },
};

// Below-bench legend explaining each printed metric.
// Format: [key, description, good-range].
export const BENCH_LEGEND = [
  ['prompt tokens', 'BPE tokens fed in. Typical SH prompt is 60-90 tokens; over 1024 wraps GPT-2`s ctx.', 'good: 60-300'],
  ['new tokens',    'Tokens generated this run. Capped by the "max new tokens" input. A valid functioncall is usually 20-60.', 'good: 20-60'],
  ['time-to-first', 'Latency from generate-start to first decoded token.', 'good: <1 s WebGPU · 3-8 s WASM'],
  ['total',         'End-to-end wall time including first-token + streaming + finalization.', ''],
  ['throughput',    'tokens/sec after first token. WebGPU/fp32 on M-series ~160 tok/s; q4 ~200 tok/s.', 'good: >100 · bad: <30 (WASM fallback)'],
  ['backend',       'Selected device + dtype. WebGPU/fp32 is the bench reference; WASM/q8 is the compatibility floor.', ''],
  ['retrieval',     '"OFF" or "top-K = [...]" with cosine score and "orig-cands-in-topK" recall sanity.', 'recall@3 = 89.67% on n=300 v3'],
  ['constrained',   '"OFF" or "N candidates (a, b, c…)" — which names the grammar admits this step.', ''],
  ['grammar steps', 'FSM steps taken + per-step overhead. Spikes mean many branches explored.', 'good: ~1-2 ms/step on M-series'],
];

// Footer copy.
export const FOOTER = {
  blurb:
    'Browser demo of GPT-2 124M fine-tuned for smart-home tool-calling. Runs fully in-tab via WebGPU + transformers.js — no server inference.',
  links: [
    { label: 'GitHub repo',     href: 'https://github.com/lifeart/smart-home-gpt2-tool' },
    { label: 'HF · v3 (best)',  href: 'https://huggingface.co/lifeart/smart-home-gpt2-v3' },
    { label: 'HF · v2',         href: 'https://huggingface.co/lifeart/smart-home-gpt2-v2' },
    { label: 'HF · v1',         href: 'https://huggingface.co/lifeart/smart-home-gpt2' },
    { label: 'HF · v4 (Iter 11, may 404)', href: 'https://huggingface.co/lifeart/smart-home-gpt2-v4' },
  ],
};
