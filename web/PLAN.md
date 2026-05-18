
| 2026-05-18 | **Iter 13 done — synthetic SFT v5.** 3000 items generated via HF Inference (Llama-3.3-70B-Instruct, 99.9% accept) targeting weak buckets: 1000 3+keys, 800 numeric, 600 twin-confusion (clean/media/sec), 600 generic. v5 dataset 20010 items (v4 + dedup synthetic). Train T4 ~135 min, train_loss 0.39 (== v4). HF bench name n=300: v1 55.7 / v2 80.3 / v3 82.3 / v4 83.3 / **v5 84.3 (+1.0)**. Per-domain v5 vs v4: clean 79.3→86.2 (+6.9 recovered), misc +1.4, light -3.1 (only regression). **Browser 4-mode n=300 (v5 webgpu/fp32, typed+wide):** baseline 84.33/57.67/53.00, **con 84.67/60.33/57.33 — NEW BEST exact**, ret 73.67/36.33/30.67, ret_con 74.33/50.33/46.33. Δ con-exact vs v4 = +2.0 pp. Synthetic helped ret_con args dramatically: +13.33 pp. **Voice n=30 ret_con K=5 alias+typed+wide: name 60.00%, args 43.33%, exact 36.67%, recall@5 80%. Δ vs v4: +10/+13.33/+13.33 pp.** **Voice gate ≥55% name finally MET** (was missed since Phase 2). Cumulative v1→v5 name: 55.7→84.67 = +28.97 pp. Cost: ~$1.20 train + $0 inference (Groq free tier via HF router) = $1.20. (Plus $3.60 wasted on 3 duplicate train jobs — agent watchdog respawn issue, manager cancelled.) |

## Final ship config (Iter 13 winner)

| Track | Model | Toggles | Numbers |
|---|---|---|---|
| **Multi-tool** | `lifeart/smart-home-gpt2-v5` (WebGPU/fp32) | constrained + typed-args + schema-union + wide-names ON, retrieval **OFF** | **n=300: name 84.67% / args 60.33% / exact 57.33%** |
| **Voice E2E** | same `v5` | + retrieval K=5 + sentinel + alias expansion ON | **n=30: name 60.00% / args 43.33% / exact 36.67% / recall@5 80%** |

Cumulative v1→v5: **+28.97 pp name** (55.7 → 84.67), **+27 pp est exact** (~30 → 57.33). Voice gate met (60% ≥ 55%). Hypothesis ≥60% exact still missed (57.33%, 2.67 pp short). Total project spend ~$9 of $12.

