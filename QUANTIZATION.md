# Quantization Plan for smart-home-gpt2 (GPT-2 124M, CPU)

**Target model:** `weights/gpt2_ft_final.pt` — full fine-tune of HF GPT-2 124M, FP32 state dict, ~475 MB on disk.
**Current latency:** 25–35 s per command on a 4-core x86 CPU (single user, batch=1, greedy ~30–60 new tokens).
**Goal:** ship faster inference for an English-text → JSON tool-call task without giving up the 71.7% text accuracy / 92% in-domain accuracy reported in the repo.

All numbers below are sourced from published benchmarks (HF, ONNX Runtime, llama.cpp, dev.to GPT-2 INT8 article). Where a number is a projection rather than a measurement, it is marked **(estimated)**.

---

## 1. Available quantization approaches for GPT-2 on CPU

### 1.1 `torch.quantization` dynamic INT8 (per-layer)
**Theory.** Replace `nn.Linear` modules with `DynamicQuantizedLinear`: INT8 weights, activations quantized on-the-fly per call. No calibration data needed. Size ~63% smaller (475 → ~180 MB). Speed 1.2–1.5× on x86 with AVX2 (dev.to GPT-2 study: +22% throughput; ONNX docs cite 1.5–3× for transformers). Quality typically <1 pp delta; for tool-call with short outputs expect <2 pp. Ships with PyTorch. **Difficulty 1/5** — one function call.

### 1.2 bitsandbytes 8-bit / 4-bit NF4
**Theory.** LLM.int8() mixed-precision (8-bit weights + FP16 outliers) or NF4 (4-bit normal-float, double-quant). Built for GPU memory pressure on big models. Size ~240 MB (8-bit) / ~120 MB (NF4). **Speed on CPU: negative or zero** — bitsandbytes kernels are CUDA-first, CPU path falls back to slow dequant-then-matmul. **Difficulty 3/5** — works, but wrong tool for CPU.

### 1.3 GPTQ (4-bit, calibration-based)
**Theory.** Per-layer second-order (Hessian-based) update picks INT4 codes minimizing reconstruction error on ~128 calibration sequences. Size ~75 MB **(estimated)**. **Speed on CPU: marginal** — GPTQ kernels target CUDA/Triton; `auto-gptq` CPU mode dequants to FP16, so latency tracks FP16. Quality degrades *more* on small models (70B Q4 fine, 7B Q4 shakier, 124M Q4 risky). **Difficulty 4/5.**

### 1.4 AWQ (4-bit, activation-aware)
**Theory.** Identify the ~1% of weight channels with largest activation magnitude, pre-scale them, then RTN-quantize to INT4. Size ~75 MB **(estimated)**. Same CPU caveat as GPTQ — kernels GPU-first, CPU fallback unimpressive. Better than GPTQ at the same bit-width on larger models; empirical curve at 124M unknown. **Difficulty 4/5.**

### 1.5 GGUF via llama.cpp (Q8_0 / Q5_K_M / Q4_K_M)
**Theory.** Block-wise quantization (groups of 32 weights with per-group scale/zero) packaged in GGUF. Inference is pure C++/SIMD with hand-tuned AVX2/AVX-512/NEON kernels. Sizes: Q8_0 ≈ 130 MB, Q5_K_M ≈ 90 MB, Q4_K_M ≈ 75 MB. **Speed on CPU: the big win** — llama.cpp's CPU kernels are ~3–6× faster than `torch` FP32 for transformer decode on the same box. GPT-2 is supported by upstream `convert_hf_to_gguf.py`. Q4_K_M is the published sweet spot (~4.8 bits/weight); expect 1–4 pp drop on 124M, Q5_K_M is safer. **Difficulty 3/5** — conversion + Python binding wiring, well-trodden.

### 1.6 ONNX Runtime INT8 (dynamic)
**Theory.** Export to ONNX, then `quantize_dynamic` weights to INT8. ORT picks the best CPU kernel at runtime (oneDNN, MLAS, VNNI if present). Size ~180 MB. **Speed: up to 6× on VNNI CPUs**, 1.5–3× on plain AVX2 (MS Azure / HF blog numbers for transformer dynamic INT8). Quality <1 pp drop. **Difficulty 2/5.**

### 1.7 `torch.compile` + AOT Inductor (not quantization)
**Theory.** TorchInductor fuses ops and emits optimized C++/OpenMP kernels; AOT Inductor produces a standalone `.so`. Size unchanged. Speed 1.3–1.7× on CPU **(estimated)** — wins come from removing per-token Python overhead. No quality loss. PyTorch 2.4 built-in. **Difficulty 2/5**, but stacks poorly with HF `generate()` KV-cache loops.

### 1.8 Pruning + quantization combo
**Theory.** Magnitude- or movement-prune ~30–50% of weights to zero, then INT8 quantize. On dense INT8 the size stays the same; sparse CPU kernels for transformers are immature, expect no speedup on commodity x86. Quality unpredictable on a 1500-example fine-tune — pruning eats the very signal you trained. **Difficulty 5/5**, worst ROI here.

---

## 2. Real numbers

| approach | size (MB) | speed-up vs FP32 CPU | accuracy delta | tooling complexity |
|---|---|---|---|---|
| `torch.quantize_dynamic` INT8 | ~180 | 1.2–1.5× | <1 pp | 1/5 |
| bitsandbytes 8-bit | ~240 | ~1.0× (CPU) | ~0 | 3/5 |
| bitsandbytes NF4 | ~120 | <1.0× (CPU) | 1–3 pp **(est.)** | 3/5 |
| GPTQ 4-bit | ~75 **(est.)** | ~1.0× on CPU | 2–6 pp **(est., risky on 124M)** | 4/5 |
| AWQ 4-bit | ~75 **(est.)** | ~1.0× on CPU | 1–4 pp **(est.)** | 4/5 |
| **GGUF Q8_0 (llama.cpp)** | ~130 | **3–4×** **(est.)** | <1 pp | 3/5 |
| **GGUF Q5_K_M** | ~90 | **4–5×** **(est.)** | 1–2 pp | 3/5 |
| **GGUF Q4_K_M** | ~75 | **5–6×** **(est.)** | 2–4 pp | 3/5 |
| ONNX Runtime INT8 dynamic | ~180 | 1.5–3× (6× w/ VNNI) | <1 pp | 2/5 |
| `torch.compile` (no quant) | 475 | 1.3–1.7× **(est.)** | 0 | 2/5 |
| Prune + INT8 | ~180 | ~1.0× | unpredictable | 5/5 |

Sources: dev.to GPT-2 INT8 study (+22%), MS Azure HF×ORT INT8 blog (3×, 6× w/ VNNI), llama.cpp CPU kernel benchmarks (3–6× over torch FP32), GPT-2 INT4 numbers (1.76 s → 1.08 s ≈ 1.6×).

---

## 3. Recommended path for THIS model

**Ship two builds.** The model is 124M and the workload is short JSON outputs, so the right call is:

1. **Quick win (today):** `torch.quantize_dynamic` INT8 → cuts size 475→180 MB, latency 25–35 s → ~18–25 s. Zero risk to the tool-call task.
2. **Real win (next iteration):** convert to GGUF Q5_K_M and serve via `llama-cpp-python`. Expect 25–35 s → **5–8 s** on the same 4-core x86 CPU, with the 92% in-domain accuracy preserved within ~1 pp.

Skip GPTQ/AWQ. Their CPU story is weak and their 4-bit quality story on a 124M model is uncertain. Skip pruning — it eats the very fine-tune signal you trained on 1500 SFT items.

### 3.1 Runnable snippet — dynamic INT8 (ship today)

```python
# scripts/quantize_dynamic.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

CKPT = "weights/gpt2_ft_final.pt"
OUT  = "weights/gpt2_ft_int8.pt"

model = GPT2LMHeadModel.from_pretrained("gpt2")
state = torch.load(CKPT, map_location="cpu")
model.load_state_dict(state)
model.eval()

# Quantize every nn.Linear to INT8 dynamic (activations quantized at runtime).
qmodel = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save state_dict — load with the same quantize_dynamic call at inference time
# (the dynamic API does NOT serialize fully; wrap on load).
torch.save(qmodel.state_dict(), OUT)
print(f"saved {OUT}")
```

Inference-time load:

```python
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
model.load_state_dict(torch.load("weights/gpt2_ft_int8.pt", map_location="cpu"))
# then model.generate(...) as before
```

**Expected end size:** ~180 MB.
**Expected latency:** 18–25 s per command on the same 4-core CPU.
**Risks:** for very short generations the per-token Python/`generate()` overhead dominates; the speedup may land closer to 1.2× than 1.5×. Tool-call accuracy drop expected <1 pp; rerun `src/bench.py` to confirm against the 71.7% / 92% baselines before shipping.

### 3.2 Runnable plan — GGUF Q5_K_M (ship next)

```bash
# one-time setup
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_NATIVE=ON && cmake --build build -j --config Release
pip install -r requirements.txt llama-cpp-python

# 1) re-materialize the HF-format model from your .pt
python - <<'PY'
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
m = GPT2LMHeadModel.from_pretrained("gpt2")
m.load_state_dict(torch.load("../weights/gpt2_ft_final.pt", map_location="cpu"))
m.save_pretrained("../weights/gpt2_ft_hf")
GPT2Tokenizer.from_pretrained("gpt2").save_pretrained("../weights/gpt2_ft_hf")
PY

# 2) convert to GGUF (FP16 intermediate)
python convert_hf_to_gguf.py ../weights/gpt2_ft_hf \
    --outfile ../weights/gpt2_ft.f16.gguf --outtype f16

# 3) quantize to Q5_K_M (sweet spot) and Q4_K_M (Pi target)
./build/bin/llama-quantize ../weights/gpt2_ft.f16.gguf ../weights/gpt2_ft.Q5_K_M.gguf Q5_K_M
./build/bin/llama-quantize ../weights/gpt2_ft.f16.gguf ../weights/gpt2_ft.Q4_K_M.gguf Q4_K_M
```

Python inference:

```python
from llama_cpp import Llama
llm = Llama(model_path="weights/gpt2_ft.Q5_K_M.gguf",
            n_ctx=512, n_threads=4, logits_all=False)
out = llm("turn on kitchen light\n", max_tokens=64, temperature=0.0, stop=["\n\n"])
print(out["choices"][0]["text"])
```

**Expected end size:** ~90 MB (Q5_K_M) or ~75 MB (Q4_K_M).
**Expected latency on x86 4-core:** **5–8 s/command (estimated)** at Q5_K_M.
**Risks specific to the tool-call task:**
- JSON closing-brace failures: 4-bit quantization can shift low-probability tokens; mitigate with `temperature=0` + grammar-constrained decoding (llama.cpp GBNF) to force valid JSON.
- 100-name function fuzzy-match still corrects single-token hallucinations — keep the fuzzy matcher in the pipeline.
- Re-run `src/bench.py` adapted to the llama-cpp-python call against the 300 held-out items; reject the build if overall accuracy drops by more than 3 pp from the FP32 baseline.

---

## 4. Pi-specific

Raspberry Pi 4 (Cortex-A72, 4 cores, no VNNI, no AVX) and Pi 5 (Cortex-A76, ~2× faster) require **GGUF — there is no other realistic path.** `torch.quantize_dynamic` on Pi delivers little because `qnnpack`/`xnnpack` paths for GPT-2's `Conv1D` layers are unoptimized.

**Mandatory config for <10 s/command:**

- **Format:** GGUF **Q4_K_M** (75 MB).
- **Runtime:** `llama-cpp-python` built with `-DGGML_NATIVE=ON` so it picks up Cortex NEON / Pi 5 SVE2.
- **Threads:** `n_threads=4` on Pi 4, `n_threads=4` on Pi 5 (leave one core free for Whisper).
- **Context:** `n_ctx=256` — smart-home prompts are short; small ctx halves KV-cache cost.
- **Generation:** `max_tokens=64`, `temperature=0.0`, `top_k=1` (greedy).
- **Grammar:** GBNF grammar that forces `{"name": "<one of 100>", "arguments": {...}}` shape. This removes the long tail of broken-JSON retries that dominate worst-case latency.

**Estimated latency (Pi 4, Q4_K_M, 64 new tokens, 4 threads):** 6–9 s/command **(estimated by extrapolation** — TinyLlama 1.1B Q4_K_M on Pi 4 4 GB runs at 8–12 tok/s per published Pi benchmarks; GPT-2 124M is ~9× smaller, so ~40–80 tok/s gives 0.8–1.6 s decode for 64 tokens + 3–5 s prompt-eval and Python startup).

**Pi 5:** same Q4_K_M build, ~2.5–4 s/command **(estimated)**. Q5_K_M (~90 MB) is also viable on Pi 5 and recovers ~1–2 pp of accuracy.

**Hard constraints:**
- Pi 4 with 2 GB RAM: do **not** load Whisper Medium (770 MB) and GPT-2 Q4_K_M (75 MB) in the same process if other services run; use 4 GB Pi 4 minimum, ideally Pi 5 4 GB.
- Avoid FP32 PyTorch on Pi entirely — single-token decode is ~3–5 s, blowing the 10 s budget on any prompt longer than two tokens.

---

## TL;DR

1. **Today, 30 minutes of work:** `torch.quantize_dynamic` → 180 MB, ~1.3× faster, zero accuracy risk.
2. **Next iteration, 2–3 hours of work:** convert to GGUF Q5_K_M + `llama-cpp-python` → 90 MB, ~4–5× faster (5–8 s/cmd on x86), and the only realistic path for Raspberry Pi.
3. **Skip:** GPTQ, AWQ, bitsandbytes, pruning. Either wrong tool for CPU or wrong tool for a 124M model.
4. **Pi target:** GGUF Q4_K_M + GBNF JSON grammar + greedy decode. Pi 4 ~6–9 s/cmd, Pi 5 ~2.5–4 s/cmd (both estimated).

**Validation gate before shipping any quantized build:** rerun `src/bench.py` against the 300 held-out items and reject if overall accuracy drops more than 3 pp from the FP32 baseline (71.7% text / 92% in-domain).
