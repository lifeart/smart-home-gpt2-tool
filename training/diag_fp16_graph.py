# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "onnx>=1.16",
#   "numpy",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 37 diagnostic 2 — inspect the v14 fp16 ONNX graph.

diag_fp16_overflow.py ruled out activation overflow (peak ~3051 vs fp16 max
65504). So the fp16-on-WebGPU garbage is a graph/conversion issue, not
range. This downloads model_fp16.onnx and reports:
  - op-type histogram
  - every Cast node and its target type
  - which initializers are fp16 vs fp32 (what the converter kept in fp32)
  - any initializer holding non-finite or out-of-fp16-range values
  - graph input/output dtypes
so we can see exactly what the float16 conversion produced.

Run: python training/diag_fp16_graph.py
"""
from __future__ import annotations

import collections

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from onnx import TensorProto, numpy_helper

REPO = "lifeart/smart-home-gpt2-v14-ctx4096"
FP16_MAX = 65504.0
_DT = {v: k for k, v in TensorProto.DataType.items()}


def main() -> None:
    print(f"[fetch] {REPO}/onnx/model_fp16.onnx")
    p = hf_hub_download(REPO, "onnx/model_fp16.onnx", subfolder=None)
    print(f"[load] {p}")
    m = onnx.load(p)
    g = m.graph

    # ---- op-type histogram ----
    ops = collections.Counter(n.op_type for n in g.node)
    print(f"\n=== op types ({len(g.node)} nodes) ===")
    for op, c in ops.most_common():
        print(f"  {c:5d}  {op}")

    # ---- Cast nodes ----
    casts = collections.Counter()
    for n in g.node:
        if n.op_type == "Cast":
            to = next((a.i for a in n.attribute if a.name == "to"), None)
            casts[_DT.get(to, to)] += 1
    print(f"\n=== Cast nodes by target type ===")
    for t, c in casts.most_common():
        print(f"  {c:5d}  -> {t}")

    # ---- initializers: dtype split + bad values ----
    by_dt = collections.Counter()
    bad = []
    for init in g.initializer:
        by_dt[_DT.get(init.data_type, init.data_type)] += 1
        try:
            arr = numpy_helper.to_array(init)
        except Exception:
            continue
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        af = arr.astype(np.float64)
        nf = ~np.isfinite(af)
        big = np.isfinite(af) & (np.abs(af) > FP16_MAX)
        if nf.any() or big.any():
            finite = af[np.isfinite(af)]
            bad.append((
                init.name, _DT.get(init.data_type, init.data_type),
                int(nf.sum()), int(big.sum()),
                float(finite.min()) if finite.size else float("nan"),
                float(finite.max()) if finite.size else float("nan"),
            ))
    print(f"\n=== initializers by dtype ({len(g.initializer)} total) ===")
    for t, c in by_dt.most_common():
        print(f"  {c:5d}  {t}")

    print(f"\n=== initializers with non-finite or >|{FP16_MAX}| values ===")
    if not bad:
        print("  none")
    for name, dt, nnf, nbig, lo, hi in bad:
        print(f"  {name}  [{dt}]  non_finite={nnf} over_fp16={nbig} "
              f"finite_range=[{lo:.3g}, {hi:.3g}]")

    # ---- graph I/O ----
    def io(vs):
        return [(v.name, _DT.get(v.type.tensor_type.elem_type,
                                 v.type.tensor_type.elem_type)) for v in vs]
    print(f"\n=== graph inputs ===")
    for n, t in io(g.input):
        print(f"  {n}: {t}")
    print(f"=== graph outputs ===")
    for n, t in io(g.output):
        print(f"  {n}: {t}")

    # ---- fp32 islands: nodes whose output value_info is still fp32 ----
    vi_fp32 = [vi.name for vi in g.value_info
               if vi.type.tensor_type.elem_type == TensorProto.FLOAT]
    print(f"\n=== intermediate tensors still fp32 (value_info): "
          f"{len(vi_fp32)} ===")
    for n in vi_fp32[:40]:
        print(f"  {n}")
    if len(vi_fp32) > 40:
        print(f"  ... +{len(vi_fp32) - 40} more")


if __name__ == "__main__":
    main()
