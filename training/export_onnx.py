# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.4.1",
#   "transformers==4.43.4",
#   "optimum[onnxruntime]==1.21.3",
#   "onnx==1.16.2",
#   "onnxruntime==1.19.2",
#   "onnxconverter-common==1.14.0",
#   "onnxslim==0.1.31",
#   "onnx-graphsurgeon==0.5.2",
#   "onnxscript",
#   "huggingface_hub>=0.25",
#   "sentencepiece",
#   "protobuf",
# ]
# ///
"""Export lifeart/smart-home-gpt2 to ONNX (fp32, fp16, q8) and push to Hub.

This is based on the transformers.js conversion pipeline (scripts/convert.py +
scripts/quantize.py from the v3 branch) with two changes that were needed to
produce variants that actually load under onnxruntime-web / transformers.js
on WebGPU:

* fp16: use `onnxruntime.transformers.float16.convert_float_to_float16` rather
  than `onnxconverter_common.float16`. The onnxconverter_common version leaves
  Cast nodes' `to` attributes in a state that causes ORT to fail session
  creation with "Type (tensor(float16)) of output arg (/transformer/Cast_2_*)
  ... does not match expected type (tensor(float))" -- the failure surfaces
  during ORT's SimplifiedLayerNormFusion pass. The ORT-transformers float16
  converter understands transformer graph topology and emits a model that
  loads cleanly on both CPU and WebGPU EPs.

* q8: dynamic quantization through `ONNXQuantizer` (not `quantize_dynamic`)
  with `weight_qType=QInt8` (signed; gpt-2 has no Conv layers so QInt8 is
  supported on ORT-Web), `per_channel=False`, `reduce_range=False` (gpt-2 is
  in the NO_PER_CHANNEL_REDUCE_RANGE_MODELS list), restricted to the
  IntegerOpsRegistry op types, and `MatMulConstBOnly=True` so activations
  stay in fp32 and only weight matrices are int8. CRITICALLY we exclude the
  LM head MatMul, the token-embedding Gather, and the position-embedding
  Gather from quantization: gpt-2 has `tie_word_embeddings=True`, so int8
  round-tripping the shared 50257x768 embedding table with a single scalar
  scale destroys the softmax distribution and produces gibberish like
  "akiaolicuyomi...". Excluding those three nodes raises the file from
  ~156 MB to ~380 MB but the model actually generates valid output.

Output layout inside the HF Hub repo (and `upload_onnx/`):
  onnx/
    model.onnx              fp32
    model_fp16.onnx         fp16
    model_quantized.onnx    int8 dynamic (Q8 / QInt8), embeddings + LM head fp32
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Set

import onnx
import onnxslim
import onnx_graphsurgeon as gs
from huggingface_hub import HfApi, snapshot_download

# `onnxruntime.transformers.float16` is much more robust than
# `onnxconverter_common.float16` for transformer models -- it correctly
# handles Cast nodes, LayerNorm fusion candidates, and the fp16 op-block
# list, which all together avoids the "Type (tensor(float16)) ... expected
# tensor(float)" load error in onnxruntime-web that the old converter
# produces on gpt-2.
from onnxruntime.transformers import float16
from onnxruntime.quantization import QuantType, QuantizationMode
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.registry import IntegerOpsRegistry
from optimum.onnx.graph_transformations import check_and_save_model
from optimum.onnxruntime import ORTModelForCausalLM

REPO = os.environ.get("TARGET_REPO", "lifeart/smart-home-gpt2")
WORK = Path("onnx_export")
WORK.mkdir(exist_ok=True)


def _operator_set(model: onnx.ModelProto) -> Set[str]:
    ops: Set[str] = set()

    def _walk(graph) -> None:
        for node in graph.node:
            ops.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _walk(attr.g)

    _walk(model.graph)
    return ops


def export_fp32(local_repo: str, dest: Path) -> Path:
    print(f"[fp32] exporting via optimum from {local_repo}")
    ort_model = ORTModelForCausalLM.from_pretrained(local_repo, export=True)
    ort_model.save_pretrained(dest)
    fp32 = dest / "model.onnx"
    assert fp32.exists(), f"missing {fp32}: {list(dest.iterdir())}"

    # Slim the graph (constant folding, dead node elimination, shape inference).
    # This is what makes the downstream fp16 conversion well-behaved.
    print("[fp32] slimming with onnxslim")
    try:
        slimmed = onnxslim.slim(str(fp32))
        check_and_save_model(slimmed, str(fp32))
    except Exception as e:  # noqa: BLE001
        print(f"  onnxslim failed (continuing with un-slimmed model): {e}")

    print(f"  -> {fp32} ({fp32.stat().st_size / 1024 / 1024:.1f} MB)")
    return fp32


def export_fp16(fp32_path: Path, dest: Path) -> Path:
    out = dest / "model_fp16.onnx"
    print(f"[fp16] converting {fp32_path} -> {out}")

    model = onnx.load(str(fp32_path))
    disable_shape_infer = model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF

    # NOTE: this is `onnxruntime.transformers.float16`, NOT
    # `onnxconverter_common.float16`. The former is what produces working fp16
    # transformer ONNX files in ORT-Web; the latter leaves Cast nodes in a
    # state that breaks ORT's SimplifiedLayerNormFusion pass with errors like:
    #   "Type (tensor(float16)) of output arg ... does not match expected type
    #   (tensor(float))"
    # and (locally on CPU):
    #   "Attempting to get index by a name which does not exist:
    #   InsertedPrecisionFreeCast_/transformer/ln_f/Constant_output_0 for node:
    #   /transformer/h.0/ln_1/Mul/SimplifiedLayerNormFusion".
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=disable_shape_infer,
    )

    # Re-toposort so any inserted Cast nodes sit before their consumers.
    graph = gs.import_onnx(model_fp16)
    graph.toposort()
    model_fp16 = gs.export_onnx(graph)

    try:
        onnx.checker.check_model(model_fp16, full_check=True)
        print("  onnx.checker.check_model: ok")
    except Exception as e:  # noqa: BLE001
        print(f"  onnx.checker.check_model failed (continuing): {e}")

    check_and_save_model(model_fp16, str(out))
    print(f"  -> {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    return out


def _find_lm_head_node(model: onnx.ModelProto) -> list[str]:
    """Locate the LM head node(s) -- the MatMul/Gemm that maps hidden states
    to vocab logits. For tied-embedding gpt-2 we want this node EXCLUDED
    from int8 quantization, otherwise the int8 round-trip through the shared
    embedding matrix destroys the softmax distribution and we get garbage
    tokens like "akiaolicuyomi...".

    Heuristic: walk graph outputs; the producer of `logits` is the LM head.
    If we can't find it by output name, fall back to "the last MatMul/Gemm
    in topological order whose output feeds a graph output".
    """
    lm_head: list[str] = []

    output_names = {o.name for o in model.graph.output}
    # Producers of logits (or graph outputs that look like logits)
    for n in model.graph.node:
        for out in n.output:
            if out == "logits" or out.endswith("/logits") or out.endswith("_logits"):
                if n.op_type in ("MatMul", "Gemm"):
                    lm_head.append(n.name or out)

    if not lm_head:
        # Fallback: find MatMul/Gemm nodes whose output is a graph output
        # (after any trailing Cast/Add) -- walk backward through Casts.
        producer_of: dict[str, onnx.NodeProto] = {o: n for n in model.graph.node for o in n.output}
        for graph_out in output_names:
            cur = graph_out
            for _ in range(8):  # cap the walk
                n = producer_of.get(cur)
                if n is None:
                    break
                if n.op_type in ("MatMul", "Gemm"):
                    if n.name:
                        lm_head.append(n.name)
                    break
                if n.op_type == "Cast" and n.input:
                    cur = n.input[0]
                    continue
                break

    return lm_head


def export_q8(fp32_path: Path, dest: Path) -> Path:
    out = dest / "model_quantized.onnx"
    print(f"[q8] quantizing {fp32_path} -> {out}")

    model = onnx.load(str(fp32_path))
    ops = _operator_set(model)
    # gpt-2 has no Conv layers; QInt8 is much more accurate than QUInt8 here.
    weight_type = QuantType.QUInt8 if "Conv" in ops else QuantType.QInt8
    print(f"  weight_type={weight_type.name}, ops_contains_Conv={'Conv' in ops}")

    # Exclude the LM head, token-embedding Gather, and position-embedding
    # Gather from quantization. gpt-2 has `tie_word_embeddings=True`, so the
    # input embeddings and the LM head share weights -- int8 round-tripping
    # those tables with a single per-tensor scale across 50257 rows produces
    # garbage tokens like "akiaolicuyomi...". Skipping all three keeps the
    # softmax distribution intact at the cost of ~110 MB of int8 savings.
    excluded: list[str] = list(_find_lm_head_node(model))
    for n in model.graph.node:
        if n.op_type == "Gather" and n.name and (
            "/wte/" in n.name or "/wpe/" in n.name
        ):
            excluded.append(n.name)
    print(f"  excluding from quantization: {excluded}")

    quantizer = ONNXQuantizer(
        model,
        per_channel=False,       # gpt-2 is in NO_PER_CHANNEL_REDUCE_RANGE_MODELS
        reduce_range=False,
        mode=QuantizationMode.IntegerOps,
        static=False,
        weight_qType=weight_type,
        activation_qType=QuantType.QUInt8,  # dynamic activation must be uint8
        tensors_range=None,
        nodes_to_quantize=[],
        nodes_to_exclude=excluded,
        op_types_to_quantize=list(IntegerOpsRegistry.keys()),
        extra_options=dict(
            EnableSubgraph=True,
            MatMulConstBOnly=True,
        ),
    )
    quantizer.quantize_model()
    check_and_save_model(quantizer.model.model, str(out))
    print(f"  -> {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    return out


def main() -> None:
    print(f"[1/4] downloading {REPO}")
    local = snapshot_download(REPO)

    print("[2/4] exporting fp32 ONNX via optimum + onnxslim")
    fp32 = export_fp32(local, WORK)

    print("[3/4] producing fp16 + q8 variants")
    export_fp16(fp32, WORK)
    export_q8(fp32, WORK)

    for name in ("model.onnx", "model_fp16.onnx", "model_quantized.onnx"):
        p = WORK / name
        print(f"  {name}: {p.stat().st_size / 1024 / 1024:.1f} MB")

    print("[4/4] uploading to HF Hub")
    api = HfApi()
    onnx_out = Path("upload_onnx")
    onnx_out.mkdir(exist_ok=True)
    for name in ("model.onnx", "model_fp16.onnx", "model_quantized.onnx"):
        shutil.copy(WORK / name, onnx_out / name)

    api.upload_folder(
        folder_path=str(onnx_out),
        repo_id=REPO,
        path_in_repo="onnx",
        commit_message="Re-export ONNX (fp32, fp16, q8) via transformers.js-style pipeline",
    )
    print(f"[done] pushed onnx/ to https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
