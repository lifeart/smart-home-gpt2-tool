# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.4", "transformers>=4.45"]
# ///
"""Iter 40 — verify retrieval-pruning recall (speed item).

main.js auto-prunes long prompts to the top-8 MiniLM-retrieved schemas
(web/retrieval.js + pruneSchemas). The risk: if the gold function is not
in the top-8, pruning drops it and the answer is unrecoverable. This
measures recall@K — does the gold survive — by replicating the browser's
MiniLM retrieval (Xenova/all-MiniLM-L6-v2 == sentence-transformers/
all-MiniLM-L6-v2) and the `buildFunctionIndex` text recipe exactly.

No browser. Run: python training/verify_retrieval_recall.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
NONE_SENTINEL = "__NONE__"
NONE_TEXT = ("__NONE__ — no function applies, decline the request, the user "
             "asks something off-topic or wants no action")


def index_text(name: str, d: dict, registry: dict) -> str:
    """Mirror web/retrieval.js buildFunctionIndex."""
    desc = (d or {}).get("description", "")
    params = (d or {}).get("params") or list(
        ((registry.get(name) or {}).get("params") or {}).keys()
    )
    pstr = f"parameters: {', '.join(params)}" if params else "no parameters"
    ex = ((d or {}).get("examples") or [])[:3]
    estr = f" examples: {' | '.join(ex)}" if ex else ""
    return f"{name} — {desc} {pstr}{estr}".strip()


def extract_user(prompt: str) -> str:
    i = prompt.rfind("USER:")
    if i == -1:
        return ""
    after = prompt[i + 5:]
    end = len(after)
    for m in ("\n\n", "ASSISTANT:"):
        j = after.find(m)
        if j != -1:
            end = min(end, j)
    return after[:end].strip()


def main() -> None:
    descs = json.loads((ROOT / "web/public/eval/function_descriptions.json").read_text())
    registry = json.loads((ROOT / "data/tool_registry.json").read_text())
    test = json.loads((ROOT / "data/sh_test.json").read_text())

    names = sorted(set(registry) | set(descs))
    texts = [index_text(n, descs.get(n), registry) for n in names]
    names.append(NONE_SENTINEL)
    texts.append(NONE_TEXT)
    print(f"[index] {len(names)} functions (incl. __NONE__ sentinel)")

    print(f"[model] loading {MINILM}")
    tok = AutoTokenizer.from_pretrained(MINILM)
    model = AutoModel.from_pretrained(MINILM).eval()

    @torch.no_grad()
    def embed(batch: list[str]) -> torch.Tensor:
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt")
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(mean, p=2, dim=1)

    idx = embed(texts)  # (N, 384)

    indexed = set(names)
    Ks = [6, 8, 10, 12]
    hit = {k: 0 for k in Ks}
    scored = 0
    not_indexed = 0
    queries = [extract_user(r["prompt"]) for r in test]
    qvecs = []
    for i in range(0, len(queries), 64):
        qvecs.append(embed(queries[i:i + 64]))
    qvecs = torch.cat(qvecs, 0)

    for r, qv in zip(test, qvecs):
        gold = r.get("gold_name")
        if gold not in indexed or gold == "none":
            not_indexed += 1
            continue
        scored += 1
        sims = (qv @ idx.T)
        order = sims.argsort(descending=True).tolist()
        ranked = [names[j] for j in order]
        for k in Ks:
            topk = [n for n in ranked if n != NONE_SENTINEL][:k]
            if gold in topk:
                hit[k] += 1

    print(f"\n[recall] scored {scored} queries "
          f"({not_indexed} skipped — gold not in the retrieval index)")
    for k in Ks:
        print(f"  recall@{k:<2} = {hit[k]}/{scored} = {hit[k]/scored*100:.1f}%")
    r8 = hit[8] / scored * 100
    print(f"\n[verdict] retrieval-pruning keeps the gold for {r8:.1f}% of "
          f"queries at top-8.")
    print("  >=97%: safe to ship.  93-97%: usable, small accuracy cost.  "
          "<93%: raise K or gate tighter.")


if __name__ == "__main__":
    main()
