# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 15.4 — Cross-domain bench on 100 held-out fresh_bench items.

Loads lifeart/smart-home-sft-v2/sh_test_external.json (or local copy)
and benches each MODEL_REPOS entry. Mirrors bench_hf.py scoring exactly
(name-only via regex parse) so v4/v5/v6 numbers are comparable.

Run on HF Jobs:
    hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN \\
        -e MODEL_REPOS="lifeart/smart-home-gpt2-v4,lifeart/smart-home-gpt2-v5,lifeart/smart-home-gpt2-v6" \\
        training/bench_external_hf.py
"""
import json
import os
import re
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_REPOS = os.environ.get(
    "MODEL_REPOS",
    "lifeart/smart-home-gpt2-v4,lifeart/smart-home-gpt2-v5,lifeart/smart-home-gpt2-v6",
).split(",")

LOCAL_TEST = Path(__file__).resolve().parent.parent / "data" / "sh_test_external.json"
RESULTS_REPO = os.environ.get("RESULTS_REPO", "lifeart/smart-home-sft-v2")

NAME_RE = re.compile(r"""["'`]?name["'`]?\s*:\s*["']([^"'(\s,\}]+)""")


def parse_name(text: str) -> str | None:
    m = NAME_RE.search(text)
    return m.group(1) if m else None


def load_test() -> list[dict]:
    if LOCAL_TEST.exists():
        with LOCAL_TEST.open() as f:
            return json.load(f)
    p = hf_hub_download(
        repo_id="lifeart/smart-home-sft-v2",
        filename="sh_test_external.json",
        repo_type="dataset",
    )
    with open(p) as f:
        return json.load(f)


@torch.no_grad()
def generate_call(model, tok, prompt: str, device, max_new: int = 80) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) > 900:
        ids = ids[-900:]
    L = len(ids)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    close_brace = tok.encode("}", add_special_tokens=False)[0]
    newline = tok.encode("\n", add_special_tokens=False)[0]
    for _ in range(max_new):
        if cur.shape[1] >= 1024:
            break
        out = model(cur)
        logits = out.logits if hasattr(out, "logits") else out[0]
        nxt = int(logits[0, -1, :].argmax().item())
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == close_brace or nxt == newline:
            break
    new_ids = cur[0, L:].tolist()
    return tok.decode(new_ids, skip_special_tokens=True).strip()


def bench_one(repo: str, test_items: list[dict], device) -> dict:
    print(f"\n[load] {repo}")
    tok = GPT2TokenizerFast.from_pretrained(repo)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(repo).to(device).eval()

    by_domain: dict[str, list[int]] = {}
    correct = 0
    t0 = time.time()
    samples = []
    for i, s in enumerate(test_items):
        out = generate_call(model, tok, s["prompt"], device)
        pred = parse_name(out)
        ok = pred == s["gold_name"]
        if ok:
            correct += 1
        d = s.get("domain", "?")
        if d not in by_domain:
            by_domain[d] = [0, 0]
        by_domain[d][1] += 1
        if ok:
            by_domain[d][0] += 1
        if i < 5 or (not ok and len(samples) < 20):
            samples.append({
                "i": i,
                "domain": d,
                "gold_name": s["gold_name"],
                "pred": pred,
                "raw": out[:200],
                "ok": ok,
            })
        if (i + 1) % 20 == 0:
            print(
                f"  [{i+1}/{len(test_items)}] acc={correct/(i+1)*100:.1f}%  "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )

    acc = correct / len(test_items)
    elapsed = time.time() - t0
    print(f"\n=== {repo} ===")
    print(f"  Accuracy: {correct}/{len(test_items)} = {acc*100:.1f}%")
    print(f"  Time: {elapsed:.0f}s  ({elapsed/len(test_items):.2f}s/q)")
    print("  By domain:")
    for d, (c, n) in sorted(by_domain.items()):
        print(f"    {d:<28} {c}/{n} = {c/n*100:.1f}%")
    return {
        "repo": repo,
        "acc": acc,
        "n": len(test_items),
        "correct": correct,
        "by_domain": {d: {"correct": c, "total": n} for d, (c, n) in by_domain.items()},
        "elapsed_s": elapsed,
        "samples": samples,
    }


def main() -> None:
    test = load_test()
    print(f"[test] {len(test)} cross-domain held-out items")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"[device] {device}")

    results: list[dict] = []
    for repo in MODEL_REPOS:
        repo = repo.strip()
        if not repo:
            continue
        results.append(bench_one(repo, test, device))

    print("\n\n===== CROSS-DOMAIN SUMMARY =====")
    for r in results:
        print(f"  {r['repo']:<40}  {r['acc']*100:5.1f}%  ({r['correct']}/{r['n']})")

    out_path = Path("bench_external_results.json")
    out_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"[save] wrote {out_path}")

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(out_path),
                path_in_repo="bench_external_results.json",
                repo_id=RESULTS_REPO,
                repo_type="dataset",
                commit_message="iter15.4 — cross-domain bench v4/v5/v6 on 100-item external holdout",
            )
            print(f"[push] results -> {RESULTS_REPO}")
        except Exception as e:
            print(f"[push] failed: {e}")


if __name__ == "__main__":
    main()
