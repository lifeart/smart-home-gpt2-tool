# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Bench a HF GPT-2 SFT checkpoint on data/sh_test.json (300 items).

Mirrors src/bench.py correctness logic (fuzzy name regex parse), but loads
weights from HF Hub via transformers (safetensors), so works for both v1
(lifeart/smart-home-gpt2) and v2 (lifeart/smart-home-gpt2-v2).

Run on HF Jobs:
    hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN \\
        -e MODEL_REPOS="lifeart/smart-home-gpt2,lifeart/smart-home-gpt2-v2" \\
        training/bench_hf.py

Or locally:
    python training/bench_hf.py
"""
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_REPOS = os.environ.get(
    "MODEL_REPOS",
    "lifeart/smart-home-gpt2,lifeart/smart-home-gpt2-v2",
).split(",")

# Test file source: prefer local copy, else fetch from github mirror
LOCAL_TEST = Path(__file__).resolve().parent.parent / "data" / "sh_test.json"
TEST_URL = (
    "https://raw.githubusercontent.com/barometech/smart-home-gpt2/master/"
    "data/sh_test.json"
)

# Push results into a dataset repo so we can grab them later (HF Jobs only)
RESULTS_REPO = os.environ.get("RESULTS_REPO", "lifeart/smart-home-sft-v2")

NAME_RE = re.compile(r"""["'`]?name["'`]?\s*:\s*["']([^"'(\s,\}]+)""")


def parse_name(text: str) -> str | None:
    m = NAME_RE.search(text)
    return m.group(1) if m else None


def load_test() -> list[dict]:
    if LOCAL_TEST.exists():
        with LOCAL_TEST.open() as f:
            return json.load(f)
    # Try HF Hub dataset repo
    try:
        p = hf_hub_download(
            repo_id="lifeart/smart-home-sft-v2",
            filename="sh_test.json",
            repo_type="dataset",
        )
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"[test] hub fetch failed: {e}; falling back to GitHub")
    with urllib.request.urlopen(TEST_URL) as r:
        return json.loads(r.read().decode("utf-8"))


@torch.no_grad()
def generate_call(model, tok, prompt: str, device, max_new: int = 80) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    # Same window logic as src/bench.py (keep last 900 to leave room)
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
        if i < 8 or (not ok and len(samples) < 16):
            samples.append({
                "i": i,
                "domain": d,
                "gold_name": s["gold_name"],
                "pred": pred,
                "raw": out[:200],
                "ok": ok,
            })
        if (i + 1) % 25 == 0:
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
        print(f"    {d:<10} {c}/{n} = {c/n*100:.1f}%")
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
    print(f"[test] {len(test)} held-out items")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"[device] {device}")

    results: list[dict] = []
    for repo in MODEL_REPOS:
        repo = repo.strip()
        if not repo:
            continue
        results.append(bench_one(repo, test, device))

    print("\n\n===== SUMMARY =====")
    for r in results:
        print(f"  {r['repo']:<40}  {r['acc']*100:5.1f}%  ({r['correct']}/{r['n']})")

    # Save JSON
    out_path = Path("bench_results.json")
    out_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"[save] wrote {out_path}")

    # Try to push results to a dataset repo (best-effort; needs HF_TOKEN)
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(out_path),
                path_in_repo="bench_results.json",
                repo_id=RESULTS_REPO,
                repo_type="dataset",
                commit_message="bench v1 + v2 on sh_test.json",
            )
            print(f"[push] results -> {RESULTS_REPO}")
        except Exception as e:
            print(f"[push] failed: {e}")


if __name__ == "__main__":
    main()
