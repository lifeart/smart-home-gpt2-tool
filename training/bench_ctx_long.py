# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
# ]
# ///
"""Iter 36 — long-context accuracy bench, per length bucket.

`sh_test.json` is entirely <=739 tokens, so the headline "v13-ctx4096 =
77.3%" only ever measured short prompts. This bench runs `sh_test_long.json`
(built by `build_longctx.py` — the same 300 requests replicated into
short / 1500 / 2500 / 3500-token buckets) and reports name + exact accuracy
PER BUCKET PER MODEL, so the 4096-token behaviour is finally visible.

Each model is shown only what its window allows: a prompt longer than
`n_positions - max_new` is front-truncated (keeps the recent tokens), which
is exactly how the browser runtime degrades. So v9 (1024 ctx) on a 2500-
token bucket measures "v9 with the schema list clipped" — a fair baseline
for "does the 4096 window actually buy accuracy".

Run on HF Jobs:
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN --timeout 2h \\
        -e MODEL_REPOS="lifeart/smart-home-gpt2-v9,\\
lifeart/smart-home-gpt2-v13-ctx4096,lifeart/smart-home-gpt2-v14-ctx4096" \\
        training/bench_ctx_long.py
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
    "lifeart/smart-home-gpt2-v9,"
    "lifeart/smart-home-gpt2-v13-ctx4096,"
    "lifeart/smart-home-gpt2-v14-ctx4096",
).split(",")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "lifeart/smart-home-sft-v2")
BUCKETS = ["short", "1500", "2500", "3500"]

LOCAL_TEST = Path(__file__).resolve().parent.parent / "data" / "sh_test_long.json"
TEST_URL = (  # fallback only; long test set normally rides in the dataset repo
    "https://huggingface.co/datasets/lifeart/smart-home-sft-v2/resolve/main/"
    "sh_test_long.json"
)


# ---------------- scoring (compact port of bench_common.py) ----------------

def extract_json(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_call(text: str) -> dict:
    js = extract_json(text)
    obj = None
    if js:
        try:
            obj = json.loads(js)
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        return {"name": None, "arguments": {}}
    name = obj.get("name") if isinstance(obj.get("name"), str) else None
    args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
    return {"name": name, "arguments": args}


def norm(v):
    if isinstance(v, bool) or v is None or isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        t = v.strip()
        if t and re.fullmatch(r"-?\d+(\.\d+)?", t):
            return float(t) if "." in t else int(t)
        return t.lower()
    return v


def args_match(a, b) -> bool:
    a = a if isinstance(a, dict) else {}
    b = b if isinstance(b, dict) else {}
    if sorted(a.keys()) != sorted(b.keys()):
        return False
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(bv, dict):
            if not args_match(av, bv):
                return False
        elif isinstance(bv, list):
            if not isinstance(av, list) or len(av) != len(bv):
                return False
            try:
                if sorted(map(norm, av)) != sorted(map(norm, bv)):
                    return False
            except TypeError:
                if json.dumps(av, sort_keys=True) != json.dumps(bv, sort_keys=True):
                    return False
        else:
            an, bn = norm(av), norm(bv)
            if an is None and bn is None:
                continue
            if an is None or bn is None:
                return False
            if isinstance(an, (int, float)) and isinstance(bn, (int, float)):
                if an != bn:
                    return False
            elif str(an) != str(bn):
                return False
    return True


def parse_gold(gold) -> dict:
    try:
        obj = json.loads(gold) if isinstance(gold, str) else gold
    except Exception:
        return {"name": None, "arguments": {}}
    if not isinstance(obj, dict):
        return {"name": None, "arguments": {}}
    return {
        "name": obj.get("name") if isinstance(obj.get("name"), str) else None,
        "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {},
    }


# ---------------- test loader ----------------

def load_test() -> list[dict]:
    if LOCAL_TEST.exists():
        return json.loads(LOCAL_TEST.read_text())
    try:
        p = hf_hub_download(
            repo_id="lifeart/smart-home-sft-v2",
            filename="sh_test_long.json",
            repo_type="dataset",
        )
        return json.loads(Path(p).read_text())
    except Exception as e:
        print(f"[test] hub fetch failed: {e}; trying direct URL")
    with urllib.request.urlopen(TEST_URL) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------- generation ----------------

NAME_RE = re.compile(r"""["']?name["']?\s*:\s*["']([^"',}\s]+)""")


@torch.no_grad()
def generate(model, tok, prompt: str, device, max_new: int = 64) -> str:
    """Greedy decode with KV cache. Generates the whole call and stops at EOS
    (the SFT target is `call + eos`), so the emitted JSON is complete — no
    mid-object truncation. A prompt longer than the window is front-truncated,
    mirroring how the browser runtime degrades."""
    ids = tok.encode(prompt, add_special_tokens=False)
    cap = model.config.n_positions
    if len(ids) > cap - max_new:
        ids = ids[-(cap - max_new):]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    out = model.generate(
        input_ids, attention_mask=attn, max_new_tokens=max_new,
        do_sample=False, num_beams=1, use_cache=True,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()


def bench_one(repo: str, test: list[dict], device) -> dict:
    print(f"\n[load] {repo}")
    tok = GPT2TokenizerFast.from_pretrained(repo)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(repo).to(device).eval()
    model.config.use_cache = True
    ctx = model.config.n_positions
    print(f"[cfg] n_positions={ctx}")

    rows = []
    t0 = time.time()
    for i, s in enumerate(test):
        out = generate(model, tok, s["prompt"], device)
        pred = parse_call(out)
        gold = parse_gold(s["gold"])
        # name: prefer the balanced-JSON parse, fall back to a tolerant regex
        # so a stray-token output still yields a name where one is present
        pred_name = pred["name"]
        if pred_name is None:
            m = NAME_RE.search(out)
            pred_name = m.group(1) if m else None
        name_ok = pred_name is not None and pred_name == s.get("gold_name")
        exact_ok = (
            name_ok
            and s.get("exact_ok_valid", True)
            and args_match(pred["arguments"], gold["arguments"])
        )
        rows.append({
            "bucket": s.get("bucket", "?"),
            "domain": s.get("domain", "?"),
            "exact_valid": s.get("exact_ok_valid", True),
            "name_ok": name_ok,
            "exact_ok": exact_ok,
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(test)}] t={time.time()-t0:.0f}s", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    by_bucket = {}
    for b in BUCKETS:
        br = [r for r in rows if r["bucket"] == b]
        if not br:
            continue
        ev = [r for r in br if r["exact_valid"]]
        by_bucket[b] = {
            "n": len(br),
            "name_acc": sum(r["name_ok"] for r in br) / len(br),
            "exact_n": len(ev),
            "exact_acc": (sum(r["exact_ok"] for r in ev) / len(ev)) if ev else None,
        }
    return {"repo": repo, "ctx": ctx, "by_bucket": by_bucket,
            "elapsed_s": time.time() - t0}


def main() -> None:
    test = load_test()
    counts = {b: sum(1 for r in test if r.get("bucket") == b) for b in BUCKETS}
    print(f"[test] {len(test)} rows  buckets={counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    results = [bench_one(r.strip(), test, device) for r in MODEL_REPOS if r.strip()]

    print("\n\n===== NAME ACCURACY  (model x token bucket) =====")
    hdr = "  " + f"{'model':<34}" + "".join(f"{b:>11}" for b in BUCKETS)
    print(hdr)
    for r in results:
        line = f"  {r['repo']:<34}"
        for b in BUCKETS:
            bb = r["by_bucket"].get(b)
            line += (f"{bb['name_acc']*100:9.1f}% " if bb else f"{'-':>11}")
        print(line)

    print("\n===== EXACT-MATCH ACCURACY (exact_ok_valid rows only) =====")
    print(hdr)
    for r in results:
        line = f"  {r['repo']:<34}"
        for b in BUCKETS:
            bb = r["by_bucket"].get(b)
            if bb and bb["exact_acc"] is not None:
                line += f"{bb['exact_acc']*100:9.1f}% "
            else:
                line += f"{'-':>11}"
        print(line)

    for r in results:
        print(f"\n  {r['repo']}  (ctx={r['ctx']}, {r['elapsed_s']:.0f}s)")
        for b in BUCKETS:
            bb = r["by_bucket"].get(b)
            if bb:
                ex = (f"{bb['exact_acc']*100:.1f}%"
                      if bb["exact_acc"] is not None else "n/a")
                print(f"    {b:<7} name={bb['name_acc']*100:5.1f}%  "
                      f"exact={ex:<7} (n={bb['n']}, exact_n={bb['exact_n']})")

    out = Path("bench_ctx_long_results.json")
    out.write_text(json.dumps({"results": results, "counts": counts}, indent=2))
    print(f"\n[save] wrote {out}")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        try:
            HfApi().upload_file(
                path_or_fileobj=str(out),
                path_in_repo="iter36_ctx_long_bench.json",
                repo_id=RESULTS_REPO,
                repo_type="dataset",
                commit_message="Iter 36 long-context bench (v9/v13/v14 x buckets)",
            )
            print(f"[push] results -> {RESULTS_REPO}")
        except Exception as e:
            print(f"[push] failed: {e}")


if __name__ == "__main__":
    main()
