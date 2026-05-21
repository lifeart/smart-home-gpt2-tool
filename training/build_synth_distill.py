# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.25"]
# ///
"""B4 stage 2 — build the synth-distillation training set.

Takes `b4_synth_candidates.json` (stage 1: per-row v6 + v9 greedy candidate
calls) and turns each row into a training example for a SYNTHESIS-aware
GPT-2: the prompt carries the user query + function schemas + the two
candidate calls as labelled evidence; the target is the TRUE gold call.

The model learns: given candidates that are each maybe-right/maybe-wrong,
pick the correct function and assemble the correct arguments — exactly the
job the Llama synthesizer does in the API pipeline, but trained on ground
truth (≥ the 81.7% Llama teacher) and runnable fully in-browser.

Output: data/sh_train_synth.json, uploaded to lifeart/smart-home-sft-v2.
Run: python training/build_synth_distill.py
"""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DATA_REPO = "lifeart/smart-home-sft-v2"
ASSIST_MARKER = "\n\n\nASSISTANT:"


def compact(call) -> str:
    if not isinstance(call, dict):
        return "(no valid call)"
    return json.dumps(call, separators=(",", ":"))


def make_synth_prompt(prompt: str, v6_call, v9_call) -> str | None:
    """Insert a CANDIDATES evidence block before the ASSISTANT marker."""
    idx = prompt.find(ASSIST_MARKER)
    if idx == -1:
        return None
    head, tail = prompt[:idx], prompt[idx:]
    block = (
        "\n\nProposed tool calls (evidence from other systems — pick the "
        "right function, merge the best arguments, and fix wrong values):\n"
        f"- candidate A: {compact(v6_call)}\n"
        f"- candidate B: {compact(v9_call)}"
    )
    return head + block + tail


def main() -> None:
    print(f"[fetch] {DATA_REPO}/b4_synth_candidates.json")
    p = hf_hub_download(DATA_REPO, "b4_synth_candidates.json", repo_type="dataset")
    src = json.loads(Path(p).read_text())
    print(f"[data] {len(src)} candidate rows")

    rows, skipped, both_none = [], 0, 0
    for r in src:
        sp = make_synth_prompt(r["prompt"], r.get("v6_call"), r.get("v9_call"))
        if sp is None:
            skipped += 1
            continue
        if not isinstance(r.get("v6_call"), dict) and not isinstance(r.get("v9_call"), dict):
            both_none += 1  # keep — model must still produce the gold unaided
        rows.append({
            "prompt": sp,
            "gold": r["gold"],
            "gold_name": r.get("gold_name"),
            "domain": r.get("domain", "?"),
            "task": "synth",
        })

    out = ROOT / "data" / "sh_train_synth.json"
    out.write_text(json.dumps(rows))
    print(f"[build] {len(rows)} synth-distillation rows "
          f"(skipped {skipped} no-marker; {both_none} had no usable candidate)")

    # how often a candidate already nails it — the headroom the synth model
    # has to beat (it can copy when a candidate is right, fix when not)
    cand_right = 0
    for r, s in zip(src, rows):
        gold = json.loads(s["gold"]) if isinstance(s["gold"], str) else s["gold"]
        for c in (r.get("v6_call"), r.get("v9_call")):
            if isinstance(c, dict) and c == gold:
                cand_right += 1
                break
    print(f"[stat] a candidate exactly matches gold in {cand_right}/{len(rows)} "
          f"= {cand_right/len(rows)*100:.1f}% (oracle-copy ceiling lower bound)")

    print(f"[sample] {rows[0]['prompt'][-400:]!r}")
    print(f"[sample] gold: {rows[0]['gold']}")

    HfApi().upload_file(
        path_or_fileobj=str(out), path_in_repo="sh_train_synth.json",
        repo_id=DATA_REPO, repo_type="dataset",
        commit_message="B4 stage 2: synth-distillation training set",
    )
    print(f"[push] -> {DATA_REPO}/sh_train_synth.json")


if __name__ == "__main__":
    main()
