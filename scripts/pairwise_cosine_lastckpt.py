#!/usr/bin/env python3
"""
pairwise_cosine_lastckpt.py — Re-run the key pairwise comparisons using the
LITERAL last checkpoint of each run, instead of the `final_adapter` directory
that HF Trainer populates via `load_best_model_at_end=True`.

This is a robustness check: our main `pairwise_cosine.json` reports cosines
between best-eval-loss adapters. If the bimodal structure (transfer ~0.78
across seeds vs direct ~0.03) only holds at best-eval but breaks at literal
end-of-training, we want to know.
"""

import json
import os
import re
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairwise_cosine import load_AB, cosine_per_module  # noqa: E402


RUNS = {
    "E3_s42":   "./output_ky_baseline_r16_lr2e4_3ep",
    "E3_s123":  "./output_ky_baseline_r16_lr2e4_3ep_seed123",
    "E5_s42":   "./output_ky_collapse_r64_lr5e4_5ep",
    "E5_s123":  "./output_ky_collapse_r64_lr5e4_5ep_seed123",
    "E5b_s42":  "./output_ky_r64_lr2e4_5ep",
    "E5b_s123": "./output_ky_r64_lr2e4_5ep_seed123",
    "E5c_s42":  "./output_ky_r64_lr2e4_3ep",
    "E5c_s123": "./output_ky_r64_lr2e4_3ep_seed123",
    "E6_s42":   "./output_ky_from_kz_r16_lr2e4_3ep",
    "E6_s123":  "./output_ky_from_kz_r16_lr2e4_3ep_seed123",
    "E6b_s42":  "./output_en_to_ky_r16_lr2e4_3ep",
    "E6b_s123": "./output_en_to_ky_r16_lr2e4_3ep_seed123",
    "E8_s42":   "./output_en_baseline_r16_lr2e4_3ep",
    "E8_s123":  "./output_en_baseline_r16_lr2e4_3ep_seed123",
    "E1b_s42":  "./output_kz_tokenmatched_r16_lr2e4_3ep",
    "E1b_s123": "./output_kz_tokenmatched_r16_lr2e4_3ep_seed123",
}


PAIRS = [
    # Same-config cross-seed (direct training)
    ("E3_s42",   "E3_s123",   "KY baseline, cross-seed"),
    ("E5_s42",   "E5_s123",   "KY r=64 collapse, cross-seed"),
    ("E5b_s42",  "E5b_s123",  "KY r=64 healthy 5ep, cross-seed"),
    ("E5c_s42",  "E5c_s123",  "KY r=64 healthy 3ep, cross-seed"),
    ("E8_s42",   "E8_s123",   "EN baseline, cross-seed"),
    ("E1b_s42",  "E1b_s123",  "KZ small-corpus, cross-seed"),

    # Same-config cross-seed (transfer)
    ("E6_s42",   "E6_s123",   "KZ->KY transfer, cross-seed"),
    ("E6b_s42",  "E6b_s123",  "EN->KY transfer, cross-seed"),

    # Collapse vs healthy
    ("E5_s42",   "E5c_s42",   "r=64 collapse vs healthy 3ep"),
    ("E5_s42",   "E5b_s42",   "r=64 collapse vs healthy 5ep"),
    ("E5b_s42",  "E5c_s42",   "r=64 healthy 5ep vs healthy 3ep"),
]


def last_checkpoint(run_dir):
    """Return the directory of the literal last checkpoint (highest step)."""
    cks = []
    for n in os.listdir(run_dir):
        m = re.match(r"checkpoint-(\d+)$", n)
        if m and os.path.isfile(os.path.join(run_dir, n, "adapter_model.safetensors")):
            cks.append((int(m.group(1)), os.path.join(run_dir, n)))
    if not cks:
        return None
    return max(cks)[1]


def main():
    cache = {}
    rows = []
    results = {}
    for a, b, desc in PAIRS:
        for name in (a, b):
            if name in cache:
                continue
            ck = last_checkpoint(RUNS[name])
            if ck is None:
                print(f"  [SKIP] {name}: no checkpoints under {RUNS[name]}")
                cache[name] = None
                continue
            cache[name] = load_AB(ck, torch.float32)
            print(f"  loaded {name}: {os.path.basename(ck)}")
        if cache.get(a) is None or cache.get(b) is None:
            continue
        with torch.no_grad():
            cs = cosine_per_module(cache[a], cache[b])
        arr = np.array(list(cs.values()))
        summary = {
            "pair": [a, b], "description": desc,
            "n_modules": len(arr),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()), "max": float(arr.max()),
            "std": float(arr.std()),
            "frac_gt_0.5": float((arr > 0.5).sum() / len(arr)),
            "frac_gt_0.9": float((arr > 0.9).sum() / len(arr)),
        }
        results[f"{a}__VS__{b}"] = summary
        rows.append((a, b, summary, desc))
        print(f"  {a:10s} vs {b:10s}  mean={summary['mean']:+.4f}  "
              f"median={summary['median']:+.4f}  "
              f">0.5: {int(summary['frac_gt_0.5']*summary['n_modules'])}/{summary['n_modules']}")

    out_path = "directional_results/pairwise_cosine_lastckpt.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] wrote {out_path}")

    print("\n" + "=" * 96)
    print(f"{'Pair':<25} {'mean':>8} {'median':>8} {'max':>7} {'>0.5':>8}   description")
    print("-" * 96)
    for a, b, s, d in rows:
        n = s["n_modules"]
        print(f"{a+' vs '+b:<25} {s['mean']:+8.4f} {s['median']:+8.4f} "
              f"{s['max']:+7.4f} {int(s['frac_gt_0.5']*n):>3}/{n:<3}  {d}")


if __name__ == "__main__":
    main()
