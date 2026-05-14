#!/usr/bin/env python3
"""
trajectory_cosine.py — Per-checkpoint pairwise cosine between LoRA adapters.

The final-state pairwise cosine of `pairwise_cosine.py` shows that direct
LoRA training from base is seed-orthogonal (cos ~0.02-0.04 across seeds) and
that the E5 vs E5c collapse/healthy gap is no larger. A natural follow-up
question is *when in training* this seed-orthogonality emerges: was it
already present at the first saved checkpoint, or did it accumulate?

This script answers four questions by computing per-module cosine at every
saved checkpoint of each run:

  Q1. WITHIN-RUN over time:  cos(adapter at step s1, adapter at step s2)
      Does the adapter rotate during training, or grow along a fixed axis?

  Q2. CROSS-SEED over time:  cos(run_A at step s, run_B at step s)
      Does seed-orthogonality emerge immediately, or grow over training?

  Q3. CROSS-CONFIG over time:  cos(E5 at step s, E5c at step s)
      Do collapse and healthy trajectories ever cross above the seed-noise
      floor before settling at the indistinguishable final state?

  Q4. TRANSFER-INIT over time:  cos(E6_s42 at step s, E6_s123 at step s)
      Does transfer-init keep the two seeds aligned throughout training,
      or do they start identical and drift?

Output: JSON dict keyed by comparison name, plus a multi-panel figure.

Memory: re-uses the AB-only formulation of pairwise_cosine.py.
"""

import argparse
import gc
import json
import os
import re
import time
import numpy as np
import torch

from safetensors.torch import load_file as load_safetensors

LORA_A_PAT = re.compile(
    r"base_model\.model\."
    r"(model\.layers\.\d+\.(?:self_attn|mlp)\.\w+)"
    r"\.lora_A(?:\.default)?\.weight"
)


# ----------------------------------------------------------------------------
# Run layout: each entry maps a short ID to its output directory.
# The script auto-discovers checkpoint-* subdirs and the final_adapter dir
# (treated as the "final" checkpoint for plotting).
# ----------------------------------------------------------------------------
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
}


def discover_checkpoints(path):
    """Return ordered list of (step:int, full_dir) for every saved point."""
    out = []
    if not os.path.isdir(path):
        return out
    for name in os.listdir(path):
        full = os.path.join(path, name)
        m = re.match(r"checkpoint-(\d+)$", name)
        if m and os.path.isfile(os.path.join(full, "adapter_model.safetensors")):
            out.append((int(m.group(1)), full))
    out.sort(key=lambda t: t[0])
    final = os.path.join(path, "final_adapter")
    if os.path.isdir(final):
        # Final step is the largest checkpoint step (or unknown if none saved).
        final_step = out[-1][0] if out else 0
        out.append((final_step, final))
    return out


def load_AB(adapter_dir, dtype=torch.float32):
    """{module_name -> (A, B)}; never forms BA."""
    f = os.path.join(adapter_dir, "adapter_model.safetensors")
    sd = load_safetensors(f, device="cpu")
    out = {}
    for key in sd:
        m = LORA_A_PAT.match(key)
        if not m:
            continue
        mod = m.group(1)
        bkey = key.replace("lora_A", "lora_B")
        if bkey not in sd:
            continue
        A = sd[key].to(dtype).contiguous()
        B = sd[bkey].to(dtype).contiguous()
        out[mod] = (A, B)
    return out


def ba_norm_sq(A, B):
    return ((B.T @ B) * (A @ A.T)).sum().item()


def ba_inner(A1, B1, A2, B2):
    return (A1 * ((B1.T @ B2) @ A2)).sum().item()


def cosine_per_module(d1, d2):
    cs = {}
    for n, (A1, B1) in d1.items():
        if n not in d2:
            continue
        A2, B2 = d2[n]
        n1 = ba_norm_sq(A1, B1)
        n2 = ba_norm_sq(A2, B2)
        denom = (n1 * n2) ** 0.5
        cs[n] = 0.0 if denom < 1e-30 else ba_inner(A1, B1, A2, B2) / denom
    return cs


# ----------------------------------------------------------------------------
# Trajectory comparisons.
# Each entry: (name, run_A, run_B, mode) where mode is one of:
#   "across_steps"  — for each step in run_A's checkpoint list, compute cosine
#                     between run_A@step and run_B@same-or-nearest step.
#                     Used for cross-seed and cross-config trajectories.
#   "within_to_final" — cos(run_A@step, run_A@final) at every step. Used to
#                       diagnose whether an adapter rotates during training
#                       or simply grows along a fixed axis.
# ----------------------------------------------------------------------------
COMPARISONS = [
    # ---- cross-seed trajectories ----
    ("E3_cross_seed",  "E3_s42",   "E3_s123",  "across_steps"),
    ("E5_cross_seed",  "E5_s42",   "E5_s123",  "across_steps"),
    ("E5b_cross_seed", "E5b_s42",  "E5b_s123", "across_steps"),
    ("E5c_cross_seed", "E5c_s42",  "E5c_s123", "across_steps"),
    ("E6_cross_seed",  "E6_s42",   "E6_s123",  "across_steps"),
    ("E6b_cross_seed", "E6b_s42",  "E6b_s123", "across_steps"),

    # ---- cross-config trajectories at r=64 (collapse vs healthy) ----
    ("E5_vs_E5c",  "E5_s42",  "E5c_s42",  "across_steps"),
    ("E5_vs_E5b",  "E5_s42",  "E5b_s42",  "across_steps"),
    ("E5b_vs_E5c", "E5b_s42", "E5c_s42",  "across_steps"),

    # ---- within-run rotation: how much does each run move from its own final? ----
    ("E5_to_final",  "E5_s42",  "E5_s42",  "within_to_final"),
    ("E5c_to_final", "E5c_s42", "E5c_s42", "within_to_final"),
    ("E6_to_final",  "E6_s42",  "E6_s42",  "within_to_final"),
]


def nearest_step_match(steps_a, steps_b):
    """Return [(step_a, step_b)] pairing each step in steps_a with the
    closest available step in steps_b. Useful when training durations differ
    (E5 has 5 epochs ~4885 steps; E5c has 3 epochs ~2931 steps)."""
    out = []
    for s in steps_a:
        sb = min(steps_b, key=lambda x: abs(x - s))
        out.append((s, sb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="./directional_results/trajectory_cosine.json")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    # Discover available checkpoints per run
    run_ckpts = {}
    for name, path in RUNS.items():
        cks = discover_checkpoints(path)
        if cks:
            run_ckpts[name] = cks
            print(f"  {name}: {len(cks)} points  steps={[s for s,_ in cks]}")
        else:
            print(f"  [SKIP] {name}: no checkpoints found at {path}")

    # ---- LRU cache of loaded adapters ----
    cache, order = {}, []
    CACHE_MAX = 4

    def get(adapter_dir):
        if adapter_dir in cache:
            order.remove(adapter_dir)
            order.append(adapter_dir)
            return cache[adapter_dir]
        t0 = time.time()
        d = load_AB(adapter_dir, dtype)
        cache[adapter_dir] = d
        order.append(adapter_dir)
        while len(order) > CACHE_MAX:
            evict = order.pop(0)
            del cache[evict]
            gc.collect()
        print(f"    loaded {os.path.basename(adapter_dir)} ({time.time()-t0:.1f}s)")
        return d

    results = {}
    for comp_name, a, b, mode in COMPARISONS:
        if a not in run_ckpts or b not in run_ckpts:
            print(f"  [SKIP] {comp_name}: missing run")
            continue
        ck_a = run_ckpts[a]
        ck_b = run_ckpts[b]
        pairs = []

        if mode == "across_steps":
            steps_a = [s for s, _ in ck_a]
            steps_b = [s for s, _ in ck_b]
            for sa, sb in nearest_step_match(steps_a, steps_b):
                pa = dict(ck_a)[sa]
                pb = dict(ck_b)[sb]
                pairs.append((sa, sb, pa, pb))
        elif mode == "within_to_final":
            final_path = ck_a[-1][1]
            for sa, pa in ck_a:
                pairs.append((sa, ck_a[-1][0], pa, final_path))

        print(f"\n{comp_name}: {len(pairs)} comparisons")
        traj = []
        for sa, sb, pa, pb in pairs:
            da = get(pa)
            db = get(pb)
            with torch.no_grad():
                cs = cosine_per_module(da, db)
            arr = np.array(list(cs.values()))
            entry = {
                "step_a": sa,
                "step_b": sb,
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "std": float(arr.std()),
                "frac_gt_0.5": float((arr > 0.5).sum() / len(arr)),
                "n_modules": len(arr),
            }
            traj.append(entry)
            print(f"  step ({sa},{sb}): mean={entry['mean']:+.4f}  "
                  f"median={entry['median']:+.4f}  "
                  f">0.5: {int(entry['frac_gt_0.5']*entry['n_modules'])}/{entry['n_modules']}")
        results[comp_name] = {
            "mode": mode,
            "run_a": a,
            "run_b": b,
            "trajectory": traj,
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()
