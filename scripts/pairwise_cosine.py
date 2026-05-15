#!/usr/bin/env python3
"""
pairwise_cosine.py — Per-module Frobenius cosine between LoRA adapter updates.

For each pair of adapters (A, B), compute per-target-module cosine similarity
between their flattened delta-W matrices:
    cos_m = <BA_A, BA_B>_F /
            (||BA_A||_F * ||BA_B||_F)

This is a direction-only diagnostic: it removes scale (Frobenius norm) and
shape (singular-value distribution) effects, leaving only "are the two
adapters going in the same direction in weight space".

Key hypothesis we are testing: catastrophic functional collapse (E5) and
healthy training at the same rank (E5c) live in essentially orthogonal
regions of LoRA-update space, even though SE and Frobenius norms are
similar throughout training.

Memory-efficient implementation: we *never* form the full out x in BA
matrix. Instead, given the LoRA factors (A: r x in, B: out x r), we compute
all Frobenius products through r x r intermediates:

    <B1 A1, B2 A2>_F = trace(A1^T B1^T B2 A2)
    ||BA||_F^2       = trace((B^T B)(A A^T))

This drops per-adapter memory from ~tens of GB to ~100 MB even at r=64.

Output:
  - JSON with mean/median/min/max cosine and the full per-module array
    for every requested pair.
  - Plain-text summary printed to stdout.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str,
                   default="./directional_results/pairwise_cosine.json")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "bfloat16"],
                   help="dtype for matmul. float32 is safer for very small "
                        "cosines; bf16 halves memory.")
    return p.parse_args()


def find_adapter_file(path):
    """Return path to adapter weights (safetensors preferred)."""
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        f = os.path.join(path, name)
        if os.path.isfile(f):
            return f
    raise FileNotFoundError(f"no adapter weights in {path}")


def load_AB(path, dtype):
    """Return dict {module_name -> (A, B)} as CPU tensors in `dtype`.

    A has shape (r, in); B has shape (out, r). We do not form BA here.
    """
    weights_path = find_adapter_file(path)
    if weights_path.endswith(".safetensors"):
        sd = load_safetensors(weights_path, device="cpu")
    else:
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)

    out = {}
    for key in sd:
        m = LORA_A_PAT.match(key)
        if not m:
            continue
        mod_name = m.group(1)
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in sd:
            continue
        A = sd[key].to(dtype=dtype)
        B = sd[b_key].to(dtype=dtype)
        out[mod_name] = (A.contiguous(), B.contiguous())
    return out


def ba_norm_sq(A, B):
    """||BA||_F^2 = trace((B^T B)(A A^T)) via r x r intermediates."""
    BtB = B.T @ B          # r x r
    AAt = A @ A.T          # r x r
    # trace(BtB @ AAt) = sum elementwise of BtB * AAt^T; both are symmetric.
    return (BtB * AAt).sum().item()


def ba_inner(A1, B1, A2, B2):
    """<B1 A1, B2 A2>_F = trace(A1^T B1^T B2 A2) without forming BA."""
    M = B1.T @ B2          # r1 x r2
    N = M @ A2             # r1 x in
    # trace(A1^T N) = elementwise sum of A1 * N
    return (A1 * N).sum().item()


def cosine_per_module(d1, d2):
    """{module -> cosine} for modules common to both adapters."""
    cs = {}
    for n, (A1, B1) in d1.items():
        if n not in d2:
            continue
        A2, B2 = d2[n]
        n1_sq = ba_norm_sq(A1, B1)
        n2_sq = ba_norm_sq(A2, B2)
        denom = (n1_sq * n2_sq) ** 0.5
        if denom < 1e-30:
            cs[n] = 0.0
            continue
        inner = ba_inner(A1, B1, A2, B2)
        cs[n] = inner / denom
    return cs


ADAPTERS = {
    "E1_s42":    "./output_kz_baseline_r16_lr2e4_3ep/final_adapter",
    "E1b_s42":   "./output_kz_tokenmatched_r16_lr2e4_3ep/final_adapter",
    "E1b_s123":  "./output_kz_tokenmatched_r16_lr2e4_3ep_seed123/final_adapter",
    "E2_s42":    "./output_uz_baseline_r16_lr2e4_3ep/final_adapter",
    "E3_s42":    "./output_ky_baseline_r16_lr2e4_3ep/final_adapter",
    "E3_s123":   "./output_ky_baseline_r16_lr2e4_3ep_seed123/final_adapter",
    "E4_s42":    "./output_ky_overfit_r16_lr2e4_10ep/final_adapter",
    "E5_s42":    "./output_ky_collapse_r64_lr5e4_5ep/final_adapter",
    "E5_s123":   "./output_ky_collapse_r64_lr5e4_5ep_seed123/final_adapter",
    "E5b_s42":   "./output_ky_r64_lr2e4_5ep/final_adapter",
    "E5b_s123":  "./output_ky_r64_lr2e4_5ep_seed123/final_adapter",
    "E5c_s42":   "./output_ky_r64_lr2e4_3ep/final_adapter",
    "E5c_s123":  "./output_ky_r64_lr2e4_3ep_seed123/final_adapter",
    "E6_s42":    "./output_ky_from_kz_r16_lr2e4_3ep/final_adapter",
    "E6_s123":   "./output_ky_from_kz_r16_lr2e4_3ep_seed123/final_adapter",
    "E6b_s42":   "./output_en_to_ky_r16_lr2e4_3ep/final_adapter",
    "E6b_s123":  "./output_en_to_ky_r16_lr2e4_3ep_seed123/final_adapter",
    "E8_s42":    "./output_en_baseline_r16_lr2e4_3ep/final_adapter",
    "E8_s123":   "./output_en_baseline_r16_lr2e4_3ep_seed123/final_adapter",
    "C3_s42":    "./output_ky_bf16_r16_lr2e4_3ep",
    "alpha32_s42": "./output_ky_r64_lr2e4_3ep_alpha32/final_adapter",
}

PAIRS = [
    # SAME-SETUP SEED BASELINES (expected cos ~ 1 if training converges to same basin)
    ("E3_s42",  "E3_s123",  "r=16 KY baseline: only seed differs (same-setup baseline)"),
    ("E5_s42",  "E5_s123",  "r=64 LR=5e-4 KY collapse: only seed differs"),
    ("E5b_s42", "E5b_s123", "r=64 LR=2e-4 5ep KY: only seed differs"),
    ("E5c_s42", "E5c_s123", "r=64 LR=2e-4 3ep KY: only seed differs"),
    ("E6_s42",  "E6_s123",  "KZ->KY transfer: only seed differs"),
    ("E6b_s42", "E6b_s123", "EN->KY transfer: only seed differs"),
    ("E8_s42",  "E8_s123",  "EN baseline: only seed differs"),
    ("E1b_s42", "E1b_s123", "KZ small-corpus: only seed differs"),

    # SAME-RANK COMPARISONS (r=64)
    ("E5_s42",  "E5b_s42",  "r=64 collapse vs healthy (only LR differs: 5e-4 vs 2e-4)"),
    ("E5_s42",  "E5c_s42",  "r=64 collapse vs healthy epoch-matched (LR+epoch differ)"),
    ("E5b_s42", "E5c_s42",  "r=64 healthy 5ep vs healthy 3ep (only epochs differ)"),
    ("E5b_s42", "E5c_s123", "r=64 5ep s=42 vs 3ep s=123 (different seed too)"),

    # CROSS-LANGUAGE r=16 BASELINES
    ("E1_s42",  "E3_s42",   "KZ baseline vs KY baseline (different languages)"),
    ("E1_s42",  "E2_s42",   "KZ baseline vs UZ baseline"),
    ("E3_s42",  "E2_s42",   "KY baseline vs UZ baseline"),

    # TRANSFER COMPARISONS
    ("E3_s42",  "E6_s42",   "direct KY vs KZ->KY transfer (related)"),
    ("E3_s42",  "E6b_s42",  "direct KY vs EN->KY transfer (cross-family)"),
    ("E6_s42",  "E6b_s42",  "KZ->KY vs EN->KY (related vs cross-family)"),

    # BF16 vs 4-BIT
    ("E3_s42",  "C3_s42",   "4-bit KY baseline vs BF16 KY baseline"),

    # COLLAPSE-AS-OWN-BASIN PROBE
    ("E5_s42",  "E3_s42",   "r=64 collapse vs r=16 baseline (different rank too)"),
    ("E5_s42",  "E1_s42",   "r=64 collapse vs KZ baseline"),
    ("E5_s42",  "E8_s42",   "r=64 collapse vs EN baseline"),

    # ALPHA=32 CONTROL (Block 4a, L3 closure)
    ("E5c_s42",   "alpha32_s42", "r=64 alpha/r=2 (E5c, alpha=128) vs alpha/r=0.5 (alpha=32)"),
    ("alpha32_s42", "E3_s42",    "r=64 alpha=32 (matched-effective-LR) vs r=16 baseline (alpha=32)"),
    ("alpha32_s42", "E5_s42",    "r=64 alpha=32 vs r=64 collapse (alpha=128)"),
]


def main():
    args = parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    # Reuse adapters across consecutive pairs via a tiny LRU-style cache.
    # Two slots cover any single pair; we hold up to 3 so that a common
    # adapter (e.g. E5_s42) is not reloaded between adjacent pairs that share it.
    cache = {}
    cache_order = []
    CACHE_MAX = 3

    def get(name):
        if name in cache:
            cache_order.remove(name)
            cache_order.append(name)
            return cache[name]
        path = ADAPTERS.get(name)
        if path is None or not os.path.isdir(path):
            return None
        print(f"  loading {name}...")
        t0 = time.time()
        d = load_AB(path, dtype)
        print(f"    {name}: {len(d)} modules in {time.time()-t0:.1f}s")
        cache[name] = d
        cache_order.append(name)
        while len(cache_order) > CACHE_MAX:
            evict = cache_order.pop(0)
            del cache[evict]
            gc.collect()
        return d

    results = {}
    rows = []
    for a, b, desc in PAIRS:
        da = get(a)
        db = get(b)
        if da is None or db is None:
            print(f"  [SKIP] {a} vs {b}: adapter missing")
            continue

        with torch.no_grad():
            cs_dict = cosine_per_module(da, db)

        arr = np.array(list(cs_dict.values()))
        summary = {
            "pair": [a, b],
            "description": desc,
            "n_modules": len(arr),
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "std":    float(arr.std()),
            "frac_gt_0.5": float((arr > 0.5).sum() / len(arr)),
            "frac_gt_0.9": float((arr > 0.9).sum() / len(arr)),
            "frac_lt_0":   float((arr < 0).sum() / len(arr)),
            "per_module":  cs_dict,
        }
        results[f"{a}__VS__{b}"] = summary
        rows.append((a, b, summary, desc))
        print(f"  {a:10s} vs {b:10s}  mean={summary['mean']:+.4f}  "
              f"median={summary['median']:+.4f}  "
              f"max={summary['max']:+.4f}  >0.5: "
              f"{int(summary['frac_gt_0.5']*len(arr))}/{len(arr)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] wrote {args.output}")

    print("\n" + "=" * 96)
    print(f"{'Pair':<25} {'mean':>8} {'median':>8} {'max':>7} {'>0.5':>8}   description")
    print("-" * 96)
    for a, b, s, desc in rows:
        n = s["n_modules"]
        print(f"{a+' vs '+b:<25} {s['mean']:+8.4f} {s['median']:+8.4f} "
              f"{s['max']:+7.4f} {int(s['frac_gt_0.5']*n):>3}/{n:<3}  {desc}")


if __name__ == "__main__":
    main()
