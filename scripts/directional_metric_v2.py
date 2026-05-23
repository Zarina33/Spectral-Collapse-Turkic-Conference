#!/usr/bin/env python3
"""
directional_metric_v2.py — Alternative directional diagnostics for LoRA.

The top-r subspace alignment in directional_metric.py was non-discriminative
across our experiments (all values clustered at r/n_in random baseline).
This script tries three alternative direction-only metrics:

  1. Frobenius cosine similarity between ΔW = BA and W₀:
        cos(ΔW, W₀) = <ΔW, W₀>_F / (||ΔW||_F * ||W₀||_F)
     Range [-1, 1]. Positive = LoRA update is aligned with pretrained map
     (reinforces existing structure); near 0 = orthogonal; negative = cancels.

  2. Top-k subspace alignment for k = 256 (much larger than LoRA rank):
        alignment_k = || V_dW^T V_k(W₀) ||_F^2 / r
     If LoRA learns inside W₀'s top-256 (not top-r), this metric will
     discriminate while top-r does not.

  3. Top-k subspace energy: fraction of dW energy in top-k W₀ subspace:
        ratio_k = || dW V_k V_k^T ||_F^2 / || dW ||_F^2
     Range [0, 1]. Tells us where in W₀'s spectrum the LoRA acts.

Output: JSON with all three metrics, per-module and aggregate.
"""

import argparse
import gc
import json
import os
import re
import time
import numpy as np
import torch

from transformers import AutoModelForCausalLM
from peft import PeftModel

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", type=str, action="append", default=[],
                   help="Path to adapter. Repeat flag for multiple adapters.")
    p.add_argument("--output", type=str, action="append", default=[],
                   help="Output JSON path. Must match --adapter_path 1:1.")
    p.add_argument("--base_model", type=str, default="google/gemma-2-9b")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--top_k_large", type=int, default=256,
                   help="k for the larger-subspace alignment metric")
    p.add_argument("--w0_cache_dir", type=str, default="./w0_svd_cache_v2")
    args = p.parse_args()
    if len(args.adapter_path) != len(args.output):
        p.error("--adapter_path and --output must be supplied the same "
                "number of times")
    if not args.adapter_path:
        p.error("at least one --adapter_path / --output pair required")
    return args


def topk_right_singular(W, k, device):
    W = W.to(device=device, dtype=torch.float32)
    q = min(W.shape[0], W.shape[1], k + 10)
    U, S, V = torch.svd_lowrank(W, q=q, niter=4)
    return V[:, :k].contiguous()


def get_target_modules(model):
    pat = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)$")
    for name, mod in model.named_modules():
        m = pat.match(name)
        if not m or m.group(3) not in TARGET_MODULES:
            continue
        if hasattr(mod, "weight"):
            yield name, mod.weight


def lora_AB(adapter_model, module_name):
    sd = adapter_model.state_dict()
    a = sd.get(f"base_model.model.{module_name}.lora_A.default.weight")
    b = sd.get(f"base_model.model.{module_name}.lora_B.default.weight")
    return a, b


def compute_metrics_v2(W0_full, A, B, V_k_large, device):
    """Three direction metrics for one module."""
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)
    V_k = V_k_large.to(device=device, dtype=torch.float32)
    W0 = W0_full.to(device=device, dtype=torch.float32)
    r = A.shape[0]

    # ---- Metric 1: Frobenius cosine(BA, W0) ----
    # <BA, W0>_F = trace((BA)^T W0) = trace(A^T B^T W0)
    BtW0 = B.T @ W0          # r x in
    inner = (A * BtW0).sum().item()  # trace(A^T (B^T W0)) = sum elementwise of A * BtW0

    # ||BA||_F^2 = trace(A^T B^T B A) = trace((B^T B)(A A^T))
    BtB = B.T @ B            # r x r
    AAt = A @ A.T            # r x r
    BA_norm_sq = (BtB * AAt.T).sum().item()
    BA_norm = max(BA_norm_sq, 1e-20) ** 0.5

    # ||W0||_F^2
    W0_norm = (W0 * W0).sum().item() ** 0.5
    cos_BA_W0 = inner / (BA_norm * W0_norm)

    # ---- Metric 2: alignment with top-k(W0) subspace ----
    # V_A (in x r) from SVD of A
    U_A, S_A, Vh_A = torch.linalg.svd(A, full_matrices=False)
    V_A = Vh_A.T  # in x r

    # alignment_k = || V_A^T V_k ||_F^2 / r   (now k can be > r)
    M = V_A.T @ V_k  # r x k
    alignment_k = (M * M).sum().item() / r

    # ---- Metric 3: projected energy onto top-k(W0) ----
    # || BA V_k ||_F^2 / || BA ||_F^2
    # = || B (A V_k) ||_F^2 / BA_norm_sq
    AVk = A @ V_k             # r x k
    projected = B @ AVk        # out x k
    projected_energy = (projected * projected).sum().item()
    energy_ratio_k = projected_energy / max(BA_norm_sq, 1e-20)

    return {
        "cos_BA_W0": cos_BA_W0,
        "alignment_topk": alignment_k,
        "energy_ratio_topk": energy_ratio_k,
        "lora_rank": r,
        "ba_frob": BA_norm,
        "w0_frob": W0_norm,
    }


def get_base_weight(base, module_name):
    """Walk the model tree to return W0 for module_name.

    Handles both unwrapped base and PEFT-wrapped modules (which keep the
    original weight at `.base_layer.weight`)."""
    mod = base
    for part in module_name.split("."):
        mod = getattr(mod, part)
    if hasattr(mod, "base_layer"):
        return mod.base_layer.weight
    return mod.weight


def run_one_adapter(base, V_cache, adapter_path, output_path, args):
    print(f"[INFO] loading adapter from {adapter_path}...")
    adapter = PeftModel.from_pretrained(base, adapter_path)
    adapter.eval()

    per_module = {}
    pat = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)$")
    n_done = 0
    t0 = time.time()
    for name in V_cache:
        m = pat.match(name)
        if not m:
            continue
        A, B = lora_AB(adapter, name)
        if A is None:
            continue
        W0 = get_base_weight(base, name)
        with torch.no_grad():
            metrics = compute_metrics_v2(W0, A, B,
                                          V_cache[name], args.device)
        metrics["layer_idx"] = int(m.group(1))
        metrics["module"] = m.group(3)
        per_module[name] = metrics
        n_done += 1
        if n_done % 60 == 0:
            print(f"  [{n_done}] {name}  cos={metrics['cos_BA_W0']:.4f}  "
                  f"align_k={metrics['alignment_topk']:.4f}  "
                  f"ratio_k={metrics['energy_ratio_topk']:.4f}")

    print(f"[INFO] {n_done} modules, {time.time()-t0:.1f}s")

    cos_vals = [m["cos_BA_W0"] for m in per_module.values()]
    align_vals = [m["alignment_topk"] for m in per_module.values()]
    ratio_vals = [m["energy_ratio_topk"] for m in per_module.values()]

    aggregate = {
        "cos_BA_W0_mean": float(np.mean(cos_vals)),
        "cos_BA_W0_median": float(np.median(cos_vals)),
        "cos_BA_W0_abs_mean": float(np.mean(np.abs(cos_vals))),
        "alignment_topk_mean": float(np.mean(align_vals)),
        "alignment_topk_median": float(np.median(align_vals)),
        "energy_ratio_topk_mean": float(np.mean(ratio_vals)),
        "energy_ratio_topk_median": float(np.median(ratio_vals)),
        "n_modules": len(per_module),
        "top_k_large": args.top_k_large,
    }

    out = {"adapter_path": adapter_path, "aggregate": aggregate,
           "per_module": per_module}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[SUMMARY {os.path.basename(adapter_path) or adapter_path}] "
          f"cos(BA, W0): mean={aggregate['cos_BA_W0_mean']:+.5f}  "
          f"|·|-mean={aggregate['cos_BA_W0_abs_mean']:.5f}")
    print(f"[SUMMARY] align top-{args.top_k_large}:   "
          f"mean={aggregate['alignment_topk_mean']:.4f}  "
          f"random≈{args.top_k_large/3584:.4f}")
    print(f"[SUMMARY] energy top-{args.top_k_large}:  "
          f"mean={aggregate['energy_ratio_topk_mean']:.4f}  "
          f"random≈{args.top_k_large/3584:.4f}")

    # Restore the underlying base model so the next adapter can be loaded
    # cleanly. unload() removes the LoRA wrappers and returns base.
    restored = adapter.unload()
    del adapter
    gc.collect()
    return restored


def main():
    args = parse_args()
    os.makedirs(args.w0_cache_dir, exist_ok=True)

    print(f"[INFO] {len(args.adapter_path)} adapter(s) to process")
    print(f"[INFO] top_k_large: {args.top_k_large}")

    # Load base in BF16 (kept in memory for the whole run — no W_cache duplicate)
    print("[INFO] loading base model in BF16...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    base.eval()

    # Cache only the small top-k right singular vectors of W0 (≈600 MB).
    cache_path = os.path.join(args.w0_cache_dir,
                              f"w0_topk_r{args.top_k_large}.pt")

    if os.path.isfile(cache_path):
        print(f"[INFO] loading cached W0 top-{args.top_k_large} SVD")
        V_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        n_targets = sum(1 for _ in get_target_modules(base))
        print(f"[INFO] computing top-{args.top_k_large} W0 SVD for "
              f"{n_targets} modules")
        V_cache = {}
        t0 = time.time()
        for i, (name, W) in enumerate(get_target_modules(base)):
            with torch.no_grad():
                Vk = topk_right_singular(W, args.top_k_large, args.device)
            V_cache[name] = Vk.to("cpu", dtype=torch.bfloat16)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n_targets}] {name} ({time.time()-t0:.1f}s)")
        torch.save(V_cache, cache_path)
        print(f"[INFO] cached in {time.time()-t0:.1f}s")

    for adapter_path, output_path in zip(args.adapter_path, args.output):
        base = run_one_adapter(base, V_cache, adapter_path, output_path, args)


if __name__ == "__main__":
    main()
