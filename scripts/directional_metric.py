#!/usr/bin/env python3
"""
directional_metric.py — Subspace alignment and projected-energy diagnostics
============================================================================

Computes two metrics that measure whether a LoRA adapter's weight updates
operate inside or outside the top-rank subspace of the pretrained weights.

For each target module with pretrained W0 (out x in) and LoRA delta dW = BA
(B is out x r, A is r x in, so dW has rank <= r):

  * V_W0  = top-r right singular vectors of W0   (in x r)
  * V_dW  = right singular vectors of dW          (in x r)
            (equal to V_A, the right singular vectors of A,
             because dW and A share the same row-space when B has full
             column-rank, which holds for trained LoRA adapters)

Metric A (subspace alignment):
    alignment = || V_dW^T V_W0 ||_F^2 / r          in [0, 1]
    1.0  = LoRA update lives entirely inside the pretrained top-r subspace
    0.0  = LoRA update is orthogonal to the pretrained top-r subspace

Metric B (projected energy ratio):
    P_k = V_W0 V_W0^T  (in-space projector onto top-r pretrained subspace)
    aligned_energy = || dW P_k ||_F^2  = || B (A V_W0) ||_F^2
    total_energy   = || dW ||_F^2
    ratio = aligned_energy / total_energy           in [0, 1]

Both metrics are scale-invariant in the LoRA update (alignment depends only
on subspace orientation; ratio is normalized by total LoRA energy). They
therefore capture *direction*, not magnitude.

Hypothesis (Block 1 of the Main-paper plan):
  * E5 catastrophic collapse  ->  LOW alignment / LOW ratio (orthogonal to W0)
  * E5c healthy r=64          ->  HIGH alignment / HIGH ratio (in-subspace)
  * Same SE trajectory in both => SE cannot see this, directional metric can.

Output: JSON per-adapter with per-module and aggregate alignment+ratio.
"""

import argparse
import json
import os
import sys
import time
import re
import numpy as np
import torch

from transformers import AutoModelForCausalLM
from peft import PeftModel

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", type=str, required=True,
                   help="Path to a LoRA adapter directory (PEFT format)")
    p.add_argument("--base_model", type=str, default="google/gemma-2-9b")
    p.add_argument("--output", type=str, required=True,
                   help="Output JSON path")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for SVD computations")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float32"],
                   help="Compute dtype")
    p.add_argument("--w0_cache_dir", type=str, default="./w0_svd_cache",
                   help="Directory to cache top-r SVD of W0 per module")
    p.add_argument("--w0_cache_rank", type=int, default=64,
                   help="Number of right singular vectors of W0 to cache "
                        "(should be >= LoRA rank in any adapter you analyse).")
    return p.parse_args()


def get_target_layer_modules(model):
    """Iterate over (module_name, weight_tensor) for LoRA target modules."""
    pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)$")
    for name, mod in model.named_modules():
        m = pattern.match(name)
        if not m:
            continue
        proj_name = m.group(3)
        if proj_name not in TARGET_MODULES:
            continue
        if hasattr(mod, "weight"):
            yield name, mod.weight


def topk_right_singular(W, k, device, dtype):
    """Return top-k right singular vectors of W (shape out x in).
    Returns V of shape in x k with orthonormal columns.

    SVD is computed in float32 for numerical accuracy and bnb/svd_lowrank
    dtype compatibility, then cast back to the requested dtype for storage.
    """
    # svd_lowrank's internal random projection is fp32, so the input matrix
    # must also be fp32 to avoid a dtype-mismatch in the matmul.
    W = W.to(device=device, dtype=torch.float32)
    q = min(W.shape[0], W.shape[1], k + 10)
    U, S, V = torch.svd_lowrank(W, q=q, niter=4)
    return V[:, :k].contiguous().to(dtype)


def lora_AB_from_adapter(adapter_model, module_name):
    """Extract A and B matrices for a given target module from a PeftModel.

    Returns (A, B) where A is r x in, B is out x r, or (None, None) if the
    module is not adapted.
    """
    # PEFT module naming: model.layers.N.<self_attn|mlp>.<proj>.lora_A.default
    base_name = f"base_model.model.{module_name}"
    a_name = f"{base_name}.lora_A.default"
    b_name = f"{base_name}.lora_B.default"
    state = adapter_model.state_dict()

    # PEFT keys include ".weight" suffix
    a_key = a_name + ".weight"
    b_key = b_name + ".weight"
    if a_key not in state or b_key not in state:
        return None, None
    return state[a_key], state[b_key]


def compute_module_metrics(W0, A, B, V_k, device, dtype):
    """Compute alignment and projected-energy ratio for one module.

    Args:
        W0: out x in (unused once V_k cached)
        A : r x in
        B : out x r
        V_k: in x r  (top-r right singular vectors of W0)
    Returns dict with alignment, projected_ratio, total_frob, lora_rank.
    """
    A = A.to(device=device, dtype=dtype)
    B = B.to(device=device, dtype=dtype)
    V_k = V_k.to(device=device, dtype=dtype)
    r = A.shape[0]

    # ----- Metric A: subspace alignment -----
    # Right singular vectors of dW = BA span the same subspace as the
    # right singular vectors of A (when B has full column rank, which holds
    # for trained LoRA adapters). Use SVD of A directly.
    # A is r x in; we want V_A of shape in x r.
    # Use full SVD of A (small matrix).
    A32 = A.to(torch.float32)
    U_A, S_A, Vh_A = torch.linalg.svd(A32, full_matrices=False)
    # Vh_A shape: r x in. V_A = Vh_A.T  shape: in x r
    V_A = Vh_A.T  # in x r

    # alignment = || V_A^T V_k ||_F^2 / r
    M = (V_A.T.to(dtype) @ V_k)  # r x r
    alignment = (M.float() ** 2).sum().item() / r

    # ----- Metric B: projected energy ratio -----
    # aligned = || dW P_k ||_F^2 = || B (A V_k) ||_F^2
    A_proj = A @ V_k                      # r x r
    aligned = B @ A_proj                  # out x r
    aligned_energy = (aligned.float() ** 2).sum().item()

    # total = || dW ||_F^2 = || B A ||_F^2 ; compute via trace identities
    # for memory safety: || BA ||_F^2 = trace(A^T B^T B A) = trace((B^T B)(A A^T))
    BtB = B.T @ B                          # r x r
    AAt = A @ A.T                          # r x r
    total_energy = (BtB.float() * AAt.float().T).sum().item()
    # equivalent: torch.trace(BtB @ AAt).item()

    ratio = aligned_energy / total_energy if total_energy > 0 else 0.0

    return {
        "alignment": alignment,
        "projected_ratio": ratio,
        "aligned_energy": aligned_energy,
        "total_energy": total_energy,
        "lora_rank": r,
    }


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    os.makedirs(args.w0_cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"[INFO] adapter: {args.adapter_path}")
    print(f"[INFO] base:    {args.base_model}")
    print(f"[INFO] device:  {args.device}, dtype: {dtype}")

    # ---- Load base in BF16 (no quantization) for clean SVD reference ----
    print("[INFO] loading base model in BF16...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager",
    )
    base.eval()

    # ---- Cache or load top-k SVD of W0 per module ----
    cache_path = os.path.join(args.w0_cache_dir,
                              f"w0_topk_r{args.w0_cache_rank}.pt")
    if os.path.isfile(cache_path):
        print(f"[INFO] loading cached W0 SVD from {cache_path}")
        V_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        print(f"[INFO] computing top-{args.w0_cache_rank} right singular "
              f"vectors of W0 for each target module (one-time)...")
        V_cache = {}
        n_done = 0
        t0 = time.time()
        for name, W in get_target_layer_modules(base):
            with torch.no_grad():
                Vk = topk_right_singular(W, args.w0_cache_rank,
                                         args.device, dtype)
            V_cache[name] = Vk.to("cpu", dtype=torch.bfloat16)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  [{n_done}] {name} ({time.time()-t0:.1f}s elapsed)")
        torch.save(V_cache, cache_path)
        print(f"[INFO] cached W0 SVD: {n_done} modules in "
              f"{time.time()-t0:.1f}s -> {cache_path}")

    # ---- Load adapter on top of base ----
    print(f"[INFO] loading adapter from {args.adapter_path}...")
    adapter = PeftModel.from_pretrained(base, args.adapter_path)
    adapter.eval()

    # ---- Iterate modules and compute metrics ----
    per_module = {}
    layer_pat = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)$")

    n_done = 0
    skipped = 0
    t0 = time.time()
    for name in list(V_cache.keys()):
        m = layer_pat.match(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        proj_name = m.group(3)

        A, B = lora_AB_from_adapter(adapter, name)
        if A is None or B is None:
            skipped += 1
            continue

        # Use the lora rank actually present in the adapter
        r_adapter = A.shape[0]
        if r_adapter > args.w0_cache_rank:
            print(f"[WARN] adapter rank {r_adapter} > cached rank "
                  f"{args.w0_cache_rank}; using cached rank.")
            r = args.w0_cache_rank
        else:
            r = r_adapter

        V_k = V_cache[name][:, :r]
        with torch.no_grad():
            metrics = compute_module_metrics(None, A, B, V_k,
                                              args.device, dtype)
        metrics["layer_idx"] = layer_idx
        metrics["module"] = proj_name
        per_module[name] = metrics
        n_done += 1
        if n_done % 40 == 0:
            print(f"  [{n_done}] {name}  align={metrics['alignment']:.3f}  "
                  f"ratio={metrics['projected_ratio']:.3f}  "
                  f"({time.time()-t0:.1f}s)")

    print(f"[INFO] processed {n_done} modules, skipped {skipped}, "
          f"{time.time()-t0:.1f}s")

    # ---- Aggregate ----
    aligns = [m["alignment"] for m in per_module.values()]
    ratios = [m["projected_ratio"] for m in per_module.values()]
    aligned_tot = sum(m["aligned_energy"] for m in per_module.values())
    total_tot = sum(m["total_energy"] for m in per_module.values())

    aggregate = {
        "alignment_mean": float(np.mean(aligns)) if aligns else None,
        "alignment_median": float(np.median(aligns)) if aligns else None,
        "alignment_min": float(np.min(aligns)) if aligns else None,
        "alignment_max": float(np.max(aligns)) if aligns else None,
        "projected_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "projected_ratio_median": float(np.median(ratios)) if ratios else None,
        "projected_ratio_min": float(np.min(ratios)) if ratios else None,
        "projected_ratio_max": float(np.max(ratios)) if ratios else None,
        "global_aligned_energy": aligned_tot,
        "global_total_energy": total_tot,
        "global_projected_ratio": (aligned_tot / total_tot
                                   if total_tot > 0 else None),
        "n_modules": len(per_module),
    }

    out = {
        "adapter_path": args.adapter_path,
        "base_model": args.base_model,
        "lora_rank_observed": (per_module[next(iter(per_module))]["lora_rank"]
                               if per_module else None),
        "aggregate": aggregate,
        "per_module": per_module,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] wrote {args.output}")
    print(f"[SUMMARY] alignment mean = {aggregate['alignment_mean']:.4f}")
    print(f"[SUMMARY] projected_ratio mean = "
          f"{aggregate['projected_ratio_mean']:.4f}")
    print(f"[SUMMARY] global projected_ratio = "
          f"{aggregate['global_projected_ratio']:.4f}")


if __name__ == "__main__":
    main()
