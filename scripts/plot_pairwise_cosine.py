#!/usr/bin/env python3
"""
plot_pairwise_cosine.py — Figure 10 for the directional-geometry section.

Two panels:
  (a) Cross-seed cosines for 8 same-configuration pairs (E3/E5/E5b/E5c/E6/E6b/E8/E1b).
      Transfer-initialised configurations (E6, E6b) stand apart from direct training.
  (b) Per-module cosine distributions for four representative pairs, showing that
      the bimodality is reproduced module-by-module, not only on the mean.

Usage:
    python scripts/plot_pairwise_cosine.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT  = "directional_results/pairwise_cosine.json"
OUTPUT = "figures/fig10_pairwise_cosine.png"


def main():
    with open(INPUT) as f:
        data = json.load(f)

    # ---- Panel A: 8 cross-seed bars ----
    cross_seed_keys = [
        ("E1b_s42__VS__E1b_s123",  "E1b (KZ, $r{=}16$)"),
        ("E3_s42__VS__E3_s123",    "E3 (KY, $r{=}16$)"),
        ("E5_s42__VS__E5_s123",    "E5 (KY collapse, $r{=}64$, lr$=5e{-}4$)"),
        ("E5b_s42__VS__E5b_s123",  "E5b (KY, $r{=}64$, lr$=2e{-}4$, 5ep)"),
        ("E5c_s42__VS__E5c_s123",  "E5c (KY, $r{=}64$, lr$=2e{-}4$, 3ep)"),
        ("E8_s42__VS__E8_s123",    "E8 (EN, $r{=}16$)"),
        ("E6b_s42__VS__E6b_s123",  "E6b (EN$\\rightarrow$KY transfer)"),
        ("E6_s42__VS__E6_s123",    "E6 (KZ$\\rightarrow$KY transfer)"),
    ]

    means   = [data[k]["mean"]        for k, _ in cross_seed_keys]
    medians = [data[k]["median"]      for k, _ in cross_seed_keys]
    fracs   = [data[k]["frac_gt_0.5"] for k, _ in cross_seed_keys]
    labels  = [lab for _, lab in cross_seed_keys]
    # transfer-init configurations to highlight
    transfer_mask = ["E6_" in k or "E6b_" in k for k, _ in cross_seed_keys]

    # ---- Panel B: per-module histograms for 4 selected pairs ----
    select_keys = [
        ("E6_s42__VS__E6_s123",   "E6 vs E6 (KZ$\\rightarrow$KY, cross-seed)",   "#2c8a4a"),
        ("E6b_s42__VS__E6b_s123", "E6b vs E6b (EN$\\rightarrow$KY, cross-seed)", "#88aa44"),
        ("E5c_s42__VS__E5c_s123", "E5c vs E5c (healthy direct, cross-seed)",     "#999999"),
        ("E5_s42__VS__E5c_s42",   "E5 vs E5c (collapse vs healthy, same seed)",  "#c0392b"),
    ]

    # ---- Build figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})

    # Panel A
    ax = axes[0]
    y_pos = np.arange(len(labels))
    colors = ["#2c8a4a" if t else "#666666" for t in transfer_mask]
    bars = ax.barh(y_pos, means, color=colors, edgecolor="black", linewidth=0.6)
    for i, (m, f) in enumerate(zip(means, fracs)):
        if m < 0.5:
            ax.text(m + 0.01, i, f"{m:.3f}  ({int(f * 294)}/294)",
                    va="center", fontsize=8.5)
        else:
            ax.text(m - 0.01, i, f"{m:.3f}  ({int(f * 294)}/294)",
                    va="center", ha="right", fontsize=8.5, color="white",
                    fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean per-module cosine $\\cos(\\Delta W_X, \\Delta W_Y)$",
                  fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title("(a) Cross-seed cosines for 8 configurations\n"
                 "transfer-init (green) vs.\\ direct (grey)",
                 fontsize=10.5)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Panel B
    ax = axes[1]
    for key, lab, col in select_keys:
        vals = list(data[key]["per_module"].values())
        ax.hist(vals, bins=40, range=(-0.2, 1.0), histtype="step",
                linewidth=1.8, label=lab, color=col)
    ax.set_xlabel("Per-module cosine", fontsize=10)
    ax.set_ylabel("Number of modules (out of 294)", fontsize=10)
    ax.set_xlim(-0.2, 1.0)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_title("(b) Per-module distributions for 4 representative pairs",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper center")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    plt.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
