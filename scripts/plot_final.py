#!/usr/bin/env python3
"""
plot_final.py — Publication-quality figures for Spectral Collapse paper
=====================================================================
Generates:
  1. fig1_eval_summary.png         — Bar chart: PPL / NER F1 / TUMLU across experiments
  2. fig2_svd_dynamics.png         — SE over training progress: baseline vs collapse vs transfer
  3. fig3_layer_heatmap.png        — Heatmap of SE per layer at final step
  4. fig4_cross_ppl.png            — Cross-lingual PPL degradation matrix
  5. fig5_effective_rank.png       — Effective rank dynamics comparison
  6. fig6_frobenius_dynamics.png   — FrobNorm growth (key finding)
  7. fig7_overfit_paradox.png      — Train loss vs eval PPL dual-axis
  8. fig8_singular_values.png      — Singular value distributions (histograms)
  9. fig9_frobnorm_vs_ppl.png      — FrobNorm growth vs cross-lingual PPL scatter

Usage:
    python scripts/plot_final.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ──────────────────────────────────────────────────────────
EXPERIMENTS = {
    "EN baseline":    "output_en_baseline_r16_lr2e4_3ep",
    "KZ baseline":    "output_kz_baseline_r16_lr2e4_3ep",
    "UZ baseline":    "output_uz_baseline_r16_lr2e4_3ep",
    "KY baseline":    "output_ky_baseline_r16_lr2e4_3ep",
    "KY 10ep":        "output_ky_overfit_r16_lr2e4_10ep",
    "KY r=64":        "output_ky_collapse_r64_lr5e4_5ep",
    "KZ→KY transfer": "output_ky_from_kz_r16_lr2e4_3ep",
    "EN→KY transfer": "output_en_to_ky_r16_lr2e4_3ep",
}

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
COLORS = {
    "EN baseline":    "#607D8B",
    "KZ baseline":    "#2196F3",
    "UZ baseline":    "#4CAF50",
    "KY baseline":    "#FF9800",
    "KY 10ep":        "#795548",
    "KY r=64":        "#F44336",
    "KZ→KY transfer": "#9C27B0",
    "EN→KY transfer": "#00ACC1",
}

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

LINE_WIDTH = 2.5


def read_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def normalize_steps(steps):
    """Convert absolute step numbers to training progress percentage (0–100)."""
    max_step = max(steps)
    return [s / max_step * 100.0 for s in steps]


def suptitle_tight(fig, title, fontsize=14):
    """Add suptitle and adjust layout so it stays within the saved figure."""
    fig.suptitle(title, fontsize=fontsize, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])


# ════════════════════════════════════════════════════════════════════
# Figure 1: Evaluation Summary (PPL, NER F1, TUMLU)
# ════════════════════════════════════════════════════════════════════
def fig1_eval_summary():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    exp_names = list(EXPERIMENTS.keys())
    exp_colors = [COLORS[n] for n in exp_names]

    # Load eval reports
    reports = {}
    for name, path in EXPERIMENTS.items():
        rpath = os.path.join(path, "eval", "eval_report.json")
        if os.path.isfile(rpath):
            reports[name] = json.load(open(rpath))

    target_lang = {
        "EN baseline": "kz", "KZ baseline": "kz", "UZ baseline": "uz", "KY baseline": "ky",
        "KY 10ep": "ky", "KY r=64": "ky", "KZ→KY transfer": "ky", "EN→KY transfer": "ky",
    }

    # --- Panel A: Target-language PPL ---
    ax = axes[0]
    ppls = []
    for name in exp_names:
        lang = target_lang[name]
        ppl = reports.get(name, {}).get("perplexity", {}).get(lang, {}).get("ppl", 0)
        ppls.append(ppl)

    bars = ax.bar(range(len(exp_names)), ppls, color=exp_colors, edgecolor="white", width=0.65)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Perplexity")
    ax.set_title("A) Target-Language PPL ↓")
    max_ppl = max(ppls) if ppls else 1
    ax.set_ylim(0, max_ppl * 1.22)
    for bar, val in zip(bars, ppls):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_ppl * 0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # --- Panel B: NER F1 (target language) ---
    ax = axes[1]
    ner_f1s = []
    for name in exp_names:
        lang = target_lang[name]
        f1 = reports.get(name, {}).get("ner_wikiann", {}).get(lang, {}).get("f1", 0)
        ner_f1s.append(f1)

    bars = ax.bar(range(len(exp_names)), ner_f1s, color=exp_colors, edgecolor="white", width=0.65)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("B) NER F1 (target lang) ↑")
    max_f1 = max(ner_f1s) if max(ner_f1s) > 0 else 1
    ax.set_ylim(0, max_f1 * 1.3)
    for bar, val in zip(bars, ner_f1s):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_f1 * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # --- Panel C: TUMLU accuracy (target language) ---
    ax = axes[2]
    tumlu_lang_map = {"kz": "kazakh", "uz": "uzbek", "ky": "kyrgyz"}
    tumlu_accs = []
    for name in exp_names:
        lang = target_lang[name]
        tl = tumlu_lang_map.get(lang, "kyrgyz")
        acc = reports.get(name, {}).get("tumlu_qa", {}).get(tl, {}).get("accuracy", 0)
        tumlu_accs.append(acc)

    bars = ax.bar(range(len(exp_names)), tumlu_accs, color=exp_colors, edgecolor="white", width=0.65)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("C) TUMLU QA Accuracy ↑")
    max_acc = max(tumlu_accs) if max(tumlu_accs) > 0 else 1
    ax.set_ylim(0, max(max_acc * 1.3, 0.35))
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")
    ax.legend(fontsize=9, loc="upper right")
    for bar, val in zip(bars, tumlu_accs):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_acc * 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    suptitle_tight(fig, "Evaluation Summary Across Experiments", fontsize=14)
    out = os.path.join(OUTPUT_DIR, "fig1_eval_summary.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ════════════════════════════════════════════════════════════════════
# Figure 2: SVD Spectral Energy Dynamics
# ════════════════════════════════════════════════════════════════════
def fig2_svd_dynamics():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Compare: EN vs KY baseline vs KY r=64 vs KZ→KY transfer
    compare = ["EN baseline", "KY baseline", "KY r=64", "KZ→KY transfer"]

    # --- Panel A: Mean SE over training progress ---
    ax = axes[0]
    for name in compare:
        svd_path = os.path.join(EXPERIMENTS[name], "svd_log.jsonl")
        if not os.path.isfile(svd_path):
            continue
        entries = read_jsonl(svd_path)

        step_se = {}
        for e in entries:
            s = e["step"]
            step_se.setdefault(s, []).append(e["spectral_energy"])

        steps = sorted(step_se.keys())
        mean_se = [np.mean(step_se[s]) for s in steps]
        x = normalize_steps(steps)

        ax.plot(x, mean_se, label=name, color=COLORS[name], linewidth=LINE_WIDTH, alpha=0.88)

    ax.axhline(y=0.7, color="red", linestyle="--", linewidth=1.5, alpha=0.6,
               label="Collapse threshold (0.7)")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Mean Spectral Energy (S₁/ΣSᵢ)")
    ax.set_title("A) Spectral Energy Dynamics")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Max SE over training progress ---
    ax = axes[1]
    for name in compare:
        svd_path = os.path.join(EXPERIMENTS[name], "svd_log.jsonl")
        if not os.path.isfile(svd_path):
            continue
        entries = read_jsonl(svd_path)

        step_se = {}
        for e in entries:
            s = e["step"]
            step_se.setdefault(s, []).append(e["spectral_energy"])

        steps = sorted(step_se.keys())
        max_se = [max(step_se[s]) for s in steps]
        x = normalize_steps(steps)

        ax.plot(x, max_se, label=name, color=COLORS[name], linewidth=LINE_WIDTH, alpha=0.88)

    ax.axhline(y=0.7, color="red", linestyle="--", linewidth=1.5, alpha=0.6,
               label="Collapse threshold (0.7)")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Max Spectral Energy (S₁/ΣSᵢ)")
    ax.set_title("B) Worst-Layer Spectral Energy")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    suptitle_tight(fig, "SVD Dynamics: Baseline vs Aggressive vs Transfer", fontsize=14)
    out = os.path.join(OUTPUT_DIR, "fig2_svd_dynamics.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ════════════════════════════════════════════════════════════════════
# Figure 3: Layer-wise SE Heatmap (final checkpoint)
# ════════════════════════════════════════════════════════════════════
def fig3_layer_heatmap():
    compare = ["EN baseline", "KY baseline", "KY 10ep", "KY r=64", "KZ→KY transfer", "KZ baseline"]

    all_data = {}
    all_layers = set()

    for name in compare:
        svd_path = os.path.join(EXPERIMENTS[name], "svd_log.jsonl")
        if not os.path.isfile(svd_path):
            continue
        entries = read_jsonl(svd_path)
        if not entries:
            continue

        max_step = max(e["step"] for e in entries)
        final = [e for e in entries if e["step"] == max_step]

        layer_se = {}
        for e in final:
            idx = e["layer_index"]
            layer_se[idx] = e["spectral_energy"]
            all_layers.add(idx)

        all_data[name] = layer_se

    if not all_data:
        print("[SKIP] fig3: no SVD data")
        return

    layers = sorted(all_layers)
    matrix = np.zeros((len(compare), len(layers)))

    for i, name in enumerate(compare):
        for j, layer in enumerate(layers):
            matrix[i, j] = all_data.get(name, {}).get(layer, 0)

    fig, ax = plt.subplots(figsize=(17, 5.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)

    ax.set_yticks(range(len(compare)))
    ax.set_yticklabels(compare, fontsize=11)

    # Show every 5th layer
    tick_positions = [i for i in range(len(layers)) if layers[i] % 5 == 0]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(layers[i]) for i in tick_positions], fontsize=9)
    ax.set_xlabel("Transformer Layer Index")
    ax.set_title("Spectral Energy by Layer (Final Checkpoint)", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("S₁/ΣSᵢ", fontsize=12)

    fig.tight_layout(pad=1.2)
    out = os.path.join(OUTPUT_DIR, "fig3_layer_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ════════════════════════════════════════════════════════════════════
# Figure 4: Cross-lingual PPL Matrix
# ════════════════════════════════════════════════════════════════════
def fig4_cross_ppl():
    """Heatmap: rows = experiments, columns = eval language (KY, KZ, UZ)."""
    exp_names = list(EXPERIMENTS.keys())
    langs = ["ky", "kz", "uz"]
    lang_labels = ["Kyrgyz", "Kazakh", "Uzbek"]

    matrix = np.zeros((len(exp_names), len(langs)))

    for i, name in enumerate(exp_names):
        rpath = os.path.join(EXPERIMENTS[name], "eval", "eval_report.json")
        if not os.path.isfile(rpath):
            continue
        report = json.load(open(rpath))
        for j, lang in enumerate(langs):
            ppl = report.get("perplexity", {}).get(lang, {}).get("ppl", None)
            matrix[i, j] = min(ppl, 200) if ppl else 200

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=200)

    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(lang_labels, fontsize=12)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names, fontsize=11)

    # Annotate cells
    for i, name in enumerate(exp_names):
        rpath = os.path.join(EXPERIMENTS[name], "eval", "eval_report.json")
        if not os.path.isfile(rpath):
            continue
        report = json.load(open(rpath))
        for j, lang in enumerate(langs):
            raw = report.get("perplexity", {}).get(lang, {}).get("ppl", None)
            text = f"{raw:.1f}" if raw and raw < 200 else (f"{raw:.0f}" if raw else "—")
            color = "white" if matrix[i, j] > 100 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)

    ax.set_title("Cross-Lingual Perplexity (eval set)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Evaluation Language")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("PPL (capped at 200)", fontsize=12)

    fig.tight_layout(pad=1.2)
    out = os.path.join(OUTPUT_DIR, "fig4_cross_ppl.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ════════════════════════════════════════════════════════════════════
# Figure 5: Effective Rank Dynamics
# ════════════════════════════════════════════════════════════════════
def fig5_effective_rank():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    compare = ["KY baseline", "KY r=64", "KZ→KY transfer"]

    for name in compare:
        svd_path = os.path.join(EXPERIMENTS[name], "svd_log.jsonl")
        if not os.path.isfile(svd_path):
            continue
        entries = read_jsonl(svd_path)

        step_rank = {}
        for e in entries:
            s = e["step"]
            step_rank.setdefault(s, []).append(e["effective_rank"])

        steps = sorted(step_rank.keys())
        mean_rank = [np.mean(step_rank[s]) for s in steps]
        x = normalize_steps(steps)

        ax.plot(x, mean_rank, label=name, color=COLORS[name], linewidth=LINE_WIDTH, alpha=0.88)

    ax.axhline(y=3.0, color="red", linestyle="--", linewidth=1.5, alpha=0.6,
               label="Critical threshold (rank=3)")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Mean Effective Rank")
    ax.set_title("Effective Rank Dynamics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=1.2)
    out = os.path.join(OUTPUT_DIR, "fig5_effective_rank.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ════════════════════════════════════════════════════════════════════
# Figure 6: Frobenius Norm Dynamics (KEY FIGURE)
# ════════════════════════════════════════════════════════════════════
def fig6_frobenius_dynamics():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    compare = ["KY baseline", "KY 10ep", "KY r=64", "KZ→KY transfer", "EN→KY transfer"]

    # --- Panel A: Mean FrobNorm_B over training progress ---
    ax = axes[0]
    for name in compare:
        svd_path = os.path.join(EXPERIMENTS[name], "svd_log.jsonl")
        if not os.path.isfile(svd_path):
            continue
        entries = read_jsonl(svd_path)

        step_frob = {}
        for e in entries:
            s = e["step"]
            step_frob.setdefault(s, []).append(e["frobenius_norm_B"])

        steps = sorted(step_frob.keys())
        mean_frob = [np.mean(step_frob[s]) for s in steps]
        x = normalize_steps(steps)

        ax.plot(x, mean_frob, label=name, color=COLORS[name], linewidth=LINE_WIDTH, alpha=0.88)

    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Mean ‖B‖_F")
    ax.set_title("A) Frobenius Norm of lora_B")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Growth ratio (normalized to step 0) ---
    ax = axes[1]
    for name in compare:
        svd_path = os.path.join(EXPERIMENTS[name], "svd_log.jsonl")
        if not os.path.isfile(svd_path):
            continue
        entries = read_jsonl(svd_path)

        step_frob = {}
        for e in entries:
            s = e["step"]
            step_frob.setdefault(s, []).append(e["frobenius_norm_B"])

        steps = sorted(step_frob.keys())
        mean_frob = [np.mean(step_frob[s]) for s in steps]
        x = normalize_steps(steps)

        if mean_frob and mean_frob[0] > 0:
            ratio = [v / mean_frob[0] for v in mean_frob]
        else:
            ratio = mean_frob

        ax.plot(x, ratio, label=name, color=COLORS[name], linewidth=LINE_WIDTH, alpha=0.88)

    ax.axhline(y=5, color="orange", linestyle="--", linewidth=1.5, alpha=0.6, label="Warning (5×)")
    ax.axhline(y=10, color="red", linestyle="--", linewidth=1.5, alpha=0.6, label="Critical (10×)")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("‖B‖_F Growth Ratio (vs init)")
    ax.set_title("B) Frobenius Norm Growth Factor")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    suptitle_tight(fig, "Frobenius Norm Dynamics: The Missing Diagnostic", fontsize=14)
    out = os.path.join(OUTPUT_DIR, "fig6_frobenius_dynamics.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ════════════════════════════════════════════════════════════════════
# Figure 7: Overfit Paradox (Train Loss vs Eval PPL)
# ════════════════════════════════════════════════════════════════════
def fig7_overfit_paradox():
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    tlog_path = os.path.join(EXPERIMENTS["KY 10ep"], "training_log.jsonl")
    if not os.path.isfile(tlog_path):
        print("[SKIP] fig7: no training log for KY 10ep")
        return

    entries = read_jsonl(tlog_path)
    steps = [e["step"] for e in entries]
    train_loss = [e["train_loss"] for e in entries]

    ppl_path = os.path.join(EXPERIMENTS["KY 10ep"], "ppl_per_lang_log.jsonl")
    ppl_entries = read_jsonl(ppl_path) if os.path.isfile(ppl_path) else []
    ky_ppl_entries = [e for e in ppl_entries if e.get("lang") == "ky"]
    ppl_steps = [e["step"] for e in ky_ppl_entries]
    ppl_vals = [e["ppl"] for e in ky_ppl_entries]

    color1 = "#2196F3"
    ax1.plot(steps, train_loss, color=color1, linewidth=LINE_WIDTH, alpha=0.85, label="Train Loss")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Train Loss", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#F44336"
    if ppl_steps:
        ax2.plot(ppl_steps, ppl_vals, color=color2, linewidth=LINE_WIDTH, alpha=0.85,
                 marker="o", markersize=4, label="KY Eval PPL")
    ax2.set_ylabel("Eval PPL (KY)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)

    ax1.set_title("The Overfit Paradox: Train Loss ↓↓ but Eval PPL Improves",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout(pad=1.2)
    out = os.path.join(OUTPUT_DIR, "fig7_overfit_paradox.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ════════════════════════════════════════════════════════════════════
# Figure 8: Singular Value Distributions (histograms)
# ════════════════════════════════════════════════════════════════════
def fig8_singular_values():
    compare = {
        "KY baseline (final)": ("KY baseline", -1),
        "KY r=64 (final)": ("KY r=64", -1),
        "KZ→KY transfer (final)": ("KZ→KY transfer", -1),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for idx, (label, (exp_name, step_idx)) in enumerate(compare.items()):
        ax = axes[idx]
        svd_path = os.path.join(EXPERIMENTS[exp_name], "svd_log.jsonl")
        if not os.path.isfile(svd_path):
            continue
        entries = read_jsonl(svd_path)
        if not entries:
            continue

        max_step = max(e["step"] for e in entries)
        final = [e for e in entries if e["step"] == max_step]

        all_svs = []
        for e in final:
            svs = e.get("singular_values", [])
            if svs:
                s_arr = np.array(svs)
                if s_arr.sum() > 0:
                    all_svs.append(s_arr / s_arr.sum())

        if not all_svs:
            continue

        max_len = max(len(s) for s in all_svs)
        padded = np.zeros((len(all_svs), max_len))
        for i, s in enumerate(all_svs):
            padded[i, :len(s)] = s

        mean_sv = padded.mean(axis=0)
        ranks = np.arange(1, max_len + 1)

        ax.bar(ranks, mean_sv, color=COLORS[exp_name], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Normalized σᵢ / Σσ")
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0.5, min(max_len + 0.5, 20.5))
        ax.grid(True, alpha=0.3, axis="y")

    suptitle_tight(fig, "Singular Value Distributions at Final Checkpoint", fontsize=14)
    out = os.path.join(OUTPUT_DIR, "fig8_singular_values.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ════════════════════════════════════════════════════════════════════
# Figure 9: FrobNorm Growth vs Cross-lingual PPL (scatter)
# ════════════════════════════════════════════════════════════════════
def fig9_frobnorm_vs_ppl():
    fig, ax = plt.subplots(figsize=(10, 7))

    target_lang = {
        "EN baseline": None,  # skip from cross-lingual analysis
        "KZ baseline": "kz", "UZ baseline": "uz", "KY baseline": "ky",
        "KY 10ep": "ky", "KY r=64": "ky", "KZ→KY transfer": "ky", "EN→KY transfer": "ky",
    }

    points = []  # (frob_growth, cross_ppl, name)

    for name, path in EXPERIMENTS.items():
        tgt = target_lang.get(name)
        if tgt is None:
            continue  # skip EN baseline from this cross-lingual analysis

        svd_path = os.path.join(path, "svd_log.jsonl")
        rpath = os.path.join(path, "eval", "eval_report.json")
        if not os.path.isfile(svd_path) or not os.path.isfile(rpath):
            continue

        entries = read_jsonl(svd_path)
        if not entries:
            continue

        report = json.load(open(rpath))
        ppl_data = report.get("perplexity", {})

        steps_sorted = sorted(set(e["step"] for e in entries))
        if len(steps_sorted) < 2:
            continue

        first_step, last_step = steps_sorted[0], steps_sorted[-1]
        first_frob = np.mean([e["frobenius_norm_B"] for e in entries if e["step"] == first_step])
        last_frob = np.mean([e["frobenius_norm_B"] for e in entries if e["step"] == last_step])

        if first_frob <= 0:
            continue
        growth = last_frob / first_frob

        other_ppls = []
        for lang in ["ky", "kz", "uz"]:
            if lang == tgt:
                continue
            ppl = ppl_data.get(lang, {}).get("ppl", None)
            if ppl is not None:
                other_ppls.append(min(ppl, 800))

        if not other_ppls:
            continue

        mean_cross_ppl = np.mean(other_ppls)
        points.append((growth, mean_cross_ppl, name))

    if not points:
        print("[SKIP] fig9: no data")
        return

    for growth, ppl, name in points:
        ax.scatter(growth, ppl, color=COLORS[name], s=150, zorder=5,
                   edgecolors="black", linewidth=0.8)

    # Smart annotations: avoid right-edge overflow by flipping direction for large x
    all_growths = [p[0] for p in points]
    log_min, log_max = np.log10(min(all_growths)), np.log10(max(all_growths))
    log_range = log_max - log_min

    for growth, ppl, name in points:
        log_pos = (np.log10(growth) - log_min) / log_range  # 0=left, 1=right
        if log_pos > 0.65:
            # Right side: place label to the left
            ax.annotate(name, (growth, ppl), textcoords="offset points",
                        xytext=(-12, 8), fontsize=9.5, ha="right",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
        else:
            ax.annotate(name, (growth, ppl), textcoords="offset points",
                        xytext=(10, 6), fontsize=9.5, ha="left")

    ax.set_xlabel("Frobenius Norm Growth (final / init)", fontsize=13)
    ax.set_ylabel("Mean Cross-Lingual PPL", fontsize=13)
    ax.set_title("Frobenius Norm Growth vs Cross-Lingual Degradation",
                 fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    # Add margin so annotations don't clip at edges
    ax.margins(x=0.2, y=0.2)

    fig.tight_layout(pad=1.5)
    out = os.path.join(OUTPUT_DIR, "fig9_frobnorm_vs_ppl.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Generating publication figures")
    print("=" * 60)

    fig1_eval_summary()
    fig2_svd_dynamics()
    fig3_layer_heatmap()
    fig4_cross_ppl()
    fig5_effective_rank()
    fig6_frobenius_dynamics()
    fig7_overfit_paradox()
    fig8_singular_values()
    fig9_frobnorm_vs_ppl()

    print("\n" + "=" * 60)
    print(f"  All figures saved → {OUTPUT_DIR}/")
    print("=" * 60)
