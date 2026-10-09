"""Recompute the trajectory-probe results of Section 4.8 / Table 4 from the
artifacts tracked in this repository, with no access to the full SVD logs.

Inputs (all in git):
  directional_results/early_frobenius_probe.json   31 runs: ||B||_F at steps
                                                   100/300, SE at 300, labels
  output_*heldout*/probe_prediction.json           four held-out verdicts,
                                                   written at step 300
  output_*heldout*/eval_report.json                post-hoc evaluations
  heldout_probe_predictions.log                    append-only prediction log

Reproduces and checks:
  - g = (||B||_F(300) - ||B||_F(100)) / 200 for each of the 31 runs
  - complete separation: max healthy g < min collapsed g, and the paper's
    bounds (healthy <= 0.0055, collapsed >= 0.0076)
  - AUC of g (expected 1.000) and of SE at step 300 (expected ~0.073)
  - Table 4 held-out rows: g, g/theta, verdict, and the functional outcome
    (TUMLU at chance) from the eval reports

Usage:  python scripts/recompute_probe_table.py
Exits non-zero if any reproduced number disagrees with the stored one.
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-9


def auc(scores_pos, scores_neg):
    """Rank-based AUC (probability a collapsed run scores above a healthy one),
    equivalent to the Mann-Whitney U statistic; ties count 1/2."""
    wins = ties = 0
    for p in scores_pos:
        for n in scores_neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(scores_pos) * len(scores_neg))


def main():
    failures = []

    with open(os.path.join(ROOT, "directional_results/early_frobenius_probe.json")) as f:
        probe = json.load(f)
    runs = probe["runs"]

    # 1. Recompute g from the stored norms and compare with the stored value.
    for name, r in runs.items():
        g = (r["norm_300"] - r["norm_100"]) / 200.0
        if abs(g - r["g_100_300"]) > TOL:
            failures.append(f"{name}: recomputed g={g:.6f} != stored {r['g_100_300']:.6f}")

    healthy = {k: v for k, v in runs.items() if v["healthy"]}
    collapsed = {k: v for k, v in runs.items() if not v["healthy"]}
    print(f"runs: {len(runs)} = {len(healthy)} healthy + {len(collapsed)} collapsed")

    # 2. Separation and the paper's class bounds.
    hmax = max(v["g_100_300"] for v in healthy.values())
    cmin = min(v["g_100_300"] for v in collapsed.values())
    print(f"max healthy g = {hmax:.4f}   min collapsed g = {cmin:.4f}   margin = {cmin/hmax:.2f}x")
    if not (hmax < cmin):
        failures.append("classes overlap in g")
    if not (hmax <= 0.0055 + 5e-5 and cmin >= 0.0076 - 5e-5):
        failures.append(f"class bounds differ from the paper: {hmax:.4f} / {cmin:.4f}")

    # 3. AUCs: g should be 1.000, SE at the same step ~0.073 (anti-correlated).
    auc_g = auc([v["g_100_300"] for v in collapsed.values()],
                [v["g_100_300"] for v in healthy.values()])
    auc_se = auc([v["se_300"] for v in collapsed.values()],
                 [v["se_300"] for v in healthy.values()])
    print(f"AUC(g) = {auc_g:.3f}   AUC(SE@300) = {auc_se:.3f}")
    if auc_g != 1.0:
        failures.append(f"AUC(g) = {auc_g:.3f}, expected 1.000")
    if abs(auc_se - 0.073) > 0.005:
        failures.append(f"AUC(SE@300) = {auc_se:.3f}, expected ~0.073")

    # 4. Held-out rows of Table 4.
    print("\nheld-out runs (verdict written at step 300, before evaluation):")
    for pred_path in sorted(glob.glob(os.path.join(ROOT, "output_*heldout*/probe_prediction.json"))):
        d = os.path.dirname(pred_path)
        with open(pred_path) as f:
            p = json.load(f)
        g = (p["norm_t1"] - p["norm_t0"]) / (p["t1"] - p["t0"])
        if abs(g - p["g"]) > TOL:
            failures.append(f"{d}: recomputed held-out g != stored")
        ratio = p["g"] / p["threshold"]
        if abs(ratio - p["margin_ratio"]) > 1e-6:
            failures.append(f"{d}: margin ratio mismatch")
        tumlu = None
        eval_path = os.path.join(d, "eval_report.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                rep = json.load(f)

            def find_acc(node):
                out = []
                if isinstance(node, dict):
                    for k, v in node.items():
                        if k == "accuracy" and isinstance(v, (int, float)):
                            out.append(v)
                        out += find_acc(v)
                elif isinstance(node, list):
                    for v in node:
                        out += find_acc(v)
                return out

            accs = find_acc(rep)
            tumlu = min(accs) if accs else None
        chance = "at chance" if tumlu is not None and tumlu < 0.27 else "n/a"
        print(f"  {os.path.basename(d):42s} g={p['g']:.4f}  {ratio:.1f}x thr "
              f"pred={p['prediction']}  TUMLU_min={tumlu if tumlu is None else round(tumlu,3)} ({chance})")
        if p["prediction"] != "collapse":
            failures.append(f"{d}: stored prediction is not 'collapse'")

    print()
    if failures:
        print("MISMATCHES:")
        for x in failures:
            print(" -", x)
        sys.exit(1)
    print("All reproduced numbers match the paper (Section 4.8, Table 4).")


if __name__ == "__main__":
    main()
