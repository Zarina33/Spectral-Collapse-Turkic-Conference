#!/usr/bin/env python
"""
Early ||B||_F growth-rate probe (paper Section 4.9, Eq. 2).

    g = ( mean_m ||B_m||_F(t1) - mean_m ||B_m||_F(t0) ) / (t1 - t0)

read from svd_log.jsonl, averaged over all LoRA modules.  The default
threshold is the midpoint between the healthy maximum and the collapse
minimum over the 31 runs in directional_results/early_frobenius_probe.json
(candidate_threshold = 0.00653).  Prints the prediction and, with --out,
writes a timestamped JSON so the prediction is on record before evaluation.

Usage:
    python scripts/probe_predict.py RUN_DIR [RUN_DIR ...] [--t0 100 --t1 300]
                                    [--threshold 0.00653] [--out FILE] [--log FILE]
Exit status 3 if a run's log does not yet contain step t1 (still training).
"""
import argparse, json, os, sys, datetime
from collections import defaultdict


def growth_rate(svd_log, t0, t1):
    norms = defaultdict(list)  # step -> list of ||B||_F over modules
    with open(svd_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("step") in (t0, t1) and "frobenius_norm_B" in r:
                norms[r["step"]].append(r["frobenius_norm_B"])
    if t0 not in norms or t1 not in norms:
        return None
    m0 = sum(norms[t0]) / len(norms[t0])
    m1 = sum(norms[t1]) / len(norms[t1])
    return {"t0": t0, "t1": t1, "n_modules": len(norms[t1]),
            "norm_t0": m0, "norm_t1": m1, "g": (m1 - m0) / (t1 - t0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--t0", type=int, default=100)
    p.add_argument("--t1", type=int, default=300)
    p.add_argument("--threshold", type=float, default=0.00653)
    p.add_argument("--out", help="write per-run JSON (single run) or a dict (several)")
    p.add_argument("--log", help="append one line per run to this log file")
    a = p.parse_args()

    results, pending = {}, []
    for d in a.run_dirs:
        log = os.path.join(d, "svd_log.jsonl")
        if not os.path.isfile(log):
            pending.append(d); continue
        r = growth_rate(log, a.t0, a.t1)
        if r is None:
            pending.append(d); continue
        r["threshold"] = a.threshold
        r["prediction"] = "collapse" if r["g"] > a.threshold else "healthy"
        r["margin_ratio"] = r["g"] / a.threshold
        r["timestamp_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds")
        results[d] = r
        print(f"{d}: g[{a.t0}->{a.t1}] = {r['g']:.5f}  "
              f"({r['margin_ratio']:.2f}x threshold)  -> {r['prediction'].upper()}")
        if a.log:
            with open(a.log, "a") as f:
                f.write(f"{r['timestamp_utc']}  {d}  g={r['g']:.6f}  "
                        f"thr={a.threshold}  pred={r['prediction']}\n")
    for d in pending:
        print(f"{d}: step {a.t1} not yet logged", file=sys.stderr)
    if a.out and results:
        payload = next(iter(results.values())) if len(results) == 1 else results
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2)
    sys.exit(3 if pending else 0)


if __name__ == "__main__":
    main()
