#!/usr/bin/env python3
"""
wikiann_overlap.py — Quantify entity overlap between training corpora and
WikiANN test sets to estimate the contamination level for NER F1.

For each language, scans the training JSONL and checks how many gold WikiANN
test entities (PER/ORG/LOC strings) appear verbatim in the training text.
Reports per-language coverage rates as an upper bound on memorization-driven
NER F1.

Usage:
    python scripts/wikiann_overlap.py
"""

import json
import os
from datasets import load_dataset

DATA_DIR = "./data/pretrain"
LANG_FILES = {
    "ky": "kyrgyz_raw.jsonl",
    "kz": "kazakh_raw.jsonl",
    "uz": "uzbek_final_cyrillic.jsonl",
}
WIKIANN_LANG = {"ky": "ky", "kz": "kk", "uz": "uz"}
TAG_MAP = {1: "PER", 2: "PER", 3: "ORG", 4: "ORG", 5: "LOC", 6: "LOC"}


def wikiann_to_spans(tokens, tags):
    spans, cur_type, cur_toks = [], None, []
    for t, g in zip(tokens, tags):
        if g in (1, 3, 5):
            if cur_type:
                spans.append((cur_type, " ".join(cur_toks)))
            cur_type, cur_toks = TAG_MAP[g], [t]
        elif g in (2, 4, 6) and cur_type == TAG_MAP.get(g):
            cur_toks.append(t)
        else:
            if cur_type:
                spans.append((cur_type, " ".join(cur_toks)))
            cur_type, cur_toks = (None, []) if g == 0 else (TAG_MAP.get(g), [t])
    if cur_type:
        spans.append((cur_type, " ".join(cur_toks)))
    return spans


def load_corpus_text(path):
    """Concatenate all training text into a single lowercase string for substring search."""
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                chunks.append(obj.get("text", "").lower())
            except json.JSONDecodeError:
                continue
    return " \n ".join(chunks)


def main():
    print("=" * 64)
    print("  WikiANN Test-Set Entity Overlap with Training Corpora")
    print("=" * 64)

    rows = []
    for lang, fname in LANG_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(path):
            print(f"[SKIP] {lang}: {path} not found")
            continue

        print(f"\n[{lang}] Loading training corpus: {path}")
        corpus = load_corpus_text(path)
        print(f"[{lang}]   {len(corpus):,} characters")

        wlang = WIKIANN_LANG[lang]
        print(f"[{lang}] Loading WikiANN test split ({wlang})...")
        ds = load_dataset("unimelb-nlp/wikiann", wlang)
        test = ds.get("test")
        if test is None:
            print(f"[{lang}] No test split.")
            continue

        # Collect all gold spans
        per_type = {"PER": [], "ORG": [], "LOC": []}
        all_spans = []
        for ex in test:
            for t, e in wikiann_to_spans(ex["tokens"], ex["ner_tags"]):
                if t in per_type:
                    per_type[t].append(e.lower().strip())
                    all_spans.append(e.lower().strip())

        # Unique spans
        unique = sorted(set(all_spans))
        print(f"[{lang}]   total spans: {len(all_spans):,}, unique: {len(unique):,}")

        # Substring-membership (entity appears verbatim somewhere in the training text)
        n_unique_present = sum(1 for s in unique if s and s in corpus)
        n_total_present = sum(1 for s in all_spans if s and s in corpus)

        per_type_present = {}
        for t, spans_list in per_type.items():
            if not spans_list:
                per_type_present[t] = (0, 0)
                continue
            uniq = sorted(set(spans_list))
            uniq_in = sum(1 for s in uniq if s and s in corpus)
            per_type_present[t] = (uniq_in, len(uniq))

        rows.append({
            "lang": lang,
            "corpus_chars": len(corpus),
            "n_test_spans": len(all_spans),
            "n_unique_test_spans": len(unique),
            "unique_overlap": (n_unique_present, len(unique)),
            "instance_overlap": (n_total_present, len(all_spans)),
            "per_type_unique_overlap": per_type_present,
        })

    # ── Print summary table ──
    print("\n" + "=" * 64)
    print("  Summary")
    print("=" * 64)
    print(f"  {'Lang':<6} {'Total':>8} {'Unique':>8} {'UniqOverlap':>15} {'TotalOverlap':>15}")
    for r in rows:
        u_in, u_n = r["unique_overlap"]
        t_in, t_n = r["instance_overlap"]
        print(f"  {r['lang']:<6} {r['n_test_spans']:>8,} {r['n_unique_test_spans']:>8,} "
              f"{u_in:>5,}/{u_n:<5,} ({100*u_in/u_n:>5.1f}%)  "
              f"{t_in:>5,}/{t_n:<5,} ({100*t_in/t_n:>5.1f}%)")

    print(f"\n  Per-type unique overlap:")
    print(f"  {'Lang':<6} {'PER':>14} {'ORG':>14} {'LOC':>14}")
    for r in rows:
        cells = []
        for t in ["PER", "ORG", "LOC"]:
            n_in, n_t = r["per_type_unique_overlap"][t]
            if n_t > 0:
                cells.append(f"{n_in}/{n_t} ({100*n_in/n_t:.0f}%)")
            else:
                cells.append("—")
        print(f"  {r['lang']:<6} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")

    # Save report
    report = {"method": "verbatim_substring_overlap", "per_language": rows}
    out_path = "wikiann_overlap_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved → {out_path}")


if __name__ == "__main__":
    main()
