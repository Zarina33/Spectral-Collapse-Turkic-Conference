"""
Комплексный анализ pretrain-датасетов (KY / UZ / KZ)
для включения в научную статью.

Генерирует:
  1. Консольный отчёт со всеми таблицами
  2. dataset_analysis.png  — сводная визуализация (4 графика)
  3. dataset_report.txt    — текстовый отчёт (копипаст в статью)
"""

import argparse
import json
import os
import re
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

COLORS = {"Kyrgyz (KY)": "#2196F3", "Uzbek (UZ)": "#4CAF50", "Kazakh (KZ)": "#FF9800"}


# ── Загрузка ─────────────────────────────────────────────────
def load_jsonl(path: str) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


# ── Анализ одного датасета ───────────────────────────────────
def analyze(name: str, texts: list[str]) -> dict:
    char_lens = [len(t) for t in texts]
    byte_lens = [len(t.encode("utf-8")) for t in texts]
    word_counts = [len(t.split()) for t in texts]
    # Split on sentence-ending punctuation followed by space/newline/end
    # (avoids breaking on abbreviations like "г.", "т.д.", "ж.")
    sent_counts = [len(re.split(r'[.!?]+(?:\s|$)', t)) for t in texts]

    total_chars = sum(char_lens)
    total_bytes = sum(byte_lens)
    total_words = sum(word_counts)
    total_sents = sum(sent_counts)

    # Уникальные слова (vocabulary)
    all_words = []
    for t in texts:
        all_words.extend(t.lower().split())
    word_freq = Counter(all_words)
    vocab_size = len(word_freq)

    # Type-Token Ratio (TTR) на сэмпле — первые 100K слов
    sample_words = all_words[:100_000]
    ttr = len(set(sample_words)) / len(sample_words) if sample_words else 0

    # Средняя длина слова (в символах)
    avg_word_len = sum(len(w) for w in all_words) / len(all_words) if all_words else 0

    # Процент кириллицы (sample evenly across dataset)
    step = max(1, len(texts) // 2000)
    full_text_sample = " ".join(texts[::step][:2000])
    cyrillic = len(re.findall(r"[\u0400-\u04FF]", full_text_sample))
    latin = len(re.findall(r"[a-zA-Z]", full_text_sample))
    total_alpha = cyrillic + latin
    cyrillic_pct = (cyrillic / total_alpha * 100) if total_alpha > 0 else 0

    # Распределение по длине (символы)
    bins = [(200, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 99999)]
    bin_counts = {}
    for lo, hi in bins:
        label = f"{lo}-{hi}" if hi < 99999 else f"{lo}+"
        bin_counts[label] = sum(1 for l in char_lens if lo <= l < hi)

    return {
        "name": name,
        "num_records": len(texts),
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "total_words": total_words,
        "total_sents": total_sents,
        "char_lens": char_lens,
        "word_counts": word_counts,
        "char_min": min(char_lens),
        "char_max": max(char_lens),
        "char_mean": statistics.mean(char_lens),
        "char_median": statistics.median(char_lens),
        "char_std": statistics.stdev(char_lens) if len(char_lens) > 1 else 0,
        "words_mean": statistics.mean(word_counts),
        "words_median": statistics.median(word_counts),
        "sents_mean": statistics.mean(sent_counts),
        "vocab_size": vocab_size,
        "ttr": ttr,
        "avg_word_len": avg_word_len,
        "cyrillic_pct": cyrillic_pct,
        "bin_counts": bin_counts,
        "top_words": word_freq.most_common(20),
    }


# ── Форматирование таблиц ───────────────────────────────────
def fmt(n, decimals=0):
    if decimals == 0:
        return f"{int(n):,}".replace(",", " ")
    return f"{n:,.{decimals}f}".replace(",", " ")


def build_report(results: list[dict]) -> str:
    lines = []
    W = 72

    lines.append("=" * W)
    lines.append("  DATASET ANALYSIS REPORT")
    lines.append("  Pretrain corpora for Causal LLM (Gemma-2-9B)")
    lines.append("=" * W)

    # ── Table 1: General statistics ──
    lines.append("\n" + "─" * W)
    lines.append("  Table 1. General corpus statistics")
    lines.append("─" * W)
    header = f"  {'Metric':<35} {'KY':>10} {'UZ':>10} {'KZ':>10}"
    lines.append(header)
    lines.append("  " + "─" * (W - 2))

    r = {d["name"]: d for d in results}
    ky, uz, kz = r["Kyrgyz (KY)"], r["Uzbek (UZ)"], r["Kazakh (KZ)"]

    rows = [
        ("Number of records",       ky["num_records"],  uz["num_records"],  kz["num_records"], 0),
        ("Total size (MB)",         ky["total_mb"],     uz["total_mb"],     kz["total_mb"], 1),
        ("Total characters",        ky["total_chars"],  uz["total_chars"],  kz["total_chars"], 0),
        ("Total words",             ky["total_words"],  uz["total_words"],  kz["total_words"], 0),
        ("Total sentences (approx)",ky["total_sents"],  uz["total_sents"],  kz["total_sents"], 0),
        ("Vocabulary size (unique)", ky["vocab_size"],   uz["vocab_size"],   kz["vocab_size"], 0),
        ("Type-Token Ratio (100K)",  ky["ttr"],          uz["ttr"],          kz["ttr"], 3),
        ("Avg word length (chars)",  ky["avg_word_len"], uz["avg_word_len"], kz["avg_word_len"], 1),
        ("Cyrillic script (%)",      ky["cyrillic_pct"], uz["cyrillic_pct"], kz["cyrillic_pct"], 1),
    ]
    for label, v1, v2, v3, dec in rows:
        lines.append(f"  {label:<35} {fmt(v1, dec):>10} {fmt(v2, dec):>10} {fmt(v3, dec):>10}")

    # ── Table 2: Text length statistics ──
    lines.append("\n" + "─" * W)
    lines.append("  Table 2. Text length distribution (characters per record)")
    lines.append("─" * W)
    header = f"  {'Metric':<35} {'KY':>10} {'UZ':>10} {'KZ':>10}"
    lines.append(header)
    lines.append("  " + "─" * (W - 2))

    rows2 = [
        ("Min length",    ky["char_min"],    uz["char_min"],    kz["char_min"], 0),
        ("Max length",    ky["char_max"],    uz["char_max"],    kz["char_max"], 0),
        ("Mean length",   ky["char_mean"],   uz["char_mean"],   kz["char_mean"], 0),
        ("Median length", ky["char_median"], uz["char_median"], kz["char_median"], 0),
        ("Std deviation", ky["char_std"],    uz["char_std"],    kz["char_std"], 0),
        ("Mean words/record",   ky["words_mean"],   uz["words_mean"],   kz["words_mean"], 0),
        ("Median words/record", ky["words_median"], uz["words_median"], kz["words_median"], 0),
        ("Mean sents/record",   ky["sents_mean"],   uz["sents_mean"],   kz["sents_mean"], 1),
    ]
    for label, v1, v2, v3, dec in rows2:
        lines.append(f"  {label:<35} {fmt(v1, dec):>10} {fmt(v2, dec):>10} {fmt(v3, dec):>10}")

    # ── Table 3: Length bins ──
    lines.append("\n" + "─" * W)
    lines.append("  Table 3. Record count by character length range")
    lines.append("─" * W)
    header = f"  {'Range (chars)':<18} {'KY':>8} {'%':>6} {'UZ':>8} {'%':>6} {'KZ':>8} {'%':>6}"
    lines.append(header)
    lines.append("  " + "─" * (W - 2))

    for bin_label in ky["bin_counts"]:
        c_ky = ky["bin_counts"][bin_label]
        c_uz = uz["bin_counts"][bin_label]
        c_kz = kz["bin_counts"][bin_label]
        p_ky = c_ky / ky["num_records"] * 100
        p_uz = c_uz / uz["num_records"] * 100
        p_kz = c_kz / kz["num_records"] * 100
        lines.append(
            f"  {bin_label:<18} {c_ky:>8,} {p_ky:>5.1f}% {c_uz:>8,} {p_uz:>5.1f}% {c_kz:>8,} {p_kz:>5.1f}%"
        )

    # ── Table 4: Top-20 words ──
    lines.append("\n" + "─" * W)
    lines.append("  Table 4. Top-20 most frequent words per language")
    lines.append("─" * W)
    header = f"  {'#':<4} {'Kyrgyz':<22} {'Uzbek':<22} {'Kazakh':<22}"
    lines.append(header)
    lines.append("  " + "─" * (W - 2))
    for i in range(20):
        ky_w, ky_c = ky["top_words"][i]
        uz_w, uz_c = uz["top_words"][i]
        kz_w, kz_c = kz["top_words"][i]
        lines.append(
            f"  {i+1:<4} {ky_w:<12} {ky_c:>8,}  {uz_w:<12} {uz_c:>8,}  {kz_w:<12} {kz_c:>8,}"
        )

    # ── Data sources ──
    lines.append("\n" + "─" * W)
    lines.append("  Data Sources")
    lines.append("─" * W)
    lines.append("  Kyrgyz:  Local corpus (kyrgyz_cleaned_corpus.txt)")
    lines.append("           Literature, history, encyclopedias")
    lines.append("  Uzbek:   HuggingFace murodbek/uz-books (Cyrillic-filtered)")
    lines.append("           + wikimedia/wikipedia (20231101.uz)")
    lines.append("           + HuggingFaceFW/fineweb-2 (uzn_Cyrl)")
    lines.append("  Kazakh:  HuggingFace wikimedia/wikipedia (20231101.kk)")
    lines.append("           + stukenov/sozkz-corpus-clean-kk-pretrain-v2")

    lines.append("\n" + "─" * W)
    lines.append("  Preprocessing")
    lines.append("─" * W)
    lines.append("  - HTML tags and entities removed")
    lines.append("  - Special characters filtered (Cyrillic + basic punctuation kept)")
    lines.append("  - Multiple whitespace normalized")
    lines.append("  - Long texts chunked by paragraph/sentence boundaries")
    lines.append("  - Chunk length filter: 200–5,000 characters")
    lines.append("  - Uzbek corpus filtered for Cyrillic script (>90% threshold)")
    lines.append("  - Each language balanced to exactly 150 MB of clean text")
    lines.append("  - Output format: JSONL ({\"text\": \"...\"})")

    lines.append("\n" + "=" * W)
    return "\n".join(lines)


# ── Визуализация ─────────────────────────────────────────────
def plot_analysis(results: list[dict], out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pretrain Dataset Analysis: Kyrgyz / Uzbek / Kazakh", fontsize=14, fontweight="bold")

    names = [r["name"] for r in results]
    colors = [COLORS[n] for n in names]
    short_names = ["KY", "UZ", "KZ"]

    # ── (a) Corpus size comparison ──
    ax = axes[0, 0]
    metrics = ["Records", "Words (K)", "Vocab (K)"]
    x = np.arange(len(metrics))
    width = 0.25
    for i, r in enumerate(results):
        vals = [r["num_records"], r["total_words"] / 1000, r["vocab_size"] / 1000]
        bars = ax.bar(x + i * width, vals, width, label=short_names[i], color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:,.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_title("(a) Corpus metrics comparison")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ── (b) Text length distribution (histogram) ──
    ax = axes[0, 1]
    bins_hist = np.linspace(200, 5000, 50)
    for i, r in enumerate(results):
        ax.hist(r["char_lens"], bins=bins_hist, alpha=0.5, label=short_names[i],
                color=colors[i], density=True)
    ax.set_xlabel("Characters per record")
    ax.set_ylabel("Density")
    ax.set_title("(b) Text length distribution")
    ax.legend()

    # ── (c) Length range breakdown (stacked bar) ──
    ax = axes[1, 0]
    bin_labels = list(results[0]["bin_counts"].keys())
    x = np.arange(len(bin_labels))
    width = 0.25
    for i, r in enumerate(results):
        vals = [r["bin_counts"][bl] / r["num_records"] * 100 for bl in bin_labels]
        ax.bar(x + i * width, vals, width, label=short_names[i], color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% of records")
    ax.set_title("(c) Records by character length range")
    ax.legend()

    # ── (d) Lexical diversity metrics ──
    ax = axes[1, 1]
    metrics_d = ["TTR (100K)", "Avg word\nlength", "Cyrillic %\n(÷10)"]
    x = np.arange(len(metrics_d))
    width = 0.25
    for i, r in enumerate(results):
        vals = [r["ttr"], r["avg_word_len"], r["cyrillic_pct"] / 10]
        ax.bar(x + i * width, vals, width, label=short_names[i], color=colors[i])
        for j, val in enumerate(vals):
            real_val = r["cyrillic_pct"] if j == 2 else val
            label_txt = f"{real_val:.1f}%" if j == 2 else f"{real_val:.3f}" if j == 0 else f"{real_val:.1f}"
            ax.text(x[j] + i * width, val + 0.02, label_txt,
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_d)
    ax.set_title("(d) Lexical diversity metrics")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Графики сохранены → {out_path}")


# ── CLI Arguments ────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Комплексный анализ pretrain-датасетов (KY / UZ / KZ)")
    p.add_argument("--data_dir", type=str, default="./data/pretrain",
                    help="Директория с JSONL-файлами (default: ./data/pretrain)")
    p.add_argument("--ky_file", type=str, default="kyrgyz_raw.jsonl",
                    help="Имя файла кыргызского корпуса")
    p.add_argument("--uz_file", type=str, default="uzbek_final_cyrillic.jsonl",
                    help="Имя файла узбекского корпуса")
    p.add_argument("--kz_file", type=str, default="kazakh_raw.jsonl",
                    help="Имя файла казахского корпуса")
    p.add_argument("--output_dir", type=str, default="./reports",
                    help="Директория для отчётов (default: ./reports)")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────
def main():
    args = parse_args()

    datasets = {
        "Kyrgyz (KY)": os.path.join(args.data_dir, args.ky_file),
        "Uzbek (UZ)":  os.path.join(args.data_dir, args.uz_file),
        "Kazakh (KZ)": os.path.join(args.data_dir, args.kz_file),
    }
    out_png = os.path.join(args.output_dir, "dataset_analysis.png")
    out_txt = os.path.join(args.output_dir, "dataset_report.txt")
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for name, path in datasets.items():
        print(f"Анализирую {name}...")
        texts = load_jsonl(path)
        info = analyze(name, texts)
        results.append(info)
        print(f"  {info['num_records']:,} записей, {info['total_mb']:.1f} МБ, "
              f"vocab={info['vocab_size']:,}")

    # Текстовый отчёт
    report = build_report(results)
    print("\n" + report)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nОтчёт сохранён → {out_txt}")

    # Графики
    plot_analysis(results, out_png)

    print("\nГОТОВО!")


if __name__ == "__main__":
    main()
