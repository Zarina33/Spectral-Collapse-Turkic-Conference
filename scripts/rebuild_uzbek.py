"""
Пересборка узбекского датасета: только кириллица (>90%).

1. Фильтрует uzbek_raw.jsonl — оставляет записи с кириллицей >90%
2. Если < 150 МБ — добирает из wikimedia/wikipedia (20231101.uz)
3. Если всё ещё < 150 МБ — добирает из HuggingFaceFW/fineweb-2 (uzn_Cyrl)
4. Сохраняет uzbek_final_cyrillic.jsonl
"""

import argparse
import json
import re
import os
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

# ── Константы ────────────────────────────────────────────────
TARGET_BYTES = 150 * 1024 * 1024  # 150 МБ
MIN_LEN = 200
MAX_LEN = 5000
CYR_THRESHOLD = 0.90  # минимум 90% кириллицы

RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")


def cyrillic_ratio(text: str) -> float:
    """Доля кириллических символов среди всех буквенных."""
    cyrillic = len(re.findall(r"[\u0400-\u04FF]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    total = cyrillic + latin
    return cyrillic / total if total > 0 else 0.0


def clean_text(raw: str) -> str:
    text = RE_HTML_TAG.sub("", raw)
    text = RE_HTML_ENTITY.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def chunk_text(text: str) -> list[str]:
    """Нарезает длинный текст на куски по абзацам."""
    if len(text) <= MAX_LEN:
        return [text] if len(text) >= MIN_LEN else []

    chunks = []
    paragraphs = text.split("\n")
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 <= MAX_LEN:
            current = current + "\n" + para if current else para
        else:
            if len(current) >= MIN_LEN:
                chunks.append(current)
            if len(para) > MAX_LEN:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= MAX_LEN:
                        current = current + " " + sent if current else sent
                    else:
                        if len(current) >= MIN_LEN:
                            chunks.append(current)
                        current = sent[:MAX_LEN]
            else:
                current = para

    if len(current) >= MIN_LEN:
        chunks.append(current)
    return chunks


# ── CLI Arguments ────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Пересборка узбекского датасета: только кириллица (>90%)")
    p.add_argument("--input", type=str, default="./data/raw_sources/uzbek_raw.jsonl",
                    help="Путь к исходному uzbek_raw.jsonl")
    p.add_argument("--output", type=str, default="./data/pretrain/uzbek_final_cyrillic.jsonl",
                    help="Путь для сохранения результата")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    #  Шаг 1: Фильтрация существующего uzbek_raw.jsonl
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  ШАГ 1: Фильтрация uzbek_raw.jsonl (кириллица > 90%)")
    print("=" * 60)

    records = []
    total_bytes = 0
    kept = 0
    dropped = 0

    # Count lines first (memory-efficient) for progress bar
    with open(input_path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)

    with open(input_path, "r", encoding="utf-8") as f:
        pbar = tqdm(f, desc="Фильтрация", unit="rec", total=n_lines)
        for line in pbar:
            entry = json.loads(line)
            text = entry["text"]
            ratio = cyrillic_ratio(text)

            if ratio >= CYR_THRESHOLD:
                entry_bytes = len(text.encode("utf-8"))
                records.append(entry)
                total_bytes += entry_bytes
                kept += 1
            else:
                dropped += 1

            pbar.set_postfix(kept=kept, dropped=dropped, mb=f"{total_bytes/(1024*1024):.1f}")
        pbar.close()

    mb = total_bytes / (1024 * 1024)
    print(f"\n  Оставлено:  {kept:,}  |  Отброшено: {dropped:,}")
    print(f"  Размер:     {mb:.1f} МБ из 150 МБ")
    remaining = TARGET_BYTES - total_bytes


    # ══════════════════════════════════════════════════════════════
    #  Шаг 2: Добор из wikimedia/wikipedia (20231101.uz)
    # ══════════════════════════════════════════════════════════════
    if remaining > 0:
        remaining_mb = remaining / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"  ШАГ 2: Добор из wikimedia/wikipedia (20231101.uz)")
        print(f"  Нужно ещё: {remaining_mb:.1f} МБ")
        print("=" * 60)

        ds_wiki = load_dataset(
            "wikimedia/wikipedia",
            name="20231101.uz",
            split="train",
            streaming=True,
        )

        added_wiki = 0
        pbar = tqdm(desc="UZ wiki", unit="MB", total=remaining_mb,
                    bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} MB [{elapsed}<{remaining}]")

        for example in ds_wiki:
            raw = example.get("text", "")
            if not raw or not isinstance(raw, str):
                continue

            text = clean_text(raw)
            for chunk in chunk_text(text):
                if cyrillic_ratio(chunk) < CYR_THRESHOLD:
                    continue

                entry_bytes = len(chunk.encode("utf-8"))
                records.append({"text": chunk})
                total_bytes += entry_bytes
                remaining -= entry_bytes
                added_wiki += 1
                pbar.n = (TARGET_BYTES - remaining) / (1024 * 1024) - mb
                pbar.refresh()

                if remaining <= 0:
                    break
            if remaining <= 0:
                break

        pbar.close()
        print(f"  Добавлено из Wikipedia: {added_wiki:,} записей")
        print(f"  Текущий размер: {total_bytes/(1024*1024):.1f} МБ")
        remaining = TARGET_BYTES - total_bytes


    # ══════════════════════════════════════════════════════════════
    #  Шаг 3: Добор из HuggingFaceFW/fineweb-2 (uzn_Cyrl)
    # ══════════════════════════════════════════════════════════════
    if remaining > 0:
        remaining_mb = remaining / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"  ШАГ 3: Добор из HuggingFaceFW/fineweb-2 (uzn_Cyrl)")
        print(f"  Нужно ещё: {remaining_mb:.1f} МБ")
        print("=" * 60)

        ds_fw = load_dataset(
            "HuggingFaceFW/fineweb-2",
            "uzn_Cyrl",
            split="train",
            streaming=True,
        )

        start_remaining = remaining
        added_fw = 0
        pbar = tqdm(desc="fineweb-2", unit="MB", total=remaining_mb,
                    bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} MB [{elapsed}<{remaining}]")

        for example in ds_fw:
            raw = example.get("text", "")
            if not raw or not isinstance(raw, str):
                continue

            text = clean_text(raw)
            for chunk in chunk_text(text):
                if cyrillic_ratio(chunk) < CYR_THRESHOLD:
                    continue

                entry_bytes = len(chunk.encode("utf-8"))
                records.append({"text": chunk})
                total_bytes += entry_bytes
                remaining -= entry_bytes
                added_fw += 1
                pbar.n = (start_remaining - remaining) / (1024 * 1024)
                pbar.refresh()

                if remaining <= 0:
                    break
            if remaining <= 0:
                break

        pbar.close()
        print(f"  Добавлено из fineweb-2: {added_fw:,} записей")
        print(f"  Текущий размер: {total_bytes/(1024*1024):.1f} МБ")
        remaining = TARGET_BYTES - total_bytes


    # ══════════════════════════════════════════════════════════════
    #  Сохранение
    # ══════════════════════════════════════════════════════════════
    # Обрезаем до ровно 150 МБ если немного превысили
    final_records = []
    final_bytes = 0
    for rec in records:
        entry_bytes = len(rec["text"].encode("utf-8"))
        if final_bytes + entry_bytes > TARGET_BYTES:
            break
        final_records.append(rec)
        final_bytes += entry_bytes

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"  ИТОГ")
    print(f"{'='*60}")
    print(f"  Файл:       {output_path}")
    print(f"  Записей:    {len(final_records):,}")
    print(f"  Размер:     {final_bytes/(1024*1024):.1f} МБ")

    # ── Финальная статистика ─────────────────────────────────────
    print(f"\n  Вычисляю статистику...")
    all_words = []
    for rec in final_records:
        all_words.extend(rec["text"].lower().split())

    total_words = len(all_words)
    unique_tokens = len(set(all_words))

    print(f"  Всего слов:         {total_words:,}")
    print(f"  Уникальных токенов: {unique_tokens:,}")
    print(f"  TTR (полный):       {unique_tokens/total_words:.4f}")

    # Проверка кириллицы
    sample_text = " ".join(r["text"] for r in final_records[:200])
    ratio = cyrillic_ratio(sample_text)
    print(f"  Кириллица (сэмпл):  {ratio*100:.1f}%")

    if remaining > 0:
        print(f"\n  ⚠ Не удалось набрать 150 МБ. Нехватка: {remaining/(1024*1024):.1f} МБ")
        print(f"    Все три источника исчерпаны.")

    print(f"\nГОТОВО!")


if __name__ == "__main__":
    main()
