"""
Скрипт для загрузки и препроцессинга текстовых датасетов
на казахском и узбекском языках из Hugging Face.

Цель: собрать по 150 МБ чистого текста для каждого языка
для последующего обучения Causal LLM (Gemma-2-9B).
"""

import argparse
import json
import os
import re
import sys
from datasets import load_dataset
from tqdm import tqdm

# ── Константы (defaults, overridable via CLI) ────────────────
DEFAULT_TARGET_MB = 150
MIN_LEN = 200   # минимальная длина текста (символы)
MAX_LEN = 5000  # максимальная длина текста (символы)

# ── HTML / спецсимволы ───────────────────────────────────────
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
RE_SPECIAL = re.compile(r"[^\w\s.,!?;:\"'()\-–—/№%«»\u0400-\u04FF\u0600-\u06FF\u0100-\u024F]")
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Очистка текста от HTML, спецсимволов и лишних пробелов."""
    text = RE_HTML_TAG.sub("", raw)
    text = RE_HTML_ENTITY.sub(" ", text)
    text = RE_SPECIAL.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def chunk_text(text: str, max_len: int = MAX_LEN, min_len: int = MIN_LEN) -> list[str]:
    """
    Нарезает длинный текст на куски по абзацам/предложениям.
    Если текст уже подходящей длины — возвращает как есть.
    Слишком короткие куски отбрасываются.
    """
    if len(text) <= max_len:
        return [text] if len(text) >= min_len else []

    chunks = []
    paragraphs = text.split("\n")
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Если добавление абзаца не превышает лимит — добавляем
        if len(current) + len(para) + 1 <= max_len:
            current = current + "\n" + para if current else para
        else:
            # Сохраняем накопленный чанк
            if len(current) >= min_len:
                chunks.append(current)
            # Если абзац сам по себе длинный — режем по предложениям
            if len(para) > max_len:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_len:
                        current = current + " " + sent if current else sent
                    else:
                        if len(current) >= min_len:
                            chunks.append(current)
                        current = sent[:max_len]  # обрезаем сверхдлинные предложения
            else:
                current = para

    if len(current) >= min_len:
        chunks.append(current)

    return chunks


def collect_from_stream(stream, text_field: str, target_bytes: int,
                        desc: str) -> tuple[list[dict], int]:
    """
    Итерирует по потоковому датасету, очищает, нарезает и фильтрует тексты.
    Останавливается, когда суммарный размер UTF-8 текста >= target_bytes.
    """
    collected = []
    total_bytes = 0

    pbar = tqdm(desc=desc, unit="MB", total=target_bytes / (1024 * 1024),
                bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} MB [{elapsed}<{remaining}]")

    for example in stream:
        raw = example.get(text_field)
        if not raw or not isinstance(raw, str):
            continue

        text = clean_text(raw)
        # Нарезаем длинные тексты на чанки подходящей длины
        for chunk in chunk_text(text):
            entry_bytes = len(chunk.encode("utf-8"))
            collected.append({"text": chunk})
            total_bytes += entry_bytes
            pbar.n = total_bytes / (1024 * 1024)
            pbar.refresh()

            if total_bytes >= target_bytes:
                break

        if total_bytes >= target_bytes:
            break

    pbar.close()
    return collected, total_bytes


def save_jsonl(records: list[dict], path: str):
    """Сохраняет список словарей в JSONL-файл."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════
#  УЗБЕКСКИЙ ЯЗЫК
# ══════════════════════════════════════════════════════════════
def collect_uzbek(output_path: str, target_bytes: int) -> None:
    print("\n" + "=" * 60)
    print("  УЗБЕКСКИЙ ЯЗЫК  —  murodbek/uz-books")
    print("=" * 60)

    ds = load_dataset(
        "murodbek/uz-books",
        split="original",
        streaming=True,
    )

    records, total = collect_from_stream(
        stream=ds,
        text_field="text",
        target_bytes=target_bytes,
        desc="UZ  сбор",
    )

    mb = total / (1024 * 1024)
    target_mb = target_bytes / (1024 * 1024)
    print(f"  Собрано записей: {len(records):,}  |  {mb:.1f} МБ")

    if total < target_bytes:
        print(f"  ⚠ Доступно только {mb:.1f} МБ из {target_mb:.0f} МБ — "
              "это весь датасет.")

    save_jsonl(records, output_path)
    print(f"  Сохранено → {output_path}")


# ══════════════════════════════════════════════════════════════
#  КАЗАХСКИЙ ЯЗЫК
# ══════════════════════════════════════════════════════════════
def collect_kazakh(output_path: str, target_bytes: int) -> None:
    print("\n" + "=" * 60)
    print("  КАЗАХСКИЙ ЯЗЫК")
    print("=" * 60)

    target_mb = target_bytes / (1024 * 1024)
    records: list[dict] = []
    total_bytes = 0

    # ── Источник 1: Wikipedia ────────────────────────────────
    print("\n  [1/2]  wikimedia/wikipedia  (20231101.kk)")
    ds_wiki = load_dataset(
        "wikimedia/wikipedia",
        name="20231101.kk",
        split="train",
        streaming=True,
    )

    wiki_records, wiki_bytes = collect_from_stream(
        stream=ds_wiki,
        text_field="text",
        target_bytes=target_bytes,
        desc="KZ wiki",
    )
    records.extend(wiki_records)
    total_bytes += wiki_bytes

    mb_wiki = wiki_bytes / (1024 * 1024)
    print(f"  Wikipedia: {len(wiki_records):,} записей  |  {mb_wiki:.1f} МБ")

    # ── Источник 2: sozkz-corpus (дополнение) ────────────────
    if total_bytes < target_bytes:
        remaining = target_bytes - total_bytes
        remaining_mb = remaining / (1024 * 1024)
        print(f"\n  [2/2]  stukenov/sozkz-corpus-clean-kk-pretrain-v2  "
              f"(нужно ещё {remaining_mb:.1f} МБ)")

        ds_sozk = load_dataset(
            "stukenov/sozkz-corpus-clean-kk-pretrain-v2",
            split="train",
            streaming=True,
        )

        sozk_records, sozk_bytes = collect_from_stream(
            stream=ds_sozk,
            text_field="text",
            target_bytes=remaining,
            desc="KZ sozk",
        )
        records.extend(sozk_records)
        total_bytes += sozk_bytes

        mb_sozk = sozk_bytes / (1024 * 1024)
        print(f"  sozkz-corpus: {len(sozk_records):,} записей  |  "
              f"{mb_sozk:.1f} МБ")
    else:
        print("\n  Wikipedia достаточно — sozkz-corpus не нужен.")

    mb_total = total_bytes / (1024 * 1024)
    print(f"\n  ИТОГО KZ: {len(records):,} записей  |  {mb_total:.1f} МБ")

    if total_bytes < target_bytes:
        print(f"  ⚠ Доступно только {mb_total:.1f} МБ из {target_mb:.0f} МБ — "
              "оба источника исчерпаны.")

    save_jsonl(records, output_path)
    print(f"  Сохранено → {output_path}")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Загрузка казахского и узбекского текстов из HuggingFace")
    p.add_argument("--output_dir", type=str, default="./data/raw_sources",
                    help="Директория для сохранения JSONL файлов")
    p.add_argument("--kz_file", type=str, default="kazakh_raw.jsonl",
                    help="Имя выходного файла для казахского")
    p.add_argument("--uz_file", type=str, default="uzbek_raw.jsonl",
                    help="Имя выходного файла для узбекского")
    p.add_argument("--target_mb", type=int, default=DEFAULT_TARGET_MB,
                    help="Целевой объём на язык в МБ (default: 150)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    target_bytes = args.target_mb * 1024 * 1024
    os.makedirs(args.output_dir, exist_ok=True)

    kz_path = os.path.join(args.output_dir, args.kz_file)
    uz_path = os.path.join(args.output_dir, args.uz_file)

    print(f"Целевой объём: {args.target_mb} МБ чистого текста на язык")
    print(f"Фильтр длины:  {MIN_LEN}–{MAX_LEN} символов")
    print(f"Выход:          {args.output_dir}/")

    collect_uzbek(uz_path, target_bytes)
    collect_kazakh(kz_path, target_bytes)

    print("\n" + "=" * 60)
    print("  ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()
