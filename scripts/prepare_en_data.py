"""
Скрипт для загрузки и препроцессинга английского контрольного корпуса
из Hugging Face (English Wikipedia).

Цель: собрать ~150 МБ чистого текста — контрольный корпус (E8/E6b),
byte-matched к тюркским корпусам, для обучения Causal LLM (Gemma-2-9B).

Пайплайн (очистка, нарезка 200–5000 символов, стриминг до целевого
объёма) идентичен prepare_kz_uz_data.py, чтобы EN-контроль отличался
от тюркских корпусов только языком, а не препроцессингом.
"""

import argparse
import json
import os
import re
from datasets import load_dataset
from tqdm import tqdm

# ── Константы (defaults, overridable via CLI) ────────────────
DEFAULT_TARGET_MB = 150
MIN_LEN = 200   # минимальная длина текста (символы)
MAX_LEN = 5000  # максимальная длина текста (символы)

# ── HTML / спецсимволы (те же правила, что и для KZ/UZ) ──────
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
RE_SPECIAL = re.compile(r"[^\w\s.,!?;:\"'()\-–—/№%«»Ѐ-ӿ؀-ۿĀ-ɏ]")
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
#  АНГЛИЙСКИЙ ЯЗЫК (контрольный корпус)
# ══════════════════════════════════════════════════════════════
def collect_english(output_path: str, target_bytes: int) -> None:
    print("\n" + "=" * 60)
    print("  АНГЛИЙСКИЙ ЯЗЫК  —  wikimedia/wikipedia (20231101.en)")
    print("=" * 60)

    ds = load_dataset(
        "wikimedia/wikipedia",
        name="20231101.en",
        split="train",
        streaming=True,
    )

    records, total = collect_from_stream(
        stream=ds,
        text_field="text",
        target_bytes=target_bytes,
        desc="EN wiki",
    )

    mb = total / (1024 * 1024)
    print(f"  Собрано записей: {len(records):,}  |  {mb:.1f} МБ")

    save_jsonl(records, output_path)
    print(f"  Сохранено → {output_path}")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Загрузка английского контрольного корпуса из HuggingFace")
    p.add_argument("--output_dir", type=str, default="./data/raw_sources",
                    help="Директория для сохранения JSONL файла")
    p.add_argument("--en_file", type=str, default="english_raw.jsonl",
                    help="Имя выходного файла для английского")
    p.add_argument("--target_mb", type=int, default=DEFAULT_TARGET_MB,
                    help="Целевой объём в МБ (default: 150)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    target_bytes = args.target_mb * 1024 * 1024
    os.makedirs(args.output_dir, exist_ok=True)

    en_path = os.path.join(args.output_dir, args.en_file)

    print(f"Целевой объём: {args.target_mb} МБ чистого текста")
    print(f"Фильтр длины:  {MIN_LEN}–{MAX_LEN} символов")
    print(f"Выход:          {args.output_dir}/")

    collect_english(en_path, target_bytes)

    print("\n" + "=" * 60)
    print("  ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()
