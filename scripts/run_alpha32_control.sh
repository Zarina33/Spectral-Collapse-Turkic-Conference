#!/bin/bash
# Block 4a (Limitation L3): alpha=32 control at r=64.
#
# Closes the alpha/r confound that E5b/E5c leave open. E5c uses alpha/r=2
# (alpha=128 at r=64), so its effective learning rate at adapter weights
# differs by 4x from the r=16 baseline (alpha=32). This run holds the
# effective LR constant by using alpha=32 at r=64 to match E3's alpha=32
# at r=16. Everything else is identical to E5c.
#
# Expected runtime: ~5 hours on RTX 5080 (16 GB), 4-bit NF4.
# Output: ./output_ky_r64_lr2e4_3ep_alpha32/
#
# Run from repo root.

set -euo pipefail

cd "$(dirname "$0")/.."

OUT=output_ky_r64_lr2e4_3ep_alpha32

if [ -d "$OUT" ]; then
    echo "[WARN] $OUT already exists. Move or delete it before re-running."
    exit 1
fi

python scripts/train_svd.py \
    --model_name google/gemma-2-9b \
    --data_dir ./data/pretrain \
    --ky_file kyrgyz_raw.jsonl \
    --kz_file __not_present__.jsonl \
    --uz_file __not_present__.jsonl \
    --output_dir "$OUT" \
    --lora_r 64 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --max_seq_length 256 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --eval_steps 200 \
    --save_steps 200 \
    --svd_every_steps 100 \
    --seed 42 \
    2>&1 | tee "$OUT.launch.log"

echo
echo "[DONE] alpha=32 control finished. Adapter at $OUT/final_adapter/"
echo "Compare against E5c (output_ky_r64_lr2e4_3ep/) on KY PPL, NER, TUMLU,"
echo "cross-lingual PPL, and (sec 4.6 protocol) pairwise cosine."
