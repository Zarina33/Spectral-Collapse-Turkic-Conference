#!/bin/bash
# Revision Block 2: Llama-3-8B E6 (KZ -> KY transfer) replication.
#
# Closes the second "Gemma-only" architecture-independence gap. The
# transfer-pinning side of the bimodality (cross-seed cos approx 0.78
# for E6) is currently established on Gemma only; this run replicates
# the KZ -> KY warm-start setup on Llama-3-8B.
#
# Two-stage pipeline (Gemma E6 mirror):
#   Stage 1 (L3-KZ baseline): KZ pretrain on Llama-3-8B, r=16, alpha=32,
#                             lr=2e-4, 3 epochs. Same setup as Gemma E1.
#                             Output: output_l3_kz_baseline_r16_lr2e4_3ep/
#                             Runtime ~5h on RTX 5080 (14.1M KZ tokens).
#
#   Stage 2 (L3-E6 KZ->KY):   Continue training the KZ adapter on KY
#                             corpus, same hyperparameters. Output:
#                             output_l3_ky_from_kz_r16_lr2e4_3ep/
#                             Runtime ~5h (4.4M KY tokens).
#
# Total runtime: ~10-12h.
#
# Prediction (if Gemma's bimodality holds on Llama-3):
#   - L3-E6 KZ PPL << L3-E3 KZ PPL (warm-start retains source language)
#   - L3-E6 vs L3-E3 directional cosine > L3-E3 cross-seed cosine
#     (i.e. transfer-init pins direction even when seeds differ)
#
# Usage (overnight):
#   nohup bash scripts/run_llama3_e6.sh > l3_e6.nohup.out 2>&1 &
#   echo $! > l3_e6.pid

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="NousResearch/Meta-Llama-3-8B"
SEED=42

OUT_KZ=output_l3_kz_baseline_r16_lr2e4_3ep
OUT_E6=output_l3_ky_from_kz_r16_lr2e4_3ep

# ─────────────────────────────────────────────────────────────────────
# Stage 1: L3-KZ baseline (Llama-3 mirror of Gemma E1)
# ─────────────────────────────────────────────────────────────────────
if [ -d "$OUT_KZ" ]; then
    echo "[INFO] $OUT_KZ already exists, skipping stage 1."
else
    echo "============================================================"
    echo "[STAGE 1] L3-KZ baseline (source language pretrain)"
    echo "      model=$MODEL  r=16  alpha=32  lr=2e-4  epochs=3  seed=$SEED"
    echo "      out=$OUT_KZ"
    echo "============================================================"

    ~/anaconda3/envs/collapse/bin/python scripts/train_svd.py \
        --model_name "$MODEL" \
        --data_dir ./data/pretrain \
        --kz_file kazakh_raw.jsonl \
        --ky_file __not_present__.jsonl \
        --uz_file __not_present__.jsonl \
        --output_dir "$OUT_KZ" \
        --lora_r 16 \
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
        --seed "$SEED"

    echo "[OK] Stage 1 done. KZ adapter at $OUT_KZ/final_adapter/"
fi

# ─────────────────────────────────────────────────────────────────────
# Stage 2: L3-E6 KZ -> KY warm-start
# ─────────────────────────────────────────────────────────────────────
if [ -d "$OUT_E6" ]; then
    echo "[WARN] $OUT_E6 already exists, skipping stage 2."
    exit 0
fi

echo "============================================================"
echo "[STAGE 2] L3-E6 KZ -> KY transfer (warm-start)"
echo "      model=$MODEL  init=$OUT_KZ/final_adapter"
echo "      r=16  alpha=32  lr=2e-4  epochs=3  seed=$SEED"
echo "      out=$OUT_E6"
echo "============================================================"

~/anaconda3/envs/collapse/bin/python scripts/train_svd.py \
    --model_name "$MODEL" \
    --data_dir ./data/pretrain \
    --ky_file kyrgyz_raw.jsonl \
    --kz_file __not_present__.jsonl \
    --uz_file __not_present__.jsonl \
    --output_dir "$OUT_E6" \
    --init_adapter "$OUT_KZ/final_adapter" \
    --lora_r 16 \
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
    --seed "$SEED"

echo
echo "============================================================"
echo "[DONE] L3-E6 (KZ -> KY) finished. Adapter at $OUT_E6/final_adapter/"
echo "Next: run evaluate.py + pairwise_cosine vs L3-E3 to test"
echo "      transfer-pinning bimodality on Llama-3."
echo "============================================================"
