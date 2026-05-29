#!/bin/bash
# Revision Block 4: Llama-3-8B E6 (KZ->KY transfer) SECOND SEED (123).
#
# Gives the Llama-3 transfer cross-seed cosine -- the direct analog of the
# Gemma E6 cross-seed result (cos approx 0.785, 294/294 modules > 0.5). With
# this we can report cos(L3-E6-s42, L3-E6-s123) and confirm that the
# transfer-pinning bimodality reproduces ACROSS SEEDS on Llama-3, not just
# against the source-language initialisation.
#
# Two-stage pipeline at seed 123 (mirrors run_llama3_e6.sh at seed 42):
#   Stage 1 (L3-KZ baseline s123): KZ pretrain, r=16/a=32/lr=2e-4/3ep, ~22h.
#   Stage 2 (L3-E6 KZ->KY s123):   warm-start on KY, same hyperparameters, ~4.5h.
#
# Total runtime ~26h on RTX 5080 16GB.
#
# Prediction (if Gemma's transfer cross-seed bimodality holds on Llama-3):
#   cos(L3-E6-s42, L3-E6-s123) >> direct-training cross-seed band (~0.07);
#   ideally in the transfer-pinned regime, confirming warm-start makes the
#   trajectory reproducible across seeds on Llama-3 too.
#
# Usage (overnight):
#   nohup bash scripts/run_llama3_e6_seed123.sh > l3_e6_s123.nohup.out 2>&1 &
#   echo $! > l3_e6_s123.pid

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="NousResearch/Meta-Llama-3-8B"
SEED=123

OUT_KZ=output_l3_kz_baseline_r16_lr2e4_3ep_seed123
OUT_E6=output_l3_ky_from_kz_r16_lr2e4_3ep_seed123

# ─────────────────────────────────────────────────────────────────────
# Stage 1: L3-KZ baseline, seed 123 (source-language pretrain)
# ─────────────────────────────────────────────────────────────────────
if [ -d "$OUT_KZ" ]; then
    echo "[INFO] $OUT_KZ already exists, skipping stage 1."
else
    echo "============================================================"
    echo "[STAGE 1] L3-KZ baseline seed=123 (source-language pretrain)"
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

    echo "[OK] Stage 1 done. KZ-s123 adapter at $OUT_KZ/final_adapter/"
fi

# ─────────────────────────────────────────────────────────────────────
# Stage 2: L3-E6 KZ -> KY warm-start, seed 123
# ─────────────────────────────────────────────────────────────────────
if [ -d "$OUT_E6" ]; then
    echo "[WARN] $OUT_E6 already exists, skipping stage 2."
    exit 0
fi

echo "============================================================"
echo "[STAGE 2] L3-E6 KZ -> KY transfer seed=123 (warm-start)"
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
echo "[DONE] L3-E6 seed=123 finished. Adapter at $OUT_E6/final_adapter/"
echo "Next: evaluate.py on $OUT_E6, then pairwise cosine"
echo "      cos(L3_E6_s42, L3_E6_s123) -- the L3 transfer cross-seed test."
echo "============================================================"
