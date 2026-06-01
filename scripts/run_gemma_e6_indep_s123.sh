#!/bin/bash
# Revision Block 5: Gemma E6 second seed with INDEPENDENT KZ init (E1-s123).
#
# Critical apples-to-apples test. The existing Gemma E6 cross-seed cosine
# of 0.785 was measured with both seeds warm-starting from the SAME source
# (E1-s42), because E1-s123 did not exist at the time. Verified by:
#   cos(Gemma-E6-s123, Gemma-E1-s42)  = 0.7929  <- shared init
#   cos(Gemma-E6-s123, Gemma-E1-s123) = 0.0209  <- independent init
#
# The Llama-3 E6 cross-seed used INDEPENDENT KZ inits (L3-KZ-s42 + L3-KZ-s123)
# and gave 0.060. This script runs the architecture-fair Gemma equivalent:
# warm-start at seed 123 from E1-s123 (not E1-s42) so that the cross-seed
# comparison cos(E6-s42-from-E1-s42, E6-indep-s123-from-E1-s123) is the
# strict test of whether transfer pins direction across independent runs.
#
# Setup: r=16, alpha=32, lr=2e-4, 3 epochs, seed=123.
# Init:  output_kz_baseline_r16_lr2e4_3ep_seed123/final_adapter (E1-s123).
# Data:  kyrgyz_raw.jsonl (4.4M tokens).
# Out:   output_ky_from_kz_r16_lr2e4_3ep_indep_seed123/
# Runtime ~6h on RTX 5080 (matches the original Gemma E6 wall time).
#
# Expected outcomes:
#   cos(E6-s42, E6-indep-s123) ~ 0.06  -> shared-init was the artifact;
#                                        transfer doesn't pin across independent runs.
#                                        Architectures agree (Llama-3 result holds).
#   cos(E6-s42, E6-indep-s123) ~ 0.78  -> Gemma genuinely differs from Llama-3;
#                                        transfer-pinning is architecture-dependent.
#
# Usage:
#   nohup bash scripts/run_gemma_e6_indep_s123.sh > gemma_e6_indep.nohup.out 2>&1 &
#   echo $! > gemma_e6_indep.pid

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="google/gemma-2-9b"
SEED=123
INIT="output_kz_baseline_r16_lr2e4_3ep_seed123/final_adapter"
OUT=output_ky_from_kz_r16_lr2e4_3ep_indep_seed123

if [ ! -d "$INIT" ]; then
    echo "[FATAL] init adapter $INIT not found. Cannot run independent-init test."
    exit 1
fi
if [ -d "$OUT" ]; then
    echo "[WARN] $OUT already exists. Move or delete it before re-running."
    exit 1
fi

echo "============================================================"
echo "[RUN] Gemma E6-indep-s123 (architecture-fair cross-seed test)"
echo "      model=$MODEL  init=$INIT  seed=$SEED"
echo "      r=16  alpha=32  lr=2e-4  epochs=3"
echo "      out=$OUT"
echo "============================================================"

~/anaconda3/envs/collapse/bin/python scripts/train_svd.py \
    --model_name "$MODEL" \
    --data_dir ./data/pretrain \
    --ky_file kyrgyz_raw.jsonl \
    --kz_file __not_present__.jsonl \
    --uz_file __not_present__.jsonl \
    --output_dir "$OUT" \
    --init_adapter "$INIT" \
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
echo "[DONE] Gemma E6-indep-s123 finished. Adapter at $OUT/final_adapter/"
echo "Next: evaluate.py + pairwise_cosine to compute"
echo "      cos(E6_s42, E6_indep_s123) -- the architecture-fair cross-seed result."
echo "============================================================"
