#!/bin/bash
# Revision Block 1: Llama-3-8B C5 control (absolute alpha=32 at r=64).
#
# Closes the architecture-independence gap on the C5 finding: Gemma showed
# that the "rank-only" benefit of E5c (r=64, alpha=128) over E3 (r=16,
# alpha=32) was an artefact of the alpha=2r convention amplifying absolute
# alpha at higher rank, not of rank itself. Holding absolute alpha=32 at
# r=64 made the adapter functionally indistinguishable from the r=16
# baseline. That control was Gemma-only -- this run replicates it on
# Llama-3-8B.
#
# L3-C5 settings: r=64, alpha=32 (absolute, NOT alpha=2r), lr=2e-4, 3 epochs.
# Compared against:
#   L3-E3  (r=16, alpha=32,  lr=2e-4, 3ep)  -> baseline at same absolute alpha
#   L3-E5c (r=64, alpha=128, lr=2e-4, 3ep)  -> "rank-only healthy" with alpha=2r
#
# Prediction (if the absolute-alpha story is architecture-independent):
#   L3-C5 PPL ~ L3-E3 PPL (both at absolute alpha=32)
#   L3-C5 PPL > L3-E5c PPL (E5c had 4x larger absolute alpha)
#
# Runtime estimate on RTX 5080 (4-bit NF4): ~6-8h.
# Output: ./output_l3_ky_r64_lr2e4_3ep_alpha32/
#
# Usage (overnight):
#   nohup bash scripts/run_llama3_c5.sh > l3_c5.nohup.out 2>&1 &
#   echo $! > l3_c5.pid

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="NousResearch/Meta-Llama-3-8B"
OUT=output_l3_ky_r64_lr2e4_3ep_alpha32

if [ -d "$OUT" ]; then
    echo "[WARN] $OUT already exists. Move or delete it before re-running."
    exit 1
fi

echo "============================================================"
echo "[RUN] L3-C5 (absolute alpha=32 at r=64)"
echo "      model=$MODEL  r=64  alpha=32  lr=2e-4  epochs=3  seed=42"
echo "      out=$OUT"
echo "============================================================"

~/anaconda3/envs/collapse/bin/python scripts/train_svd.py \
    --model_name "$MODEL" \
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
    --seed 42

echo
echo "============================================================"
echo "[DONE] L3-C5 finished. Adapter at $OUT/final_adapter/"
echo "Next: run evaluate.py on this adapter, then compare against"
echo "  L3-E3  (output_l3_ky_baseline_r16_lr2e4_3ep/) at same absolute alpha"
echo "  L3-E5c (output_l3_ky_r64_lr2e4_3ep/)         at alpha=2r convention"
echo "to test whether the absolute-alpha story holds on Llama-3."
echo "============================================================"
