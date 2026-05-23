#!/bin/bash
# Block 2 (minimum viable): Llama-3-8B replication of the three critical configs.
#
# E3-l3 : r=16, alpha=32, lr=2e-4, 3 epochs (baseline)
# E5c-l3: r=64, alpha=128, lr=2e-4, 3 epochs (rank-only healthy)
# E5-l3 : r=64, alpha=128, lr=5e-4, 5 epochs (catastrophic collapse)
#
# Goal: test whether the bimodal-direction + SE-falsification pattern from
# Gemma-2-9B reproduces on Llama-3-8B. If yes, the structural claim is
# architecture-independent; if no, scope statement in Limitations.
#
# Runtime estimate on RTX 5080 (4-bit NF4):
#   E3-l3   ~5h
#   E5c-l3  ~6h
#   E5-l3   ~8h
#   Total   ~19h continuous
#
# Order: cheapest first, so we learn fast whether anything fundamental
# breaks before committing to the long collapse run.
#
# Usage:
#   bash scripts/run_llama3_minimum.sh
#   # or to keep going after SSH disconnect:
#   nohup bash scripts/run_llama3_minimum.sh > l3_minimum.nohup.out 2>&1 &
#   echo $! > l3_minimum.pid

set -euo pipefail

cd "$(dirname "$0")/.."

# Original is meta-llama/Meta-Llama-3-8B (gated). NousResearch hosts a
# byte-identical mirror without gating; we cite Meta as the source.
MODEL="NousResearch/Meta-Llama-3-8B"

run_one() {
    local out=$1
    local lora_r=$2
    local lora_alpha=$3
    local lr=$4
    local epochs=$5
    local label=$6

    if [ -d "$out" ]; then
        echo "[WARN] $out already exists, skipping. Move/delete to force re-run."
        return 0
    fi

    echo "============================================================"
    echo "[RUN] $label  ->  $out"
    echo "      model=$MODEL  r=$lora_r  alpha=$lora_alpha  lr=$lr  epochs=$epochs"
    echo "============================================================"

    ~/anaconda3/envs/collapse/bin/python scripts/train_svd.py \
        --model_name "$MODEL" \
        --data_dir ./data/pretrain \
        --ky_file kyrgyz_raw.jsonl \
        --kz_file __not_present__.jsonl \
        --uz_file __not_present__.jsonl \
        --output_dir "$out" \
        --lora_r "$lora_r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout 0.0 \
        --max_seq_length 256 \
        --num_train_epochs "$epochs" \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate "$lr" \
        --warmup_ratio 0.05 \
        --logging_steps 10 \
        --eval_steps 200 \
        --save_steps 200 \
        --svd_every_steps 100 \
        --seed 42

    echo "[OK] finished $label  ->  $out"
}

# Run order: cheapest first.
run_one  output_l3_ky_baseline_r16_lr2e4_3ep     16  32  2e-4  3  "L3-E3 (baseline)"
run_one  output_l3_ky_r64_lr2e4_3ep              64 128  2e-4  3  "L3-E5c (rank-only healthy)"
run_one  output_l3_ky_collapse_r64_lr5e4_5ep     64 128  5e-4  5  "L3-E5 (collapse)"

echo
echo "============================================================"
echo "[DONE] All three Llama-3-8B runs finished."
echo "       Next: run scripts/evaluate.py on each final_adapter,"
echo "       then compute pairwise cosines L3-E3 vs L3-E5c (the"
echo "       critical SE-falsification + directional test)."
echo "============================================================"
