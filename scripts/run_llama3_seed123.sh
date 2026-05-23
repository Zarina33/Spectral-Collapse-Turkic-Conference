#!/bin/bash
# Block 2 follow-up: second-seed Llama-3-8B replication for the two configs
# that anchor the cross-seed claim on Gemma.
#
# L3-E3 -s123 : r=16, alpha=32, lr=2e-4, 3 epochs (baseline; cross-seed direct r=16)
# L3-E5c-s123 : r=64, alpha=128, lr=2e-4, 3 epochs (cross-seed direct r=64)
#
# Goal: confirm that the seed-orthogonal regime of direct LoRA training
# (Gemma: cross-seed cos = 0.02-0.04 across 6 configs) reproduces on
# Llama-3. With these two runs we will have:
#   cos(L3-E3-s42, L3-E3-s123)   - direct r=16 cross-seed on L3
#   cos(L3-E5c-s42, L3-E5c-s123) - direct r=64 cross-seed on L3
# alongside the already-computed:
#   cos(L3-E5, L3-E5c) = 0.029   - collapse vs healthy (in the same regime)
#
# Together these establish the directional non-canonicity of direct
# LoRA training on two architectures, not just one.
#
# Runtime estimate on RTX 5080 (4-bit NF4):
#   L3-E3 -s123 ~5h
#   L3-E5c-s123 ~6h
#   Total       ~11h
#
# Usage (overnight):
#   nohup bash scripts/run_llama3_seed123.sh > l3_seed123.nohup.out 2>&1 &
#   echo $! > l3_seed123.pid

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="NousResearch/Meta-Llama-3-8B"
SEED=123

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
    echo "[RUN] $label  ->  $out  (seed=$SEED)"
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
        --seed "$SEED"

    echo "[OK] finished $label  ->  $out"
}

run_one  output_l3_ky_baseline_r16_lr2e4_3ep_seed123  16  32  2e-4  3  "L3-E3 seed=123 (baseline)"
run_one  output_l3_ky_r64_lr2e4_3ep_seed123           64 128  2e-4  3  "L3-E5c seed=123 (rank-only healthy)"

echo
echo "============================================================"
echo "[DONE] Both seed=123 runs finished."
echo "       Next: eval both with --model_name $MODEL,"
echo "       then compute pairwise cosines:"
echo "         cos(L3-E3-s42, L3-E3-s123)"
echo "         cos(L3-E5c-s42, L3-E5c-s123)"
echo "       These should land at ~0.02-0.04 if Gemma's"
echo "       seed-orthogonal regime is architecture-independent."
echo "============================================================"
