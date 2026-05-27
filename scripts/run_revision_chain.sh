#!/bin/bash
# Revision Block 3: seed-replication chain (closes L1 single-seed claims).
#
# Cheapest-first ordering so we learn fast if anything breaks. Each run
# auto-skips if its output directory already exists, so the script can
# be interrupted (Ctrl-C / SIGTERM) and re-run idempotently.
#
# Order  | ID         | Model      | Setup                                    | ~Runtime
# ───────┼────────────┼────────────┼──────────────────────────────────────────┼─────────
#  1     | E2-s123    | Gemma-2-9B | UZ baseline, r=16, alpha=32, lr=2e-4, 3ep | ~6h
#  2     | L3-E5-s123 | Llama-3-8B | KY collapse, r=64, alpha=128, lr=5e-4, 5ep| ~8h
#  3     | E1-s123    | Gemma-2-9B | KZ baseline, r=16, alpha=32, lr=2e-4, 3ep | ~8h
#
# Total ~22h. After completion, the L1 limitation drops three more
# single-seed configs to n=2.
#
# Usage:
#   nohup bash scripts/run_revision_chain.sh > revision_chain.nohup.out 2>&1 &
#   echo $! > revision_chain.pid

set -euo pipefail

cd "$(dirname "$0")/.."

GEMMA="google/gemma-2-9b"
LLAMA3="NousResearch/Meta-Llama-3-8B"
SEED=123

run_one() {
    local model=$1
    local out=$2
    local lora_r=$3
    local lora_alpha=$4
    local lr=$5
    local epochs=$6
    local ky_file=$7
    local kz_file=$8
    local uz_file=$9
    local label=${10}

    if [ -d "$out" ]; then
        echo "[SKIP] $out already exists -- $label"
        return 0
    fi

    echo "============================================================"
    echo "[RUN] $label  ->  $out"
    echo "      model=$model  r=$lora_r  alpha=$lora_alpha  lr=$lr  epochs=$epochs  seed=$SEED"
    echo "============================================================"

    ~/anaconda3/envs/collapse/bin/python scripts/train_svd.py \
        --model_name "$model" \
        --data_dir ./data/pretrain \
        --ky_file "$ky_file" \
        --kz_file "$kz_file" \
        --uz_file "$uz_file" \
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

    echo "[OK] finished $label"
}

# ─────────────────────────────────────────────────────────────────────
# Order: cheapest first.
# ─────────────────────────────────────────────────────────────────────

# 1. E2-s123 (Gemma UZ baseline, seed=123). ~6h.
run_one  "$GEMMA"  \
         "output_uz_baseline_r16_lr2e4_3ep_seed123" \
         16  32  2e-4  3 \
         __not_present__.jsonl  __not_present__.jsonl  uzbek_final_cyrillic.jsonl \
         "E2-s123 (Gemma UZ baseline seed=123)"

# 2. L3-E5-s123 (Llama-3 KY collapse, seed=123). ~8h.
run_one  "$LLAMA3" \
         "output_l3_ky_collapse_r64_lr5e4_5ep_seed123" \
         64 128  5e-4  5 \
         kyrgyz_raw.jsonl  __not_present__.jsonl  __not_present__.jsonl \
         "L3-E5-s123 (Llama-3 collapse seed=123)"

# 3. E1-s123 (Gemma KZ baseline, seed=123). ~8h.
run_one  "$GEMMA"  \
         "output_kz_baseline_r16_lr2e4_3ep_seed123" \
         16  32  2e-4  3 \
         __not_present__.jsonl  kazakh_raw.jsonl  __not_present__.jsonl \
         "E1-s123 (Gemma KZ baseline seed=123)"

echo
echo "============================================================"
echo "[DONE] revision chain finished. Three configurations now at n=2:"
echo "       E2 (UZ baseline), L3-E5 (Llama-3 collapse), E1 (KZ baseline)."
echo "       Next: run evaluate.py + pairwise cosines for the new seeds."
echo "============================================================"
