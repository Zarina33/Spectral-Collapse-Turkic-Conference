#!/bin/bash
# L3-C5 second seed — closes the last single-seed configuration.
#
# L1 currently reads "4/5 Llama-3 (only L3-C5 single-seed)". Reviewer Fziu
# flagged that the C5 absolute-alpha control was Gemma-only; the seed-42
# Llama-3 run already made it architecture-independent, and this second
# seed makes it seed-reproducible too, bringing Llama-3 to 5/5 at n=2.
#
# Identical to run_llama3_c5.sh except --seed. Batch settings deliberately
# match the seed-42 run (BS=1 x GA=16, gradient checkpointing on) so the
# two seeds are comparable — do NOT "optimize" these for the 5080.
#
# ~4.6 h on RTX 5080 (per REVISION_RESULTS.md section 1).
#
# Usage:
#   nohup bash scripts/run_llama3_c5_seed123.sh > l3_c5_s123.nohup.out 2>&1 &
#   echo $! > l3_c5_s123.pid

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="NousResearch/Meta-Llama-3-8B"
OUT=output_l3_ky_r64_lr2e4_3ep_alpha32_seed123
SEED=123

if [ -d "$OUT" ]; then
    echo "[WARN] $OUT already exists. Move or delete it before re-running."
    exit 1
fi

echo "============================================================"
echo "[RUN] L3-C5-s123 (absolute alpha=32 at r=64, second seed)"
echo "      model=$MODEL  r=64  alpha=32  lr=2e-4  epochs=3  seed=$SEED"
echo "      out=$OUT"
echo "      started: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
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
    --seed "$SEED"

echo
echo "[OK] training finished at $(date -u +'%H:%M:%SZ'). Evaluating..."

~/anaconda3/envs/collapse/bin/python scripts/evaluate.py \
    --model_name "$MODEL" \
    --adapter_path "$OUT/final_adapter" \
    --data_dir ./data/pretrain \
    --output_dir "$OUT" \
    --seed "$SEED"

echo
echo "============================================================"
echo "[DONE] L3-C5-s123 finished at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo
echo "Cross-seed cosine against seed 42:"
echo "  python scripts/pairwise_cosine.py  (add L3_C5_s123 to ADAPTERS/PAIRS)"
echo
echo "Paper updates once the numbers are in:"
echo "  L1        -> Llama-3 now 5/5 at n=2 (drop the L3-C5 single-seed caveat)"
echo "  Sec S     -> add the L3-C5 cross-seed pair"
echo "  tab:llama3_cos -> new row"
echo "============================================================"
