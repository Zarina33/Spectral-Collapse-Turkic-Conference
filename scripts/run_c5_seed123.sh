#!/bin/bash
# Gemma C5 (absolute alpha=32 at r=64) second seed, then evaluation.
# Same settings as scripts/run_alpha32_control.sh (seed 42), seed 123.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/anaconda3/envs/collapse/bin/python
OUT=output_ky_r64_lr2e4_3ep_alpha32_seed123
if [ ! -d "$OUT/final_adapter" ]; then
$PY scripts/train_svd.py --model_name google/gemma-2-9b --data_dir ./data/pretrain \
    --ky_file kyrgyz_raw.jsonl --kz_file __not_present__.jsonl --uz_file __not_present__.jsonl \
    --output_dir "$OUT" --lora_r 64 --lora_alpha 32 --lora_dropout 0.0 \
    --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 2e-4 \
    --warmup_ratio 0.05 --logging_steps 10 --eval_steps 200 --save_steps 200 \
    --svd_every_steps 100 --seed 123 > "$OUT.launch.log" 2>&1 || { echo "[FAIL] training"; exit 1; }
fi
echo "[OK] training done $(date -Is)"
$PY scripts/evaluate.py --model_name google/gemma-2-9b --adapter_path "$OUT/final_adapter" > "$OUT/eval.log" 2>&1
[ -f output/eval_report.json ] && mv output/eval_report.json "$OUT/eval_report.json" && echo "[OK] eval done $(date -Is)" || echo "[FAIL] eval"
