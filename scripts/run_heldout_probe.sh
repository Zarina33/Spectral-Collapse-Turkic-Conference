#!/bin/bash
# Held-out test of the early ||B||_F growth-rate probe (paper Sec. 4.9).
#
# The threshold (g = 0.00653 over steps 100-300) was read off the 31 logged
# runs; all six collapses there are r=64 / alpha=128 / lr>=5e-4 on Kyrgyz.
# These four runs vary exactly the factors the training sample holds fixed,
# and the probe's prediction is written to disk at step 300, *before*
# training finishes and before any evaluation is run:
#
#   H1  KY  r=16  a=32   lr=1e-3    3 ep   rank held out (no r=16 collapse seen)
#   H2  KY  r=32  a=64   lr=5e-4    5 ep   intermediate rank
#   H3  KY  r=64  a=128  lr=3.5e-4  5 ep   intermediate LR, boundary case
#   H4  KZ  r=64  a=128  lr=5e-4    5 ep   language held out (1.5M-token KZ)
#
# Predictions:  heldout_probe_predictions.log  (append-only, timestamped)
# Per-run:      <OUT>/probe_prediction.json, <OUT>/eval_report.json
#
# Usage (repo root):
#   nohup bash scripts/run_heldout_probe.sh > heldout_probe.nohup.out 2>&1 &
#   echo $! > heldout_probe.pid
#
# Expected: ~3-6 h per run on RTX 5080 (4-bit NF4), evaluation on top.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/anaconda3/envs/collapse/bin/python}
MODEL="google/gemma-2-9b"
PRED_LOG=heldout_probe_predictions.log
THRESHOLD=0.00653

run_one() {
    local name=$1 out=$2 kyfile=$3 kzfile=$4 r=$5 alpha=$6 lr=$7 epochs=$8

    if [ -f "$out/eval_report.json" ]; then
        echo "[SKIP] $name: $out/eval_report.json exists"; return 0
    fi

    if [ ! -d "$out/final_adapter" ]; then
        echo "============================================================"
        echo "[TRAIN] $name -> $out   (r=$r alpha=$alpha lr=$lr ep=$epochs)"
        echo "============================================================"
        $PY scripts/train_svd.py \
            --model_name "$MODEL" \
            --data_dir ./data/pretrain \
            --ky_file "$kyfile" --kz_file "$kzfile" --uz_file __not_present__.jsonl \
            --output_dir "$out" \
            --lora_r "$r" --lora_alpha "$alpha" --lora_dropout 0.05 \
            --max_seq_length 256 --num_train_epochs "$epochs" \
            --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 16 \
            --learning_rate "$lr" --warmup_ratio 0.05 \
            --logging_steps 10 --eval_steps 200 --save_steps 200 \
            --svd_every_steps 100 --seed 42 \
            > "$out.launch.log" 2>&1 &
        local pid=$!

        # Record the probe's prediction as soon as step 300 is logged,
        # while training is still running.
        while kill -0 "$pid" 2>/dev/null; do
            if $PY scripts/probe_predict.py "$out" --threshold $THRESHOLD \
                   --out "$out/probe_prediction.json" --log "$PRED_LOG" 2>/dev/null; then
                echo "[PREDICT] $name: $(tail -1 $PRED_LOG)"
                break
            fi
            sleep 120
        done
        wait "$pid"
        local rc=$?
        if [ $rc -ne 0 ]; then
            echo "[FAIL] $name: training exited with $rc (see $out.launch.log)"; return 1
        fi
    fi

    # Prediction fallback if the run finished before the poll caught step 300.
    if [ ! -f "$out/probe_prediction.json" ]; then
        $PY scripts/probe_predict.py "$out" --threshold $THRESHOLD \
            --out "$out/probe_prediction.json" --log "$PRED_LOG" || true
    fi

    echo "[EVAL] $name"
    $PY scripts/evaluate.py --model_name "$MODEL" --adapter_path "$out/final_adapter" \
        > "$out/eval.log" 2>&1
    if [ -f output/eval_report.json ]; then
        mv output/eval_report.json "$out/eval_report.json"
        echo "[OK] $name: eval_report.json -> $out/"
    else
        echo "[WARN] $name: no eval_report.json produced (see $out/eval.log)"
    fi
}

#        name  out                                       ky_file            kz_file                     r   alpha lr      ep
run_one  H1    output_ky_heldout_r16_lr1e3_3ep           kyrgyz_raw.jsonl   __not_present__.jsonl       16  32    1e-3    3
run_one  H2    output_ky_heldout_r32_lr5e4_5ep           kyrgyz_raw.jsonl   __not_present__.jsonl       32  64    5e-4    5
run_one  H3    output_ky_heldout_r64_lr3p5e4_5ep         kyrgyz_raw.jsonl   __not_present__.jsonl       64  128   3.5e-4  5
run_one  H4    output_kzsmall_heldout_r64_lr5e4_5ep      __not_present__.jsonl  kazakh_tokenmatched.jsonl  64  128   5e-4    5

echo
echo "[DONE] held-out probe chain. Predictions:"
cat "$PRED_LOG"
echo "Now compare each prediction with <OUT>/eval_report.json (NER F1, TUMLU, cross-lingual PPL)."
