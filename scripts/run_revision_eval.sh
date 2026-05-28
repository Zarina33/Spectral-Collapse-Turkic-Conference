#!/bin/bash
# Revision Block 3 follow-up: evaluate the three new seed-123 adapters and
# compute their cross-seed pairwise cosines.
#
# This script WAITS for the training chain (revision_chain.pid) to finish
# before touching the GPU, so it can be launched immediately and will fire
# automatically once E1-s123 completes.
#
# Steps:
#   1. Wait for revision_chain.pid process to exit (if still running).
#   2. evaluate.py on the three new adapters (cross-lingual PPL, NER, TUMLU).
#   3. pairwise_cosine.py -> full JSON including the three new cross-seed pairs.
#
# Usage:
#   nohup bash scripts/run_revision_eval.sh > revision_eval.nohup.out 2>&1 &
#   echo $! > revision_eval.pid

set -euo pipefail

cd "$(dirname "$0")/.."

PY=~/anaconda3/envs/collapse/bin/python
GEMMA="google/gemma-2-9b"
LLAMA3="NousResearch/Meta-Llama-3-8B"

# ─────────────────────────────────────────────────────────────────────
# Step 0: wait for the training chain to release the GPU.
# ─────────────────────────────────────────────────────────────────────
if [ -f revision_chain.pid ]; then
    CHAIN_PID=$(cat revision_chain.pid)
    if kill -0 "$CHAIN_PID" 2>/dev/null; then
        echo "[WAIT] training chain (PID $CHAIN_PID) still running; waiting for it to finish..."
        tail --pid="$CHAIN_PID" -f /dev/null
        echo "[WAIT] training chain finished. Starting evaluation."
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# Step 1: evaluate the three new seed-123 adapters.
# eval_report.json is written to ./output/ by evaluate.py, so we move it
# into each run directory after the call.
# ─────────────────────────────────────────────────────────────────────
eval_one() {
    local model=$1
    local run_dir=$2
    local label=$3

    if [ -f "$run_dir/eval_report.json" ]; then
        echo "[SKIP] $run_dir/eval_report.json already exists -- $label"
        return 0
    fi
    if [ ! -d "$run_dir/final_adapter" ]; then
        echo "[WARN] $run_dir/final_adapter missing -- skipping $label"
        return 0
    fi

    echo "============================================================"
    echo "[EVAL] $label -> $run_dir"
    echo "============================================================"
    $PY scripts/evaluate.py \
        --model_name "$model" \
        --adapter_path "$run_dir/final_adapter"

    if [ -f output/eval_report.json ]; then
        mv output/eval_report.json "$run_dir/eval_report.json"
        echo "[OK] eval_report.json -> $run_dir/"
    fi
}

eval_one "$GEMMA"  "output_uz_baseline_r16_lr2e4_3ep_seed123"        "E2-s123 (UZ baseline)"
eval_one "$LLAMA3" "output_l3_ky_collapse_r64_lr5e4_5ep_seed123"     "L3-E5-s123 (Llama-3 collapse)"
eval_one "$GEMMA"  "output_kz_baseline_r16_lr2e4_3ep_seed123"        "E1-s123 (KZ baseline)"

# ─────────────────────────────────────────────────────────────────────
# Step 2: pairwise cosines (includes the three new cross-seed pairs).
# ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[COSINE] recomputing pairwise cosines with new cross-seed pairs"
echo "============================================================"
mkdir -p directional_results
$PY scripts/pairwise_cosine.py \
    --output directional_results/pairwise_revision_block3.json

echo
echo "============================================================"
echo "[DONE] revision eval + cosines finished."
echo "  Eval reports in each output_*_seed123/eval_report.json"
echo "  Cosines in directional_results/pairwise_revision_block3.json"
echo "  New cross-seed pairs: E1_s42-E1_s123, E2_s42-E2_s123,"
echo "                        L3_E5_s42-L3_E5_s123."
echo "============================================================"
