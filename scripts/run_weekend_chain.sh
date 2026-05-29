#!/bin/bash
# Weekend autopilot: runs the full remaining revision queue unattended.
#
# Waits for the currently-running L3-E6-s123 pipeline (l3_e6_s123.pid) to
# release the GPU, then runs every remaining experiment + its eval, and a
# final pairwise-cosine pass. Everything is idempotent (skip-if-exists), so
# the script can be interrupted and re-run safely.
#
# Queue (after L3-E6-s123 finishes):
#   A. eval L3-E6-s123                                         ~50 min
#   B. E2b   Gemma UZ small-corpus (~1.5M tok)  + eval   ~2 h + 50 min
#   C. KZ-4.4M Gemma KZ token-matched-to-KY     + eval   ~3 h + 50 min
#   D. E4-s123 Gemma KY overfit (10 epochs)     + eval  ~20 h + 50 min
#   E. final pairwise_cosine.py (all pairs, incl. L3-E6 cross-seed)  ~10 min
#
# Order: finish the in-flight experiment first (A), then the cheap L5
# controls (B, C), then the expensive low-priority E4 (D) last.
#
# Usage:
#   nohup bash scripts/run_weekend_chain.sh > weekend_chain.nohup.out 2>&1 &
#   echo $! > weekend_chain.pid

set -uo pipefail   # NB: no -e; one failing run must not abort the whole queue.

cd "$(dirname "$0")/.."

PY=~/anaconda3/envs/collapse/bin/python
GEMMA="google/gemma-2-9b"
LLAMA3="NousResearch/Meta-Llama-3-8B"

log() { echo "[$(date '+%F %T')] $*"; }

# ─────────────────────────────────────────────────────────────────────
# Step 0: wait for the in-flight L3-E6-s123 pipeline to release the GPU.
# ─────────────────────────────────────────────────────────────────────
if [ -f l3_e6_s123.pid ]; then
    P=$(cat l3_e6_s123.pid)
    if kill -0 "$P" 2>/dev/null; then
        log "WAIT: L3-E6-s123 (PID $P) still running; blocking until it exits..."
        tail --pid="$P" -f /dev/null
        log "WAIT: L3-E6-s123 finished. Starting weekend queue."
    fi
fi

# ── helper: train one config (skip if output dir exists) ──────────────
train_one() {
    local model=$1 out=$2 r=$3 a=$4 lr=$5 ep=$6 ky=$7 kz=$8 uz=$9 seed=${10} label=${11}
    if [ -d "$out" ]; then log "SKIP train: $out exists -- $label"; return 0; fi
    log "TRAIN: $label -> $out (r=$r a=$a lr=$lr ep=$ep seed=$seed)"
    $PY scripts/train_svd.py \
        --model_name "$model" --data_dir ./data/pretrain \
        --ky_file "$ky" --kz_file "$kz" --uz_file "$uz" \
        --output_dir "$out" \
        --lora_r "$r" --lora_alpha "$a" --lora_dropout 0.0 \
        --max_seq_length 256 --num_train_epochs "$ep" \
        --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 --learning_rate "$lr" \
        --warmup_ratio 0.05 --logging_steps 10 --eval_steps 200 \
        --save_steps 200 --svd_every_steps 100 --seed "$seed" \
        && log "OK train: $label" || log "FAIL train: $label (continuing)"
}

# ── helper: eval one adapter (skip if eval_report.json exists) ────────
eval_one() {
    local model=$1 dir=$2 label=$3
    if [ -f "$dir/eval_report.json" ]; then log "SKIP eval: $dir/eval_report.json exists -- $label"; return 0; fi
    if [ ! -d "$dir/final_adapter" ]; then log "WARN eval: $dir/final_adapter missing -- skip $label"; return 0; fi
    log "EVAL: $label -> $dir"
    $PY scripts/evaluate.py --model_name "$model" --adapter_path "$dir/final_adapter" \
        && { [ -f output/eval_report.json ] && mv output/eval_report.json "$dir/eval_report.json"; log "OK eval: $label"; } \
        || log "FAIL eval: $label (continuing)"
}

# ─────────────────────────────────────────────────────────────────────
# A. eval the just-finished L3-E6-s123 transfer adapter.
# ─────────────────────────────────────────────────────────────────────
eval_one "$LLAMA3" "output_l3_ky_from_kz_r16_lr2e4_3ep_seed123" "L3-E6-s123 (KZ->KY transfer)"

# ─────────────────────────────────────────────────────────────────────
# B. E2b -- Gemma UZ small-corpus (~1.5M tok), mirror of E1b. L5 closure (UZ).
# ─────────────────────────────────────────────────────────────────────
train_one "$GEMMA" "output_uz_tokenmatched_r16_lr2e4_3ep" \
          16 32 2e-4 3  __not_present__.jsonl __not_present__.jsonl uzbek_small.jsonl \
          42 "E2b (UZ small-corpus ~1.5M)"
eval_one  "$GEMMA" "output_uz_tokenmatched_r16_lr2e4_3ep" "E2b (UZ small-corpus)"

# ─────────────────────────────────────────────────────────────────────
# C. KZ-4.4M -- Gemma KZ subsampled to KY's token budget. L5 closure (KZ).
# ─────────────────────────────────────────────────────────────────────
train_one "$GEMMA" "output_kz_4p4M_r16_lr2e4_3ep" \
          16 32 2e-4 3  __not_present__.jsonl kazakh_4p4M.jsonl __not_present__.jsonl \
          42 "KZ-4.4M (true token-matched-to-KY control)"
eval_one  "$GEMMA" "output_kz_4p4M_r16_lr2e4_3ep" "KZ-4.4M control"

# ─────────────────────────────────────────────────────────────────────
# D. E4-s123 -- Gemma KY overfit (10 epochs), second seed. Last Gemma n=1.
# ─────────────────────────────────────────────────────────────────────
train_one "$GEMMA" "output_ky_overfit_r16_lr2e4_10ep_seed123" \
          16 32 2e-4 10  kyrgyz_raw.jsonl __not_present__.jsonl __not_present__.jsonl \
          123 "E4-s123 (KY overfit, 10ep)"
eval_one  "$GEMMA" "output_ky_overfit_r16_lr2e4_10ep_seed123" "E4-s123 (KY overfit)"

# ─────────────────────────────────────────────────────────────────────
# E. final pairwise cosines (includes L3-E6 cross-seed + everything new).
# ─────────────────────────────────────────────────────────────────────
log "COSINE: recomputing all pairwise cosines"
mkdir -p directional_results
$PY scripts/pairwise_cosine.py --output directional_results/pairwise_weekend.json \
    && log "OK cosine" || log "FAIL cosine"

log "DONE: weekend queue complete."
echo "============================================================"
echo "[DONE] Weekend chain finished. New artifacts:"
echo "  output_l3_ky_from_kz_r16_lr2e4_3ep_seed123/eval_report.json"
echo "  output_uz_tokenmatched_r16_lr2e4_3ep/        (E2b + eval)"
echo "  output_kz_4p4M_r16_lr2e4_3ep/                (KZ-4.4M + eval)"
echo "  output_ky_overfit_r16_lr2e4_10ep_seed123/    (E4-s123 + eval)"
echo "  directional_results/pairwise_weekend.json"
echo "============================================================"
