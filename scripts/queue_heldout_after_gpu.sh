#!/bin/bash
# Queue the held-out probe chain behind (1) the Gemma-2-9B weight download and
# (2) any foreign GPU job. Retries the chain once if a run failed.
# Usage: nohup bash scripts/queue_heldout_after_gpu.sh > heldout_queue.nohup.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/anaconda3/envs/collapse/bin/python
CACHE=$HOME/.cache/huggingface/hub/models--google--gemma-2-9b

echo "[QUEUE] $(date -Is) waiting for Gemma-2-9B download"
until [ -d "$CACHE/snapshots" ] && [ -z "$(ls "$CACHE"/blobs/*.incomplete 2>/dev/null)" ] \
      && [ "$(ls "$CACHE"/snapshots/*/*.safetensors 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
echo "[QUEUE] $(date -Is) weights present ($(du -sh "$CACHE" | cut -f1))"

echo "[QUEUE] $(date -Is) waiting for foreign GPU jobs to finish"
while nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | awk -F', ' '$2>1000{f=1} END{exit !f}'; do sleep 120; done
echo "[QUEUE] $(date -Is) GPU free, starting chain"

for attempt in 1 2; do
    bash scripts/run_heldout_probe.sh
    missing=0
    for d in output_ky_heldout_r16_lr1e3_3ep output_ky_heldout_r32_lr5e4_5ep output_ky_heldout_r64_lr3p5e4_5ep; do
        [ -f "$d/eval_report.json" ] || missing=$((missing+1))
    done
    echo "[QUEUE] attempt $attempt done, $missing run(s) without eval_report"
    [ $missing -eq 0 ] && break
    sleep 60
done
echo "[QUEUE] $(date -Is) finished"
