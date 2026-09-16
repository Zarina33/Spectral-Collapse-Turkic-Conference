#!/bin/bash
# Wait for the Gemma-2-9B snapshot download to finish, then run the held-out chain.
# Usage: nohup bash scripts/start_heldout_when_ready.sh > heldout_queue.nohup.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
CACHE=$HOME/.cache/huggingface/hub/models--google--gemma-2-9b

echo "[QUEUE] $(date -Is) waiting for the download to complete"
while :; do
    if grep -q "DOWNLOAD COMPLETE" gemma_predownload.log 2>/dev/null; then
        echo "[QUEUE] $(date -Is) download reported complete"; break
    fi
    if ! pgrep -f snapshot_download >/dev/null 2>&1; then
        echo "[QUEUE] $(date -Is) download process gone; checking files"
        n=$(ls "$CACHE"/snapshots/*/*.safetensors 2>/dev/null | wc -l)
        inc=$(ls "$CACHE"/blobs/*.incomplete 2>/dev/null | wc -l)
        if [ "$n" -ge 8 ] && [ "$inc" -eq 0 ]; then echo "[QUEUE] files present"; break; fi
        echo "[QUEUE] ERROR: download died with $n/8 safetensors, $inc partial. Not starting."; exit 1
    fi
    sleep 60
done

echo "[QUEUE] $(date -Is) starting held-out chain"
exec bash scripts/run_heldout_probe.sh
