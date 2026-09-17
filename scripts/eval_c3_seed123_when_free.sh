#!/bin/bash
# Evaluate the BF16 second seed (C3-s123) locally with the standard evaluate.py,
# i.e. on the 4-bit NF4 base -- the same protocol the seed-42 C3 adapter was
# evaluated with (its report was produced by the unmodified evaluate.py, which
# has no unquantized path). Waits until no foreign job holds the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/anaconda3/envs/collapse/bin/python
D=output_ky_bf16_r16_lr2e4_3ep_seed123
echo "[QUEUE] $(date -Is) waiting for a free GPU (<3 GB used)"
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 3000 ]; do sleep 120; done
echo "[EVAL] $(date -Is) starting"
$PY scripts/evaluate.py --model_name google/gemma-2-9b --adapter_path "$D/final_adapter" > "$D/eval.log" 2>&1
rc=$?
if [ -f output/eval_report.json ]; then mkdir -p "$D/eval"; cp output/eval_report.json "$D/eval/eval_report.json"; mv output/eval_report.json "$D/eval_report.json"; echo "[OK] $(date -Is) eval_report.json -> $D/"; else echo "[FAIL] rc=$rc, see $D/eval.log"; fi
