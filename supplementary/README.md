# Supplementary Code

Code accompanying *"Spectral Energy Is Not a Reliable Collapse Diagnostic for Low-Resource Quantized LoRA: Evidence from Turkic Languages"*.

## Files

| File | Description |
|------|-------------|
| `train_svd.py` | LoRA fine-tuning with per-step SVD monitoring (SE, effective rank, stable rank, SVD entropy, Frobenius norms) |
| `evaluate.py` | Post-training evaluation: cross-lingual PPL, WikiANN NER (gen + log-likelihood span typing), TUMLU QA (5-shot) |
| `pairwise_cosine.py` | Memory-efficient per-module Frobenius cosine between adapter pairs (never materialises $BA$) |
| `pairwise_cosine_lastckpt.py` | Robustness check against literal last checkpoint |
| `trajectory_cosine.py` | Per-checkpoint pairwise cosine across the training trajectory |
| `directional_metric.py` / `directional_metric_v2.py` | Negative-result probes: top-$r$ alignment with $W_0$, top-256 alignment, $\cos(BA, W_0)$ |
| `plot_final.py` | Publication figures 1–9 |
| `plot_pairwise_cosine.py` | Figure 10 (central directional figure) |
| `run_alpha32_control.sh` | C5 control (α=32 at r=64) launcher |
| `run_llama3_minimum.sh` / `run_llama3_seed123.sh` | Llama-3-8B replication launchers |
| `requirements.txt` | Python dependencies |

## Usage

### Training with SVD monitoring
```bash
python train_svd.py \
  --model_name google/gemma-2-9b \
  --data_dir ./data/pretrain \
  --ky_file kyrgyz_raw.jsonl \
  --output_dir ./output_ky_baseline_r16_lr2e4_3ep \
  --lora_r 16 --learning_rate 2e-4 --num_train_epochs 3 \
  --seed 42
```

For the Llama-3-8B replication, swap `--model_name NousResearch/Meta-Llama-3-8B`.

### Evaluation

```bash
python evaluate.py \
  --model_name google/gemma-2-9b \
  --adapter_path ./output_ky_baseline_r16_lr2e4_3ep/final_adapter
```

### Pairwise-cosine directional analysis

```bash
python pairwise_cosine.py --output directional_results/pairwise.json
```

The pairs sweep (22 comparisons including the bimodal transfer-init vs.\ direct-training contrast) writes per-module cosines to JSON and prints a summary table.

### Figures
```bash
python plot_final.py
python plot_pairwise_cosine.py
```

## SVD log format

Each line in `output_*/svd_log.jsonl` (logged every 100 steps, one line per LoRA module) contains:
```json
{
  "step": 100,
  "layer_name": "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
  "layer_index": 0,
  "singular_values": [...],
  "spectral_energy": 0.142,
  "effective_rank": 12.3,
  "stable_rank": 8.7,
  "svd_entropy": 3.42,
  "frobenius_norm_A": 1.23,
  "frobenius_norm_B": 0.45,
  "vram_allocated_gb": 7.94,
  "tokens_per_second": 1780.1
}
```

## Eval report format

`evaluate.py` writes a single `eval_report.json` with sub-keys `perplexity`, `ner_wikiann`, `ner_loglik`, `tumlu_qa`. Each sub-key has per-language entries (`ky`, `kz`, `uz`).
