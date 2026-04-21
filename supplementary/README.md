# Supplementary Code: Spectral Collapse Without Collapse

Code for SVD monitoring, evaluation, and figure generation accompanying the paper
*"Spectral Collapse Without Collapse: SVD Dynamics of LoRA Adapters for Low-Resource Turkic Languages"*.

## Files

| File | Description |
|------|-------------|
| `train_svd.py` | LoRA fine-tuning with per-step SVD monitoring (SE, effective rank, stable rank, entropy, Frobenius norms) |
| `evaluate.py` | Post-training evaluation: cross-lingual PPL, WikiANN NER (3-shot), TUMLU QA (5-shot) |
| `plot_final.py` | Publication figure generation (9 figures) |
| `requirements.txt` | Python dependencies |

## Usage

### Training with SVD monitoring
```bash
python train_svd.py \
  --data_dir ./data/pretrain \
  --ky_file kyrgyz_raw.jsonl \
  --kz_file kazakh_raw.jsonl \
  --uz_file uzbek_final_cyrillic.jsonl \
  --output_dir ./output_ky_baseline_r16_lr2e4_3ep \
  --lora_r 16 --learning_rate 2e-4 --num_train_epochs 3
```

### Evaluation
```bash
python evaluate.py \
  --adapter_path ./output_ky_baseline_r16_lr2e4_3ep/final_adapter \
  --output_dir ./output_ky_baseline_r16_lr2e4_3ep/eval \
  --data_dir ./data/pretrain
```

### Figure generation
```bash
python plot_final.py
```

## SVD Log Format

Each line in `svd_log.jsonl` (logged every 100 steps) contains:
```json
{
  "step": 100,
  "layer": "model.layers.0.self_attn.q_proj",
  "spectral_energy": 0.142,
  "effective_rank": 12.3,
  "stable_rank": 8.7,
  "svd_entropy": 3.42,
  "frobenius_norm_A": 1.23,
  "frobenius_norm_B": 0.45
}
```

## Resources

- **Full repo:** https://github.com/Zarina33/Spectral-Collapse-Turkic
- **Adapters & logs:** https://huggingface.co/Zarinaaa/spectral-collapse-turkic-adapters
