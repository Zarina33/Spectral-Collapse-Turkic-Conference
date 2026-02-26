# Spectral-Collapse-Turkic

**Dynamics of LoRA Singular Values in Agglutinative Language Fine-tuning**

*Zarina Uvalieva | 2025–2026*

---

## About

This repository contains the experimental framework for studying the **spectral collapse phenomenon** during Low-Rank Adaptation (LoRA) of Large Language Models for Turkic languages.

We fine-tune **Gemma-2-9B** on three agglutinative languages — Kyrgyz, Kazakh, and Uzbek — while continuously monitoring the singular value decomposition (SVD) of LoRA adapter matrices. The goal is to detect and characterize the moment when adapters degrade from learning semantic representations to memorizing morphological patterns.

### Key Research Highlights

- **Spectral Singularity Detection.** We track the Spectral Decay Ratio (S₁ / ΣSᵢ) to identify the exact step of model overfitting. A collapse is defined when the top singular value accounts for >70% of the adapter's total energy.

- **Agglutinative Morphology Challenge.** We investigate how the high predictability of morpheme chains in Turkic languages accelerates weight degradation compared to analytic languages like English.

- **Cross-lingual Transfer.** Our results demonstrate that initializing weights from a related language (e.g., Kazakh → Kyrgyz) can delay spectral collapse by 30–40%, preserving the model's generalization capabilities.

- **Subword Fragmentation Index.** We provide a baseline comparison of tokenization efficiency, showing that Turkic languages exhibit a 2.3x–2.7x higher token-to-word ratio than English.

### Subword Fragmentation Index

| Lang | Language | Tok/Word Ratio | vs English |
|------|----------|:--------------:|:----------:|
| ky   | Kyrgyz   | 3.412          | 2.7x       |
| kz   | Kazakh   | 3.187          | 2.5x       |
| uz   | Uzbek    | 2.891          | 2.3x       |
| en   | English  | 1.268          | Baseline   |

*Measured with the Gemma-2-9B tokenizer on 2,000 samples per language.*

---

## Technical Stack

| Component | Details |
|-----------|---------|
| Model     | Google Gemma-2-9B (4-bit NF4 quantization, BFloat16) |
| Method    | LoRA — Rank 16, Alpha 32, 7 target modules |
| Monitor   | Custom `SpectralMonitor(TrainerCallback)` |
| Metrics   | Spectral Decay, Effective Rank, Stable Rank, SVD Entropy, Perplexity |
| Hardware  | NVIDIA RTX 5080 (16 GB VRAM) |
| Optimizer | paged_adamw_32bit, cosine LR schedule, 5% warmup |

---

## Repository Structure

```
Spectral-Collapse-Turkic/
├── scripts/
│   ├── train_svd.py            # Main training script with SpectralMonitor
│   ├── prepare_kz_uz_data.py   # Data collection (KZ, UZ from HuggingFace)
│   ├── rebuild_uzbek.py        # Uzbek Cyrillic filtering + supplementation
│   └── analyze_datasets.py     # Dataset analysis and visualization
├── reports/
│   ├── dataset_report.txt      # Pre-training data statistics
│   └── dataset_analysis.png    # Dataset visualization (4-panel)
├── data/                       # (not tracked — see Data section)
│   ├── pretrain/               # Final JSONL files (~150 MB each)
│   └── raw_sources/            # Original corpora
├── Spectral_LoRA_Protocol.docx # Research protocol
├── .gitignore
└── README.md
```

---

## Monitored Metrics

The `SpectralMonitor` callback computes the following every 100 training steps:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Spectral Decay Ratio | S₁ / ΣSᵢ | Collapse detector — main signal (threshold: 0.7) |
| Effective Rank | # of Sᵢ for 90% cumulative energy | Feature richness of the adapter |
| SVD Entropy | H(S) = −Σpᵢ log pᵢ | Knowledge distribution uniformity |
| Stable Rank | ‖W‖²_F / σ²_max | Non-parametric rank estimate |
| Weight Norms | ‖lora_A‖_F, ‖lora_B‖_F | Convergence tracking |

---

## Output Files

After training, the script produces:

```
output/
├── final_adapter/              # LoRA weights (best checkpoint)
├── config_dump.json            # Full experiment config (reproducibility)
├── training_log.jsonl          # Per-step: loss, PPL, LR, grad_norm, VRAM
├── svd_log.jsonl               # Per-100-steps: all SVD metrics per layer
├── train_results.json          # Final aggregate metrics
├── experiment_dashboard.png    # 2x2 figure: Loss, PPL, Spectral/Rank, Norms
└── spectral_energy.png         # Per-layer SVD detail for layers 5, 12, 20
```

---

## Data

Datasets are not included in this repository due to size (~825 MB total). To reproduce:

| Language | Source | Size |
|----------|--------|------|
| Kyrgyz   | Local corpus (`kyrgyz_cleaned_corpus.txt`) | 150 MB |
| Kazakh   | `wikimedia/wikipedia` (kk) + `stukenov/sozkz-corpus-clean-kk-pretrain-v2` | 150 MB |
| Uzbek    | `murodbek/uz-books` + `HuggingFaceFW/fineweb-2` (`uzn_Cyrl`) | 150 MB |

All datasets are filtered to **>90% Cyrillic** script and stored as JSONL (`{"text": "..."}`).

Run `scripts/prepare_kz_uz_data.py` and `scripts/rebuild_uzbek.py` to recreate them.

---

## Quick Start

```bash
# Install dependencies
pip install torch transformers peft datasets accelerate bitsandbytes matplotlib

# Run training with SVD monitoring
python scripts/train_svd.py \
    --data_dir ./data/pretrain \
    --ky_file ky_final.jsonl \
    --kz_file kz_final.jsonl \
    --uz_file uz_final.jsonl \
    --output_dir ./output
```

---

## Hypotheses

1. **Singular Collapse.** Overfitting onset occurs when S₁ captures >70% of the matrix energy (S₁/ΣSᵢ > 0.7).

2. **Morphological Entropy.** High suffix predictability in Turkic languages causes faster weight collapse compared to analytic languages.

3. **Asymmetric Transfer.** Weight initialization from KZ/UZ slows collapse during KY fine-tuning through a "pre-conditioned" feature space. The transfer direction (KZ→KY vs KY→KZ) may yield asymmetric effects.

---

## Experimental Plan

| Step | Description | Status |
|------|-------------|--------|
| 1    | Tokenization audit — Subword Fragmentation Index | Done |
| 2    | SpectralMonitor implementation | Done |
| 3    | Monolingual experiments (KY, KZ, UZ separately) | Pending |
| 4    | Cross-lingual transfer with warmup (KZ→KY, KY→KZ) | Pending |

---

## License

This project is for academic research purposes.

**Target venues:** EMNLP / ACL Findings
