# Watch the Trajectory, Not the Weights: Diagnosing LoRA Collapse in Low-Resource Fine-Tuning

**SVD, Frobenius and directional dynamics of LoRA adapters fine-tuned on three low-resource Turkic languages (Kyrgyz, Kazakh, Uzbek), with a Llama-3-8B replication and a trajectory probe that separates collapsed from healthy runs at a tenth of the training budget.**

> Paper: `arr_paper/main.pdf` (ARR submission, anonymized). `paper/main.pdf` is an earlier full-length draft kept for reference; where the two disagree, `arr_paper/` is current.

---

## TL;DR — four findings

We fine-tune **Gemma-2-9B** (4-bit QLoRA) on Kyrgyz, Kazakh and Uzbek, with a **Llama-3-8B** replication of five configurations. Every LoRA module is SVD-monitored every 100 steps; adapters are evaluated on cross-lingual perplexity, WikiANN NER (generation F1 and parse-failure-immune log-likelihood span typing), TUMLU QA, and per-module pairwise cosine of $\Delta W = BA$.

1. **The spectral-energy threshold (SE > 0.7, Biderman et al. 2024) misses learning-rate-induced collapse.** Four $r{=}64$ runs at sub-threshold SE span the full functional range from catastrophic collapse (NER F1 = 0, TUMLU ~23%, cross-lingual PPL > 600) to behaviour matching the $r{=}16$ baseline. A BF16 control (C3) rules out quantization. An absolute-$\alpha$ control (C5, $\alpha{=}32$ at $r{=}64$) shows the apparent benefit of higher rank is an artefact of the $\alpha{=}2r$ convention: at matched absolute $\alpha$, $r{=}64$ is functionally indistinguishable from $r{=}16$.

2. **Frobenius norm growth separates regimes within a language category but not across.** EN grows 33× without forgetting (KZ PPL 5.5) while KZ grows 38× with severe forgetting.

3. **The final adapter is structurally underdetermined.** Direct LoRA from base is near-orthogonal across seeds (per-module cosine 0.01–0.04, 0/294 modules above 0.5). At $n{=}3$ (seeds 42, 123, 7) the collapse-vs-healthy gap (0.010) sits inside the within-configuration seed spread (0.022–0.042) by 3–8×, which bounds the SNR of any Lipschitz function of the final adapter in this regime. The same inequality holds on Llama-3-8B (0.072 vs. 0.029). **What works is a trajectory measurement**: the mean early growth rate of $\|B\|_F$ over training steps 100–300 separates all 31 logged runs (25 healthy, 6 collapsed, both architectures) with no overlap — AUC 1.000 vs. 0.073 for spectral energy at the same step.

4. **Related-language warm-start preserves source-language knowledge; transfer pins direction to its source, not to a canonical direction.** KZ→KY transfer halves cross-lingual KZ PPL on Gemma (23.5 vs. 50.0), on Llama-3 (19.0 vs. 30.4) and under fully independent reseeding of both stages (26.8). The cross-family control EN→KY does not reproduce retention. The 0.785 cross-seed cosine of the original E6 pair measures two warm-starts from a *shared* source; with the source itself reseeded the cosine falls to 0.023 (Gemma) / 0.060 (Llama-3), inside the direct-training band.

---

## Results at a glance (Gemma-2-9B, seed 42)

Two-seed means with spreads are in the paper's Appendix J; E3, E5 and E5c are at $n{=}3$.

| ID | Config | KY PPL | KZ PPL | UZ PPL | F1 KY | TypeAcc KY | TUMLU KY | $\|B\|_F$ growth |
|----|--------|:------:|:------:|:------:|:-----:|:---------:|:--------:|:---------:|
| E1 | KZ baseline (14.1M tok, n=2) | 40.75 | **2.73** | 41.27 | 0.158 | 70.8% | 32.9% | 38.5× |
| E1b | KZ small-corpus (1.5M, n=2) | 25.55 | **4.17** | 23.17 | 0.150 | 64.0% | 32.0% | 8.5× |
| E2 | UZ baseline (n=2) | 116.89 | 64.44 | **4.03** | 0.138 | 54.0% | 35.4% | — |
| E3 | KY baseline (n=3) | **4.78** | 47.58 | 56.85 | 0.157 | 50.4% | 34.7% | 8.4× |
| E4 | KY overfit (10ep, n=2) | **4.18** | 87.65 | 95.76 | 0.173 | 56.8% | 33.9% | — |
| E5 | KY r=64, lr=5e-4, 5ep (n=3) | 5.90 | 659.25 | 742.73 | **0.000** | 59.5%‡ | 23.2%‡ | 15.6× |
| E5b | KY r=64, lr=2e-4, 5ep (n=2) | 4.55 | 128.04 | 162.68 | 0.146 | 53.1% | 33.2% | 18.2× |
| E5c | KY r=64, lr=2e-4, 3ep (n=3) | **3.86** | 93.52 | 111.25 | 0.115 | 54.9% | **38.5%** | 8.7× |
| E6 | KZ→KY transfer (n=2, + independent-init seed) | 4.73 | **23.49** | 124.19 | 0.253\* | 62.2% | 33.7% | **1.13×** |
| E6b | EN→KY (cross-family, n=2) | 4.78 | 47.46 | 63.42 | 0.133 | 55.9% | 36.1% | 3.10× |
| E8 | EN control (n=2) | 14.96 | 5.54 | 16.55 | 0.209 | 56.8% | 37.7% | 33.4× |
| C3 | KY BF16 (no quantization, n=1) | 4.99 | 43.97 | 59.02 | 0.164 | 58.6% | 36.1% | 8.6× |
| C5 | KY r=64, **α=32** (absolute-α control, n=2) | **4.46** | 49.22 | 57.36 | 0.174 | 52.3% | 33.2% | 8.55× |

‡ E5's high TypeAcc despite F1 = 0: entity knowledge survives the collapse; structured-output generation fails. \* Not robust to reseeding; E6 is framed as KZ retention, not KY improvement.

### Llama-3-8B replication (five configurations)

| ID | Config | KY PPL | KZ PPL | UZ PPL | F1 KY | TypeAcc KY | TUMLU KY |
|----|--------|:------:|:------:|:------:|:-----:|:---------:|:--------:|
| L3-E3 | KY baseline (n=2) | 4.32 | 30.4 | 54.3 | 0.143 | 57.7% | 32.3% |
| L3-E5c | KY r=64, lr=2e-4, 3ep (n=2) | 3.63 | 41.1 | 80.1 | 0.165 | 53.1% | 31.9% |
| L3-E5 | KY r=64, lr=5e-4, 5ep (n=2) | 3.43 | 145.1 | 285.6 | **0.000** | 64.9% | 24.7% |
| L3-C5 | KY r=64, α=32 (n=2) | 4.21 | 29.9 | 51.5 | 0.144 | 56.8% | 31.1% |
| L3-E6 | KZ→KY transfer (n=2) | 4.12 | **19.0** | 81.6 | 0.188 | 54.0% | 28.9% |

Every Gemma pattern reproduces: collapse at high LR with sub-threshold SE, no rank benefit at matched absolute α, KZ retention under transfer, and the underdetermination inequality.

---

## Directional underdetermination: pairwise cosines

Per-module Frobenius cosine between $\Delta W = BA$ matrices, aggregated over 294 projections (Gemma-2-9B) or 224 (Llama-3-8B).

| Comparison | mean cos | modules > 0.5 |
|------------|:--------:|:-------------:|
| Direct training, cross-seed (E1, E1b, E2, E3, E5, E5b, E5c, E8) | 0.014–0.047 | 0/294 |
| **Collapse vs. healthy at r=64 (E5 vs. E5c)** | **0.010** | 0/294 |
| Transfer, cross-seed, *shared* source (E6 from E1-s42, both seeds) | 0.785 | 294/294 |
| **Transfer, cross-seed, *independent* sources (E6-s42 vs. E6-indep-s123)** | **0.023** | 0/294 |
| Transfer vs. its own source (E6 vs. E1 init), Gemma / Llama-3 | 0.797 / 0.804 | all |
| Cross-family transfer, cross-seed (E6b: EN→KY) | 0.343 | 37/294 |
| C5 (α=32) vs. E5c (α=128), same rank | 0.072 | 0/294 |
| C5 vs. E3 (same α, different rank) | 0.155 | 2/294 |
| Llama-3: within-config cross-seed / collapse vs. healthy | 0.072 / 0.029 | — |

Collapse-vs-healthy (0.010) is inside the same-configuration cross-seed band. Direction cannot separate the two regimes better than reshuffling the seed does.

---

## The trajectory probe

$g = (\overline{\|B\|_F}(300) - \overline{\|B\|_F}(100)) / 200$, averaged over all LoRA modules, read from `svd_log.jsonl`. Over the 31 runs with logged trajectories: healthy $g \le 0.0055$, collapsed $g \ge 0.0076$ (candidate threshold 0.00653). All values are in `directional_results/early_frobenius_probe.json`.

```bash
# Prediction for any run directory (exit 3 if step 300 is not logged yet)
python scripts/probe_predict.py output_ky_collapse_r64_lr5e4_5ep
# -> g[100->300] = 0.01195  (1.83x threshold)  -> COLLAPSE

# Held-out test: four configurations absent from the 31, prediction logged at step 300 before eval
bash scripts/run_heldout_probe.sh
```

---

## Repository structure

```
.
├── arr_paper/                   # Current paper (ARR submission, anonymized)
│   ├── main.tex, main.pdf, references.bib, acl.sty, acl_natbib.bst
│   ├── REVISION_RESULTS.md      # Log of every post-submission experiment with numbers
│   └── revision_summary.tex     # Point-by-point response to the previous ARR reviews
├── paper/                       # Earlier full-length draft (reference only)
├── scripts/
│   ├── train_svd.py             # LoRA fine-tuning with the SVD/Frobenius callback
│   ├── evaluate.py              # PPL + WikiANN NER (generation + log-likelihood) + TUMLU
│   ├── pairwise_cosine.py       # Per-module cosine without materialising BA
│   ├── pairwise_cosine_lastckpt.py, trajectory_cosine.py
│   ├── directional_metric.py, directional_metric_v2.py   # W0-alignment probes (negative results)
│   ├── probe_predict.py         # Early ||B||_F growth-rate probe (Section 4.8)
│   ├── run_heldout_probe.sh     # Held-out test of the probe
│   ├── run_alpha32_control.sh   # C5
│   ├── run_llama3_minimum.sh, run_llama3_seed123.sh          # L3-E3 / E5c / E5
│   ├── run_llama3_c5.sh, run_llama3_c5_seed123.sh            # L3-C5
│   ├── run_llama3_e6.sh, run_llama3_e6_seed123.sh            # L3-E6 (two-stage)
│   ├── run_gemma_e6_indep_s123.sh                            # E6 with independently re-seeded source
│   ├── run_revision_chain.sh, run_revision_eval.sh           # Seed replications (E1, E2, L3-E5 at seed 123)
│   ├── plot_final.py, plot_pairwise_cosine.py
│   └── prepare_kz_uz_data.py, rebuild_uzbek.py, analyze_datasets.py, wikiann_overlap.py
├── figures/
├── directional_results/         # Cosine JSONs (22-pair sweep, block-3 seeds, L3-C5) and the probe JSON
├── output_*/                    # Per-run eval_report.json, config_dump.json, training logs, SVD logs
└── README.md
```

Naming: `output_<model>_<lang>_<config>[_seed<N>]/`; `l3_` prefix = Llama-3-8B. Adapter weights (`final_adapter/`, `checkpoint-*/`, `*.safetensors`) are excluded from git for size; `svd_log.jsonl` files are tracked for the runs used in the paper.

---

## Reproducing the paper's tables

```bash
pip install -r requirements.txt      # torch 2.10, transformers 5.2, peft 0.18, bitsandbytes, datasets

# Table 2 / Appendix: train + evaluate one configuration
python scripts/train_svd.py --data_dir ./data/pretrain --ky_file kyrgyz_raw.jsonl \
    --output_dir output_ky_baseline_r16_lr2e4_3ep --lora_r 16 --lora_alpha 32 \
    --learning_rate 2e-4 --num_train_epochs 3 --seed 42
python scripts/evaluate.py --adapter_path output_ky_baseline_r16_lr2e4_3ep/final_adapter

# Table 3: pairwise cosines
python scripts/pairwise_cosine.py --output directional_results/pairwise.json

# Table 4: trajectory probe over all runs with an svd_log.jsonl
python scripts/probe_predict.py output_*/

# Figures
python scripts/plot_final.py && python scripts/plot_pairwise_cosine.py
```

Llama-3-8B runs use `--model_name NousResearch/Meta-Llama-3-8B` (byte-identical mirror of the gated Meta repository). Transfer runs pass `--init_adapter <source>/final_adapter`.

---

## Experimental setup

| Component | Details |
|-----------|---------|
| Base models | Gemma-2-9B (4-bit NF4, bfloat16 compute; one BF16 control), Llama-3-8B (4-bit NF4) |
| LoRA | rank 16 or 64 on $\{q,k,v,o,\text{gate},\text{up},\text{down}\}$ (7 modules/layer); $\alpha = 2r$ except C5 ($\alpha=32$ at $r=64$) |
| Optimizer | paged AdamW 32-bit, cosine schedule, 5% warmup, effective batch 16, max grad norm 0.3, gradient checkpointing |
| Data | ~150 MB per language (KZ/KY/UZ) plus an English control; max seq length 256; 10% validation split |
| SVD monitor | every 100 steps: SE, effective rank, stable rank, SVD entropy, $\|A\|_F$, $\|B\|_F$ per module |
| Evaluation | per-language PPL on 500 held-out samples; WikiANN NER (3-shot, n=100) as generation F1 and log-likelihood span typing; TUMLU (5-shot log-likelihood, ~700 questions/language) |
| Hardware | one 16 GB consumer GPU for all 4-bit runs; rented A100 for the BF16 control and the seed-7 chain |

## Data sources

| Language | Source | Records | Tokens |
|----------|--------|:-------:|:------:|
| Kazakh | Kazakh Wikipedia + sozkz-corpus | 61,879 | 14.1 M |
| Kyrgyz | Curated local corpus (literature, history, encyclopedic) — licensed, not redistributable | 17,360 | 4.4 M |
| Uzbek | uz-books + Uzbek Wikipedia + FineWeb-2 (Cyrillic filter) | 21,242 | 5.4 M |
| English (control) | English Wikipedia | 52,268 | 12.7 M |

The Gemma tokenizer splits all three Turkic languages at ~3.2× the per-word token rate of English. The Uzbek corpus is Cyrillic while WikiANN-uz is mostly Latin (2% test-entity overlap vs. 66%/56% for KZ/KY), which makes UZ a memorization-clean cross-script probe; results may not generalize to Latin-script Uzbek.

## License

Code: MIT. Adapters and figures: CC-BY-4.0. Kazakh, Uzbek and English corpora retain their upstream licenses. The Kyrgyz corpus is not redistributable.
