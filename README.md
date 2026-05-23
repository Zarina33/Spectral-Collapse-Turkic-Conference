# Spectral Energy Is Not a Reliable Collapse Diagnostic for Low-Resource Quantized LoRA

**SVD and directional dynamics of LoRA adapters fine-tuned on three low-resource Turkic languages, with a Llama-3-8B architecture-independence replication.**

> Anonymous submission under review. Code, training logs, evaluation reports, and figures only; the paper itself is hosted on the review platform.

---

## TL;DR — four findings

We fine-tune **Gemma-2-9B** (4-bit QLoRA) on Kyrgyz, Kazakh, and Uzbek, with a **Llama-3-8B** replication. SVD-monitor every 100 steps, eval on cross-lingual PPL, WikiANN NER, TUMLU QA, and per-module pairwise cosine of $\Delta W = BA$.

1. **The SE-threshold of [Biderman et al. 2024] silently misses learning-rate-induced collapse.** A four-way decomposition (E5/E5b/E5c plus a matched-$\alpha$ control **C5**) shows four $r{=}64$ runs at sub-threshold SE spanning the full functional range from catastrophic collapse (NER F1$=0$, TUMLU $\sim 23\%$, cross-lingual PPL $>600$) to behaviour matching the $r{=}16$ baseline. C5 further shows the apparent rank-only target-PPL gain of E5c was an artefact of $\alpha/r=2$ implicit LR boost: rank in standard practice is a *proxy* for effective LR, not an independent capacity dimension. A BF16 control (**C3**) rules out 4-bit quantization.

2. **Frobenius norm growth is a useful but language-dependent diagnostic.** Pathological configurations exhibit 15–38× norm growth vs. 1.13× for related-language transfer. But across language categories the diagnostic fails: EN grows $33\times$ without forgetting (KZ PPL $5.52$) while KZ grows $38\times$ with severe forgetting.

3. **The LoRA solution is structurally underdetermined.** A 22-pair sweep of per-module Frobenius cosines between $\Delta W$ matrices shows direct LoRA from base is near-orthogonal across seeds (cosine $0.01$–$0.04$, $0/294$ modules above $0.5$); the collapse-vs-healthy gap ($\cos = 0.010$) sits inside this seed-noise band. **No Lipschitz scalar function of the final adapter alone** can resolve collapse from healthy training; detection has to come from functional or trajectory probes.

4. **Transfer initialisation is the unique intervention that pins direction.** Related-language warm-start (KZ→KY) halves cross-lingual KZ PPL ($23.5 \pm 0.1$ vs. $50.0 \pm 2.4$, $+13\%$ Frobenius growth) and reaches cross-seed cosine **0.785** ($294/294$ modules $>0.5$). A seed-replicated cross-family control (EN→KY) does not reproduce retention. **The Llama-3-8B replication** (E3/E5c/E5 at $n{=}1$; E3, E5c also at $n{=}2$) reproduces all four patterns; the underdetermination inequality holds on both architectures.

---

## Results at a glance

Best-eval-loss checkpoint (HF `load_best_model_at_end=True`); seed 42 unless noted, with $n{=}2$ replication for eight key configs (seeds 42, 123).

### Gemma-2-9B

| ID | Config | KY PPL | KZ PPL | UZ PPL | F1 KY | TypeAcc KY | TUMLU KY | $\|B\|_F$ |
|----|--------|:------:|:------:|:------:|:-----:|:---------:|:--------:|:---------:|
| E1 | KZ baseline (14.1M tok) | 40.75 | **2.73** | 41.27 | 0.219 | 57.7% | 35.7% | 38.5× |
| E1b | KZ small-corpus (1.5M, n=2) | 25.55 | **4.17** | 23.17 | 0.150 | 46.0% | 35.6% | 8.5× |
| E2 | UZ baseline | 116.89 | 64.44 | **4.03** | 0.138 | 54.0% | 35.4% | — |
| E3 | KY baseline (n=2) | **4.78** | 47.58 | 56.85 | 0.157 | 50.4% | 34.7% | 8.36× |
| E4 | KY overfit (10ep) | **4.18** | 87.65 | 95.76 | 0.173 | 56.8% | 33.9% | — |
| E5 | KY r=64, lr=5e-4, 5ep (n=2) | 5.90 | 659.25 | 742.73 | **0.000** | 59.5%‡ | 23.2%‡ | 15.6× |
| E5b | KY r=64, lr=2e-4, 5ep (n=2) | 4.55 | 128.04 | 162.68 | 0.146 | 53.1% | 33.2% | 18.2× |
| E5c | KY r=64, lr=2e-4, 3ep (n=2) | **3.86** | 93.52 | 111.25 | 0.115 | 54.9% | **38.5%** | 8.7× |
| E6 | **KZ→KY transfer (n=2)** | 4.73 | **23.49** | 124.19 | 0.253\* | 59.5% | 33.7% | **1.13×** |
| E6b | EN→KY (cross-family, n=2) | 4.78 | 47.46 | 63.42 | 0.133 | 55.9% | 36.1% | 3.10× |
| E8 | EN control (n=2) | 14.96 | 5.54 | 16.55 | 0.209 | 56.8% | 37.7% | 33.4× |
| C3 | KY BF16 (no quant) | 4.99 | 43.97 | 59.02 | 0.164 | 58.6% | 36.1% | 8.6× |
| C5 | KY r=64, **α=32** (L3 control) | **4.46** | 49.22 | 57.36 | 0.174 | 52.3% | 33.2% | 8.55× |

‡ E5's high TypeAcc despite F1$=0$ → entity knowledge survives the collapse; structured-output generation fails (parse-failure-immune log-likelihood scoring). \*Not robust to reseeding (E6 seed-123 KY F1 $= 0.179$ vs. E3 seed-123 $0.182$); we frame E6 as KZ retention, not KY improvement.

### Llama-3-8B replication (3 configs)

| ID | KY PPL | KZ PPL | UZ PPL | F1 KY | TypeAcc KY | TUMLU KY | max SE |
|----|:------:|:------:|:------:|:-----:|:---------:|:--------:|:------:|
| L3-E3 (n=2) | 4.32 | 30.4 | 54.3 | 0.143 | 57.7% | 32.3% | 0.335 |
| L3-E5c (n=2) | 3.63 | 41.1 | 80.1 | 0.165 | 53.1% | 31.9% | 0.137 |
| L3-E5 | 3.43 | 145.1 | 285.6 | **0.000** | 64.9% | 24.7% | 0.138 |

All four paper-level patterns reproduce qualitatively on L3; underdetermination inequality holds on both architectures (L3 within-config $0.072 \geq$ between-config $0.029$; G2 $0.032 \geq 0.010$).

---

## Directional underdetermination: pairwise cosines

Per-module Frobenius cosine between $\Delta W = BA$ matrices, aggregated over 294 LoRA-equipped projections in Gemma-2-9B (224 in Llama-3-8B). Two regimes:

| Comparison | mean cos | median | modules $>0.5$ |
|------------|:--------:|:------:|:--------------:|
| **Transfer-init, cross-seed (E6: KZ→KY)** | **0.785** | 0.774 | **294/294** |
| Transfer-init, cross-seed (E6b: EN→KY) | 0.343 | 0.324 | 37/294 |
| Direct training, cross-seed (E3, E5, E5b, E5c, E8, E1b) | 0.014–0.041 | 0.009–0.025 | 0/294 |
| **Collapse vs. healthy at r=64 (E5 vs. E5c)** | **0.010** | 0.006 | **0/294** |
| Cross-language at r=16 (E1/E2/E3 mutual) | 0.001–0.002 | <0.001 | 0/294 |
| 4-bit vs. BF16 (E3 vs. C3) | 0.066 | 0.053 | 0/294 |
| C5 ($\alpha{=}32$) vs. E5c ($\alpha{=}128$), same rank | 0.072 | 0.053 | 0/294 |
| **C5 vs. E3 (same $\alpha$, different rank)** | **0.155** | 0.139 | **2/294** |

Key reading: collapse-vs-healthy ($\cos = 0.010$) is **inside** the same-config cross-seed band ($0.01$–$0.04$). Direction cannot separate the two regimes any more than reshuffling the seed does. Only transfer initialisation pins direction; α matters for direction more than rank does (the C5 finding).

---

## Repository structure

```
.
├── scripts/                     # All training, eval, analysis code
│   ├── train_svd.py             # LoRA fine-tuning with SVD callback
│   ├── evaluate.py              # PPL + WikiANN NER (gen + log-likelihood) + TUMLU
│   ├── pairwise_cosine.py       # Memory-efficient per-module cosine (no BA materialisation)
│   ├── pairwise_cosine_lastckpt.py    # Last-checkpoint robustness
│   ├── trajectory_cosine.py     # Trajectory cosines across checkpoints
│   ├── directional_metric.py    # v1 top-r alignment with W0 (negative result)
│   ├── directional_metric_v2.py # v2 cos(BA, W0) + top-256 (negative result)
│   ├── plot_final.py            # Figures 1–9
│   ├── plot_pairwise_cosine.py  # Figure 10 (central directional figure)
│   ├── run_alpha32_control.sh   # Launch C5 (α=32 r=64 control)
│   ├── run_llama3_minimum.sh    # Llama-3 sweep (E3/E5c/E5 seed 42)
│   ├── run_llama3_seed123.sh    # Llama-3 second-seed (E3, E5c)
│   ├── prepare_kz_uz_data.py
│   ├── rebuild_uzbek.py
│   ├── analyze_datasets.py
│   └── wikiann_overlap.py
├── figures/                     # 10 publication figures
├── directional_results/         # Per-experiment cosine JSONs + 22-pair sweep
├── output_*/                    # Per-experiment eval reports, training logs, SVD logs
│                                # (adapter checkpoints excluded; see .gitignore)
└── README.md
```

Adapter checkpoint files (`*.safetensors`, `output_*/checkpoint-*/`, `output_*/final_adapter/`) are excluded from git due to size. Eval reports, SVD logs, configuration dumps, and figures are tracked.

---

## Quick start

```bash
pip install -r requirements.txt   # torch, transformers, peft, bitsandbytes, datasets, ...

# Train a baseline (Gemma-2-9B 4-bit QLoRA, KY corpus)
python scripts/train_svd.py \
    --data_dir ./data/pretrain \
    --ky_file kyrgyz_raw.jsonl \
    --output_dir output_ky_baseline_r16_lr2e4_3ep \
    --lora_r 16 --learning_rate 2e-4 --num_train_epochs 3 --seed 42

# Llama-3-8B baseline
python scripts/train_svd.py \
    --model_name NousResearch/Meta-Llama-3-8B \
    --data_dir ./data/pretrain --ky_file kyrgyz_raw.jsonl \
    --output_dir output_l3_ky_baseline_r16_lr2e4_3ep \
    --lora_r 16 --learning_rate 2e-4 --num_train_epochs 3

# Evaluate (PPL + WikiANN NER + TUMLU)
python scripts/evaluate.py \
    --adapter_path output_ky_baseline_r16_lr2e4_3ep/final_adapter

# 22-pair pairwise-cosine analysis
python scripts/pairwise_cosine.py --output directional_results/pairwise.json

# Figures
python scripts/plot_final.py
python scripts/plot_pairwise_cosine.py
```

---

## Experimental setup

| Component | Details |
|-----------|---------|
| Base models | Gemma-2-9B (4-bit NF4 + bfloat16 compute, plus a BF16 control), Llama-3-8B (4-bit NF4, NousResearch mirror) |
| LoRA | rank 16 or 64, target = $\{q,k,v,o,\text{gate},\text{up},\text{down}\}_{\text{proj}}$; 7 modules per layer; $\alpha = 2r$ throughout except C5 ($\alpha=32$ at $r=64$); dropout 0.0 |
| Optimizer | paged AdamW 32-bit, cosine LR schedule, 5% warmup, effective batch 16 (per-device 1 × grad-accum 16), max grad norm 0.3, gradient checkpointing |
| Data | ~150 MB per language (KZ/KY/UZ), English control; max seq length 256; 10% validation split |
| SVD monitor | Every 100 steps: SE, effective rank, stable rank, SVD entropy, $\|A\|_F$, $\|B\|_F$ on every LoRA module |
| Evaluation | Per-language PPL on 500 held-out samples; WikiANN NER (3-shot, n=100) both generation F1 and log-likelihood span typing; TUMLU QA (5-shot log-likelihood, ~700 questions/language) |
| Hardware | Single consumer GPU (16 GB VRAM) for all 4-bit runs; cloud A100 for the BF16 control |

---

## Subword fragmentation

The Gemma tokenizer splits Turkic languages at ~3.2× the per-word token rate of English, reflecting severe mismatch for agglutinative morphology:

| Language | Tokens / word | vs. English |
|----------|:-------------:|:-----------:|
| English | 1.15 | 1.00× |
| Uzbek | 3.68 | 3.20× |
| Kyrgyz | 3.69 | 3.21× |
| Kazakh | 3.74 | 3.25× |

---

## Data sources

Training corpora (~150 MB each) are not included due to size and licensing. Sources:

| Language | Source | Records | Tokens |
|----------|--------|:-------:|:------:|
| Kazakh | Kazakh Wikipedia + sozkz-corpus | 61,879 | 14.1 M |
| Kyrgyz | Curated local corpus (literature, history, encyclopedic) — **licensed, not redistributable** | 17,360 | 4.4 M |
| Uzbek | uz-books + Uzbek Wikipedia + FineWeb-2 (Cyrillic filter) | 21,242 | 5.4 M |
| English (control) | English Wikipedia | 52,268 | 12.7 M |

Note on Uzbek: corpus is Cyrillic-script; WikiANN-uz is primarily Latin script. The $2\%$ test-entity overlap (vs. $66\%$/$56\%$ for KZ/KY) makes UZ a clean cross-lingual probe in our setup. Results may not generalize to Latin-script Uzbek.

---

## License

Code: MIT. Adapters and figures: CC-BY-4.0. Kazakh, Uzbek, and English corpora retain their upstream licenses. The Kyrgyz corpus is not redistributable.
