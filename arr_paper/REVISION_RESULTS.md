# ARR Revision Results Log

Running log of all revision-iteration experiments (post initial ARR/EMNLP
submission). Each entry has exact numbers and the paper location they belong
to, so they can be pasted straight into `main.tex` without re-deriving.

All runs: RTX 5080 16 GB, 4-bit NF4 + QLoRA, seed 42 unless noted.
Conventions: PPL = held-out perplexity; "x-ling" = cross-lingual (adapter
trained on one language, evaluated on another); TypeAcc = log-likelihood NER
span-typing accuracy (parse-failure immune); cosines = per-module Frobenius
cosine on Delta-W = BA.

---

## STATUS

| Experiment | Train | Eval | Cosine | In paper |
|------------|:-----:|:----:|:------:|:--------:|
| L3-C5 (Llama-3 absolute-alpha) | done | done | done | done (Sec S) |
| L3-E6 (Llama-3 KZ->KY transfer) | done | done | done | done (Sec S) |
| E2-s123 (Gemma UZ baseline)    | done | auto | auto | TODO |
| L3-E5-s123 (Llama-3 collapse)  | done | auto | auto | TODO |
| E1-s123 (Gemma KZ baseline)    | running | auto | auto | TODO |

"auto" = handled by `scripts/run_revision_eval.sh` once training chain exits.

---

## 1. L3-C5 — Llama-3-8B absolute-alpha control  [DONE, in paper]

Setup: Kyrgyz, r=64, **absolute alpha=32** (breaks alpha=2r), lr=2e-4, 3 ep.
Dir: `output_l3_ky_r64_lr2e4_3ep_alpha32/`. Train ~4.6 h.
Closes: the C5 finding was Gemma-only; now architecture-independent.

### Functional (eval_report.json)
| Lang | PPL | NER F1 (gen) | NER TypeAcc | TUMLU |
|------|----:|----:|----:|----:|
| KY (target) | 4.21 | 0.144 | 0.568 | 31.1% |
| KZ (x-ling) | 29.88 | 0.221 | 0.679 | 25.3% |
| UZ (x-ling) | 51.46 | 0.491 | 0.641 | 25.5% |

final-step max SE = **0.205** (well below 0.7 threshold).

### Key comparison (the C5 story)
| Run | r | alpha | KY PPL | KZ PPL | UZ PPL |
|-----|--:|------:|-------:|-------:|-------:|
| L3-E3  | 16 | 32  | 4.32 | 30.4 | 54.3 |
| L3-E5c | 64 | 128 | 3.63 | 41.1 | 80.1 |
| **L3-C5** | 64 | **32** | **4.21** | **29.9** | **51.5** |

=> At matched absolute alpha=32, r=64 (C5) is functionally indistinguishable
from the r=16 baseline (E3). E5c's apparent target-PPL gain (3.63) costs
cross-lingual KZ PPL (41 vs 30). The cost tracks absolute alpha, not rank.

### Directional (pairwise cosine, 224 modules)
| Pair | mean | median | max | >0.5 |
|------|-----:|-------:|----:|-----:|
| L3-C5 vs L3-E3 (same alpha=32, diff rank) | 0.343 | 0.339 | 0.857 | 23/224 |
| L3-E5c vs L3-C5 (same rank, 4x alpha)     | 0.162 | 0.140 | 0.686 | 1/224 |
| L3-C5 vs L3-E5 (collapse)                 | 0.021 | 0.017 | 0.186 | 0/224 |

Gemma counterpart: C5-vs-E3 0.155, E5c-vs-C5 0.072 (Llama-3 ~2x, same pattern).

Paper: Table `tab:llama3` (C5 column), Table `tab:llama3_cos` (C5 rows),
Section S finding (v).

---

## 2. L3-E6 — Llama-3-8B KZ->KY transfer  [DONE, in paper]

Two-stage. Stage 1: L3-KZ baseline (Kazakh 14.1M tok, r=16/a=32/3ep), KZ eval
PPL 2.76, ~22 h. Stage 2: KZ->KY warm-start (Kyrgyz 4.4M tok), ~4.5 h.
Dirs: `output_l3_kz_baseline_r16_lr2e4_3ep/`, `output_l3_ky_from_kz_r16_lr2e4_3ep/`.
Closes: transfer-pinning bimodality was Gemma-only; now architecture-independent.

### Functional — KZ retention test
| Lang | L3-E3 (direct) PPL | L3-E6 (KZ->KY) PPL | change |
|------|----:|----:|----:|
| KY (target) | 4.32 | 4.12 | comparable |
| KZ (x-ling) | 30.38 | **19.04** | **-37% retention** |
| UZ (x-ling) | 54.27 | 81.60 | +50% (transfer was from KZ, not UZ) |

L3-E6 full eval: KY F1 0.188 / TypeAcc 0.540 / TUMLU 28.9%;
KZ F1 0.203 / TypeAcc 0.679 / TUMLU 24.1%; UZ F1 0.515 / TypeAcc 0.709 / TUMLU 25.0%.

Gemma E6: KZ PPL 50 -> 23.5 (-53%). Same direction, Llama-3 effect milder.

### Directional (pairwise cosine, 224 modules) — the bimodality
| Pair | mean | median | max | >0.5 |
|------|-----:|-------:|----:|-----:|
| **L3-E6 vs L3-KZ-init** (transfer vs source) | **0.804** | 0.807 | 0.868 | **224/224** |
| L3-E6 vs L3-E3 (transfer vs direct KY)  | 0.020 | 0.014 | 0.168 | 0/224 |
| L3-E6 vs L3-E5c (transfer vs direct r64) | 0.020 | 0.015 | 0.187 | 0/224 |

Gemma E6 cross-seed: 0.785 with 294/294 > 0.5. Llama-3 transfer-pinned at
0.804 with all 224/224 modules > 0.5; direct training stays at ~0.02.
Bimodality holds on both architectures.

Paper: Table `tab:llama3_cos` (E6 rows), Section S "Transfer-pinning
bimodality replicates on Llama-3" paragraph.

---

## 3. Seed-replication chain (Block 3) — close L1 single-seed claims

Script: `scripts/run_revision_chain.sh` (train) + `scripts/run_revision_eval.sh`
(eval + cosine, auto-fires when training chain exits).

### 3a. E2-s123 — Gemma UZ baseline, seed 123  [TRAIN DONE]
Dir: `output_uz_baseline_r16_lr2e4_3ep_seed123/`.
Eval: TODO (auto). Cross-seed cosine E2_s42 vs E2_s123: TODO (auto).
Prediction: cosine in direct band (0.01-0.04), functional metrics ~ seed 42.

### 3b. L3-E5-s123 — Llama-3 collapse, seed 123  [TRAIN DONE]
Dir: `output_l3_ky_collapse_r64_lr5e4_5ep_seed123/`.
Eval: TODO (auto). Cross-seed cosine L3_E5_s42 vs L3_E5_s123: TODO (auto).
Prediction: confirms collapse reproduces (NER F1=0, sub-threshold SE) and
cross-seed cosine ~0.07 (Llama-3 direct band). Closes the "L3-E5 not
seed-replicated" caveat currently in L7 / Section S.

### 3c. E1-s123 — Gemma KZ baseline, seed 123  [TRAINING ~40%]
Dir: `output_kz_baseline_r16_lr2e4_3ep_seed123/`.
KZ 14.1M tok = 10443 steps, ~28 h (slowest run). Eval: TODO (auto).
Cross-seed cosine E1_s42 vs E1_s123: TODO (auto).

### TO FILL when eval-chain completes
| Run | KY PPL | KZ PPL | UZ PPL | NER F1 (target) | TUMLU | cross-seed cos |
|-----|-------:|-------:|-------:|----------------:|------:|---------------:|
| E2-s123    | _ | _ | _ | _ | _ | _ |
| L3-E5-s123 | _ | _ | _ | _ | _ | _ |
| E1-s123    | _ | _ | _ | _ | _ | _ |

---

## PAPER-UPDATE CHECKLIST (after Block 3 eval completes)

- [ ] **L1 limitation**: "n=2 for 8 configurations" -> "n=2 for 11
      configurations" (add E1, E2 on Gemma; L3-E5 on Llama-3).
- [ ] **Section J** (Seed-Variance Replication): add E1, E2 cross-seed rows
      to the seed-variance table.
- [ ] **Table `tab:llama3_cos`**: add L3-E5 cross-seed row (currently only
      E3, E5c cross-seed are listed); fill the "(Gemma only)" gap.
- [ ] **L7 / Section S**: remove "We do not seed-replicate L3-E5" caveat.
- [ ] Re-run aclpubcheck -> All Clear -> push.

## FIXES ALREADY APPLIED
- Table `tab:llama3` L3-C5 max SE corrected 0.167 -> **0.205**
  (paper uses *final-step* max SE across layers, not global max).
  Verified against L3-E3 0.335 / L3-E5c 0.137 / L3-E5 0.138.
- Note: the `||B||_F growth` row was dropped from `tab:llama3` when the C5
  column was added (width). Re-add if space allows; L3-C5 value still TBD
  (raw per-layer ratio looked anomalously high, needs the paper's exact
  definition before reporting).
