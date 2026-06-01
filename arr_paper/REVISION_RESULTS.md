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

### 3a. E2 — Gemma UZ baseline, cross-seed  [DONE]
Dirs: `output_uz_baseline_r16_lr2e4_3ep{,_seed123}/`.
| seed | KY PPL | KZ PPL | UZ PPL (target) |
|------|-------:|-------:|----------------:|
| 42   | 116.89 | 64.44  | **4.03** |
| 123  | 112.60 | 52.43  | **3.91** |
Target UZ PPL within 0.12 across seeds. Cross-seed cosine **0.047**,
0/294 modules > 0.5 (direct band).

### 3b. L3-E5 — Llama-3 collapse, cross-seed  [DONE]
Dirs: `output_l3_ky_collapse_r64_lr5e4_5ep{,_seed123}/`.
| seed | KY PPL (target) | KZ PPL | UZ PPL |
|------|----------------:|-------:|-------:|
| 42   | **3.43** | 145.07 | 285.62 |
| 123  | **3.49** | 141.95 | 254.99 |
Collapse reproduces at both seeds (KY target tight within 0.06; severe
cross-lingual forgetting both seeds). Cross-seed cosine **0.037**,
0/224 modules > 0.5. Closes the "L3-E5 not seed-replicated" caveat in L7.

### 3c. E1 — Gemma KZ baseline, cross-seed  [DONE]
Dirs: `output_kz_baseline_r16_lr2e4_3ep{,_seed123}/`.
| seed | KY PPL | KZ PPL (target) | UZ PPL |
|------|-------:|----------------:|-------:|
| 42   | 40.75  | **2.73** | 41.27 |
| 123  | 42.98  | **2.69** | 47.59 |
Target KZ PPL within 0.04 across seeds. Cross-seed cosine **0.026**,
0/294 modules > 0.5 (direct band).

### SUMMARY — three new cross-seed pairs (all in direct band)
| Run | target PPL (s42 / s123) | cross-seed cos | >0.5 |
|-----|------------------------:|---------------:|-----:|
| E1 (KZ baseline)    | 2.73 / 2.69   | 0.026 | 0/294 |
| E2 (UZ baseline)    | 4.03 / 3.91   | 0.047 | 0/294 |
| L3-E5 (collapse)    | 3.43 / 3.49   | 0.037 | 0/224 |

Same behaviour-vs-direction dissociation as the existing configs: target
PPL is tight across seeds, yet the per-module direction is near-orthogonal
(all pairs < 0.05, 0 modules above 0.5). Cosine JSON:
`directional_results/pairwise_revision_block3.json`.

---

## 4. L3-E6 seed=123 + architecturally-fair Gemma E6 (Block 4-5)  [DONE]

### 4a. L3-E6-s123 -- Llama-3 transfer second seed  [DONE]
Two-stage pipeline at seed 123 (Stage 1 L3-KZ-s123 then KZ->KY warm-start).
KZ source eval PPL 2.73 (vs s42 2.76). Warm-start target KY eval PPL 5.44
(s42 5.42). Functional retention reproduces across seeds:
| L3-E6 | KY | KZ | UZ | NER F1 KY | TUMLU KY |
|-------|---:|---:|---:|----------:|---------:|
| s42   | 4.12 | 19.04 | 81.60 | 0.188 | 28.9% |
| s123  | 4.17 | 18.49 | 93.08 | 0.139 | 30.8% |
Target KY within 0.05; KZ retention 18.49/19.04 vs L3-E3 direct 30.4.

### 4b. Gemma E6 from independent E1-s123 init (Block 5)  [DONE]
Single experiment that decides the framing. Built to test whether the
original Gemma E6 cross-seed cosine 0.785 reflects (a) genuine
canonicality across seeds, or (b) two stage-2 warm-starts from a
\*shared\* source (E1-s42 was the only KZ baseline that existed before
this revision).

Pre-check (confirms shared-init protocol of the original):
- cos(Gemma-E6-s123, Gemma-E1-s42)  = 0.7929  <- shared-init signature
- cos(Gemma-E6-s123, Gemma-E1-s123) = 0.0209  <- independent-init signature

Result of the architecturally-fair run (E6 warm-started at seed 123
from the NEW E1-s123 init):
- **cos(Gemma-E6-s42-from-E1-s42, Gemma-E6-indep-s123-from-E1-s123) = 0.023, 0/294 > 0.5**
- cos(Gemma-E6-indep-s123, Gemma-E1-s123 source) = 0.797, 294/294 > 0.5  (source-init pinning replicates)

### THE VERDICT (architecture-symmetric)
| Test | Gemma | Llama-3 |
|------|------:|--------:|
| cos(E6, its own source init)                          | **0.797** (294/294) | **0.804** (224/224) |
| cos(E6-s42, E6-s123) with **shared** source           | 0.785 (294/294) -- original headline | -- (not run) |
| cos(E6-s42, E6-s123) with **independent** sources     | **0.023** (0/294) | **0.060** (1/224) |

\textbf{Conclusion:} Transfer pins the adapter to its source initialization
(cos $\approx$ 0.80 on both architectures). The original 0.785 measured two
stage-2 warm-starts from a SHARED source, so it is essentially a measure of
source-pinning, not cross-seed canonicality. Two transfers from
INDEPENDENT sources land in orthogonal directions on both architectures
(0.023 / 0.060), inside the direct-training band. Functional retention,
by contrast, reproduces under independent reseeding on both architectures.

### 4c. Architecture-fair Gemma E6-indep-s123 eval [DONE]
Dir: `output_ky_from_kz_r16_lr2e4_3ep_indep_seed123/`. Full eval results:
| Lang | PPL | NER F1 | TypeAcc | TUMLU |
|------|---:|------:|-------:|------:|
| KY (target) |  4.58 | 0.212 | 0.540 | 35.4% |
| KZ (x-ling) | 26.79 | 0.223 | 0.696 | 31.5% |
| UZ (x-ling) | 130.6 | 0.415 | 0.621 | 27.1% |

**Functional retention reproduces under independent reseeding.** KZ PPL
$50.0 \pm 2.4$ (direct training, both Gemma seeds) $\to 26.79$ (E6-indep-s123,
seed-123 warm-start from seed-123 KZ baseline), a $-47\%$ retention effect.
The original shared-init E6 headline was $-53\%$; the difference falls
inside seed spread. Together with Llama-3 ($-37\%$), this confirms the
KZ retention finding is robust to (a) architecture and (b) full
independence of both training stages -- but the directional cosine 0.785
was a measure of source-init pinning, not cross-seed canonicality.

---

## 5. Weekend Block-3 controls (L5 closure)  [DONE]

### 5a. E2b -- Gemma UZ small-corpus (~1.5M tok)
Dir: `output_uz_tokenmatched_r16_lr2e4_3ep/`. Mirror of E1b for UZ; closes
L5 corpus-imbalance confound on UZ.
| Lang | PPL | NER F1 | TypeAcc | TUMLU |
|------|---:|------:|-------:|------:|
| KY | 120.58 | 0.157 | 0.441 | 33.0% |
| KZ | 64.44  | 0.192 | 0.625 | 33.5% |
| UZ |  7.01  | 0.311 | 0.602 | 29.9% |
Compare to E2 (UZ 5.4M): UZ PPL 6.18 -> 7.01 with 1/4 the data. UZ data
volume matters but less dramatically than KZ (E1->E1b: 2.73 -> 4.17).

### 5b. KZ-4.4M -- true token-matched-to-KY KZ control
Dir: `output_kz_4p4M_r16_lr2e4_3ep/`. KZ subsampled to 4.4M tokens
(matches KY's budget). Closes L5: distinguishes data-volume from
tokenizer-coverage on KZ.
| Lang | PPL | NER F1 | TypeAcc | TUMLU |
|------|---:|------:|-------:|------:|
| KY | 36.79 | 0.114 | 0.513 | 36.6% |
| KZ |  3.61 | 0.159 | 0.723 | 36.9% |
| UZ | 35.56 | 0.342 | 0.738 | 31.8% |
**KZ PPL 3.61 at KY's exact token budget still beats KY baseline 4.78.**
Confirms KZ has *both* more data *and* a better tokenizer/pretraining
coverage. Both factors contribute; neither alone explains the gap.

### 5c. E4-s123 -- Gemma KY overfit second seed (10 epochs)
Dir: `output_ky_overfit_r16_lr2e4_10ep_seed123/`. Last single-seed Gemma
configuration brought to n=2.
| Lang | PPL | NER F1 | TypeAcc | TUMLU |
|------|---:|------:|-------:|------:|
| KY |  4.24 | 0.128 | 0.568 | 34.7% |
| KZ | 78.41 | 0.204 | 0.714 | 33.1% |
| UZ | 80.54 | 0.322 | 0.670 | 29.8% |
Overfit regime (10 ep on 4.4M KY): heavy cross-lingual damage, KY itself
no better than the 3-epoch baseline.

---

## PAPER REFINEMENT (applied in this commit)

- Abstract finding (4): transfer = functional retention (robust), source-init pinning (architecture-independent), independent-init cross-seed = 0.02-0.06 (underdetermination extends).
- Contribution C4: same refinement.
- Figure 1 caption: noted shared-init protocol + independent-init counterpoint.
- tab:pairwise_cos: added independent-init E6 row (0.023); caption rewritten.
- Section 4.6 D2 (Three regularities): transfer pins to source, not to canonical.
- Section 5.2 (Underdetermination): explicit extension to transfer under reseeding.
- Section 4 Llama-3 results paragraph: aligned with new framing.
- Conclusion finding (4): functional retention as the strong claim; directional as the refined claim.
- Section S (Llama-3 replication) E6 paragraph: rewritten architecture-symmetric.
- tab:llama3_cos: rebuilt transfer rows with shared-vs-independent contrast.
- tab:lastckpt / tab:trajectory: footnote on E6 cross-seed = shared-init protocol.
- Mechanistic Context: shared-source mode-connectivity refinement.
- PiSSA discussion: clarified PiSSA sits in shared-init regime by construction.

aclpubcheck: All Clear, 23 pages.

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
