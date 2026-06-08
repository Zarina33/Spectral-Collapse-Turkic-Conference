# Revision Safety: How to Handle ARR Reviews

Notes on what to do if ARR reviewers come back asking to keep things from
the originally-submitted version. **Nothing is lost** — every prior state
is in `git`, recoverable in one command.

---

## State of play

1. **ARR has a frozen copy** of the originally-submitted PDF. Reviewers
   read THAT, not anything on GitHub. Local/GitHub changes do not touch
   the ARR submission.

2. **Git preserves every version** of `main.tex` and `main.pdf`. Each
   commit is a full snapshot. Any past version can be restored exactly.

3. **The current GitHub state** is the revision-ready continuation
   (L3-C5, L3-E6, Gemma E6-indep, Block-3 seeds, L5 controls). It is
   READY to upload at the revision step of the ARR cycle, but does not
   touch the frozen submission.

---

## Safety tag (already pushed)

```
revision-ready-2026-06  →  commit 565fec5  →  arr_paper/main.{tex,pdf}
                          with full revision continuation
```

Pushed to origin. Always retrievable via `git show revision-ready-2026-06`.

## Key historical commits

```
a179eb5   ARR submission build (anonymized + 8-page cut)
988ca2d   final pre-submission audit fixes  ← closest to ARR-submitted PDF
ae33764   anonymous repo URL update
22ffc9f   aclpubcheck All Clear (early revision state)
194cfc4   integrate L3-C5 and L3-E6 results
f40e906   Block-3 seed replications integrated
7f9e37c   transfer-pinning claim refined (architecture-symmetric)
f9e38da   Gemma E6-indep functional retention filled
565fec5   tables 20+21 merged for tight stacking (current head)
```

---

## If reviewers ask to "keep it as was"

| Scenario | Action |
|----------|--------|
| Reviewer liked the 0.785 headline, wants it kept | Defence: "we discovered it was a shared-init artifact; the architecturally-fair test (0.023 Gemma, 0.060 Llama-3) is a STRONGER result, not weaker" |
| Reviewer wants a specific old phrase / claim restored | `git show <commit>:arr_paper/main.tex \| grep -A5 "phrase"` then paste into current file |
| Reviewer wants the entire old paper back | `git checkout <commit> -- arr_paper/main.tex arr_paper/main.pdf` |
| Reviewer asks to add something without removing anything | Just add — most revision work is additive (new experiments, new tables) |

---

## One-line commands

```bash
# Show a past version's PDF
git show 988ca2d:arr_paper/main.pdf > /tmp/old_version.pdf

# Restore a single file to a past state
git checkout 988ca2d -- arr_paper/main.tex arr_paper/main.pdf

# Diff two versions of the source
git diff 988ca2d..revision-ready-2026-06 -- arr_paper/main.tex

# See full history of one file
git log --oneline -- arr_paper/main.tex

# Recover one section from old PDF (Section 4.5 example)
pdftotext -layout main.pdf - | grep -A 50 "Cross-Lingual Transfer"
```

---

## Why the refinement is stronger than the original

Original C4 claim:
> "Transfer initialization reaches cross-seed cosine 0.785, the unique
> intervention that pins direction across seeds."

Refined C4 claim:
> "Transfer initialization pins the adapter to its source (cos ≈ 0.80 on
> both Gemma and Llama-3). Two transfers from INDEPENDENT sources land
> at cos ≈ 0.02–0.06, inside the direct-training band. The 0.785 figure
> was measured with two stage-2 warm-starts from a SHARED source — it
> is a measure of source-init pinning, not cross-seed canonicality."

Why reviewers prefer the refined version:
- **More rigorous protocol** (architecturally-fair: both stages reseeded)
- **Architecture-independent** (holds on Gemma AND Llama-3)
- **Internally consistent**: extends the underdetermination argument
  from direct training to transfer
- **Self-corrective**: shows we found the nuance ourselves rather than
  having a reviewer spot it
- **Functionally**: KZ retention (–47 % Gemma, –37 % Llama-3) reproduces
  under independent reseeding — the practical claim is unchanged

If a reviewer praises the original framing, the answer is: the refined
framing reaches the same practical conclusion (transfer preserves source
knowledge) with a stronger evidential standard.

---

## Bottom line

- Nothing is lost.
- Anything can be reverted.
- The refinement is intellectually stronger and easier to defend.
- Both states (original + revision-ready) coexist on GitHub.
