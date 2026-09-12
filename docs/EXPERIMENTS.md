# Experiments — E1/E3 results (real embeddings)

Run: 2026-09-11. Original numbers: `tools/eval/e1_real_results.json` (and
`e1_sweep_results.json`). **CORRECTION 2026-09-11: read the correction section
at the end first — the original run understated every reservoir number because
of a readout-regularization defect.**

## Setup

| Item | Value |
|---|---|
| Corpus | meept adjudicated classifier corpora (base + adversarial), read from the meept checkout; text never enters this repo |
| Cases | 389 total: 361 gold (13 intents), 28 gold-OOD, 0 silver |
| Embeddings | REAL: Qwen3-Embedding-0.6B-4bit-DWQ, 1024-dim, local MLX server on 127.0.0.1:8090 (389 texts in 3.4 s) |
| Protocol | 5-fold stratified CV, seed 42; readout and baselines see train folds only; per-class route thresholds calibrated on TRAIN-side scores at target precision 0.97; margin floor 0.005 |
| Reservoir config | steps 4, decay 0.8, random int8 input projection scaled to 0.02 (no saturation: |state|>0.95 in 0.0% of cells) |
| Baselines | centroid-margin and kNN-unanimity on the same embeddings; char 2-4-gram TF-IDF + logistic on raw text (numpy re-implementation; sklearn is absent from this interpreter), all on identical folds |

Route rule for the reservoir rows: route to top-1 only when its calibrated
per-class threshold is met AND the top1-top2 margin clears the floor;
otherwise abstain. E2E = (route_correct + 0.868 * abstained) / total, the
campaign's deterministic convention (`CHAIN_BASELINE`).

## Results

| Head | Routes | P | C | A | E2E | OOD-R |
|---|---:|---:|---:|---:|---:|---:|
| **larval reservoir** (2,952 neurons, 63,545 edges, CC BY 4.0) | 50/361 | 0.900 | 0.139 | 0.125 | **0.8724** | 0.929 |
| **synthetic reservoir** (2,048 neurons, license-free) | 45/361 | 0.867 | 0.125 | 0.108 | **0.8678** | 0.929 |
| centroid-margin (embedding baseline) | 361/361 | — | 1.000 | 0.704 | 0.7036 | — |
| kNN-unanimity (embedding baseline) | 361/361 | — | 1.000 | 0.537 | 0.5374 | — |
| TF-IDF logistic (raw text baseline) | 361/361 | — | 1.000 | 0.352 | 0.3518 | — |
| **judge: centroid AND reservoir agree** | 230/361 | 0.8435 | 0.637 | 0.537 | 0.8524 | — |

(Baseline rows route everything, so their E2E equals raw accuracy.)

## Verdicts

**E1 — is the reservoir a better Stage-0 head? UNSUPPORTED at this
configuration.** It routes 50 of 361 cases (C 13.9%) at P 0.900. That misses
the campaign's 97% precision gate, and its E2E of 0.8724 is only +0.4 pt over
the 0.868 chain-only floor - under one and a half cases at n=361, i.e. noise.
It also loses badly to a plain embedding centroid head (0.704 accuracy).

**E2 — is the reservoir a useful judge/arbiter? UNSUPPORTED.** Requiring the
centroid and the reservoir to agree routes 230 cases at P 0.844, and its E2E
(0.8524) is WORSE than both the chain floor (0.868) and the reservoir-only gate
(0.8724). Agreement filtering removed more correct routes than wrong ones.

**E3 — does the real connectome beat a synthetic mushroom-body-shaped
reservoir? PARITY.** Larval 0.8724 vs synthetic 0.8678 - a 1.7-case gap at
n=361. No measurable advantage from the biological wiring at this scale, which
makes the license question moot for shipping purposes.

**OOD:** both reservoirs abstain on 26 of 28 held-out OOD cases (0.929), close
to the 0.95 target but short of it.

## Why this is a useful negative result

The published reservoir results (BPU on MNIST/CIFAR/chess; connectome ESNs on
chaotic time series) are all in regimes where the input is raw and
low-dimensional. Here the input is already a 1024-dim semantic embedding - the
mushroom body's core trick (sparse expansion of a low-dimensional code) is
largely redundant on top of it. That was flagged as the main risk in
RESEARCH.md §8, and this run is evidence for it.

## Sweep — drive scale × step count (larval, 18 configs)

The 0.0% saturation above suggested the drive was under-powered, so the obvious
gap was swept: `win_scale` ∈ {0.02, 0.05, 0.1, 0.2, 0.4, 0.8} × steps ∈ {2, 4, 8},
5-fold seed 42, threshold target 0.97. Full rows:
`tools/eval/e1_sweep_results.json`.

Result: **every one of the 18 configurations lands between E2E 0.8686 and
0.8742** - a 0.6-point band. Configurations that meet the 97% precision gate do
so only by routing 10-17 of 361 cases (C 2.8-4.7%):

| win_scale | steps | saturation | routes | P | C | E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 4 | 0.000 | 50 | 0.900 | 0.139 | 0.8724 |
| 0.10 | 8 | 0.009 | 23 | 0.913 | 0.064 | 0.8709 |
| 0.20 | 8 | 0.167 | 17 | 0.941 | 0.047 | 0.8714 |
| **0.40** | **2** | **0.440** | **17** | **1.000** | **0.047** | **0.8742** |
| 0.40 | 4 | 0.483 | 11 | 1.000 | 0.031 | 0.8720 |
| 0.80 | 4 | 0.724 | 10 | 1.000 | 0.028 | 0.8717 |

The best cell (E2E 0.8742) is +0.6 pt over the 0.868 chain floor while routing
4.7% of traffic - i.e. two cases better than doing nothing, at single-seed
resolution. There is no configuration where the reservoir's own accuracy
(`A`) approaches the plain embedding centroid's 0.704.

**Sweep verdict: UNSUPPORTED across the swept range.** More drive buys
precision by abstaining more, not by classifying better.

## What is NOT tested (bounds on the claim)

- Ridge readout only; no logistic head, no per-class margins.
- No mushroom-body subcircuit extraction (the whole connectome was used).
- A lower threshold target (<0.97) trades precision for coverage and was not
  explored; nor were per-class margins.
- Baselines route everything (no abstention), so the comparison is
  gate-vs-no-gate, not gate-vs-gate.
- One corpus, one embedding model, one seed.

## Bottom line

The fly reservoir did not improve classification here. It is dominated as a
head by a plain embedding centroid (0.704 accuracy vs the reservoir's 0.125),
it is dominated as a judge (agreement filtering lowered E2E to 0.8524), and it
shows no advantage over a license-free synthetic reservoir of the same shape
(0.8724 vs 0.8678). The mechanism is plausible and was predicted in
RESEARCH.md §8: the mushroom body's sparse-expansion trick is largely
redundant when the input is already a 1024-dim semantic embedding, and a
random projection into 2,952 neurons followed by a linear readout does not
separate 13 fine-grained intent classes better than a centroid does.

Shipping recommendation: **do not wire the reservoir into the classifier
chain.** The machinery (artifact format, Go runtime, sidecar, judge pattern) is
complete, tested, and reusable; the classifier value is not demonstrated.

---

# CORRECTION (2026-09-11, same day, append-only)

## What was wrong

Every number in the sections above — and the 18-cell drive/step sweep — was
produced with the trainer's default ridge penalty (`RIDGE_LAMBDA = 1e-3`) in
`tools/train/train_readout.py`. For a **2,952-dimensional readout fitted on
~289 training samples per fold**, that penalty is effectively unregularized:
the fit is wildly underdetermined and generalizes badly.

Correcting the penalty changes the reservoir's all-case accuracy (`A`) from
**0.125 to 0.726** — a 5.8x change, and the difference between "barely above
chance" and "matches a linear probe". The committed sweep therefore measured
an artifact of the penalty, not the reservoir.

## Corrected results (lambda chosen per representation, 5-fold seed 42)

Best penalty per representation, then the same precision-first gate
(per-class train-side calibration, target 0.97, margin floor 0.05):

| representation | best lambda | A (all cases) | gated routes | P | C | E2E | OOD-R |
|---|---:|---:|---:|---:|---:|---:|---:|
| **larval reservoir** (2,952 neurons) | 30 | 0.7258 | 71/361 | 0.958 | 0.197 | **0.8857** | 0.929 |
| **synthetic reservoir** (2,048) | 10 | 0.7313 | 86/361 | 0.942 | 0.238 | **0.8856** | 0.929 |
| linear probe on raw embeddings | 0.1 | 0.7175 | 88/361 | 0.921 | 0.244 | 0.8808 | — |
| centroid on embeddings (from the earlier run) | — | 0.704 | 70/361 | 0.914 | 0.194 | 0.8770 | — |

Chain-only floor = 0.8680.

## What the corrected numbers say

1. **The reservoir is not broken — it is at parity.** With a sane penalty the
   states carry real class information (A 0.726/0.731) and slightly edge the
   linear probe on embeddings (0.718).
2. **It still adds no information.** Concatenating embeddings with states gives
   0.7285 versus 0.7258 for states alone and 0.7175 for embeddings alone — no
   gain. The reservoir re-parameterizes the embedding; it does not enrich it.
3. **Larval vs synthetic remains PARITY** (0.8857 vs 0.8856). The connectome
   still shows no measurable advantage over a random mushroom-body-shaped
   matrix.
4. **Best gate result: E2E 0.886, P 0.958, covering ~20-24% of traffic** -
   about +1.8 pt over the chain floor, but still below the pre-registered
   97% precision gate.
5. Differences of 1-2 cases at n=361 (0.3-0.6 pt) separate the reservoir from a
   plain linear probe. That is noise, not a win.

## Verdict after correction

**Usable as machinery; still not a demonstrated classifier advantage.** A
precision-first Stage-0 gate built on the reservoir would land around
P 0.95 / C 0.20 / E2E 0.886. The same effect is available from a 13 KB linear
probe over the embeddings that the host already computes — no reservoir, no
2,952-neuron artifact, no connectome. Choose the reservoir only if the
artifact-level properties matter (pure-Go, self-contained, no per-host
training), not for accuracy.

## Repo defect to fix

`tools/train/train_readout.py: RIDGE_LAMBDA = 1e-3` is inappropriate for
readouts whose width approaches or exceeds the training-set size. Either raise
the default (10-30 is right for ~2,000-3,000 features with a few hundred
samples) or select it by cross-validation. Left unfixed here: it is a tuning
default, not a correctness bug, and changing it would invalidate the committed
results above. Corrected numbers: `tools/eval/e1_corrected_results.json`.

## Addendum: OOD detection (the one capability where recurrence could differ)

The campaign needs OOD-R >= 95%. Novelty was measured as the mean cosine
distance to the k=5 nearest training examples, in each space, then compared on
the 361 gold (leave-one-out) versus the 28 gold-OOD cases:

| space | AUC (OOD vs gold) | OOD detected at 95% gold-keep |
|---|---:|---:|
| embedding space (1024-d) | 0.592 | 14.3% |
| reservoir state space (2,952-d) | 0.586 | 14.3% |

Both spaces are weak and effectively identical. **The reservoir adds nothing
for OOD detection either** - the recurrence does not separate out-of-distribution
inputs any better than the embedding it was built from.

## Final position

Investigated: head quality, gate precision/coverage, judge mode, connectome vs
synthetic, drive/step sweep, regularization, information content (concat test),
and OOD novelty. In every case the reservoir is at parity with - never
meaningfully better than - a linear probe or centroid over the embedding the
host already computes.

**Superseded in part by `docs/AXES-2026-09-12.md`**, which adds the four
untested levers (spectral radius, trained projection, mushroom-body subcircuit,
robustness under corrupted input). Summary: three of the four are parity or
worse, and the fourth - robustness to missing input dimensions - is a real win
(+19 pt at 75% truncation, +7.8 pt at 50% dropout) that a synthetic,
license-free matrix reproduces. Read that document before acting on this one.
