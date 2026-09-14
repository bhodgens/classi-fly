# The Stage-0 frontier with OOD as a constraint

Run 2026-09-13. Script: `tools/eval/frontier_ood.py`. Raw rows:
`tools/eval/frontier_ood_results.json`. Fixtures:
`testdata/eval-fixtures/ood_synthetic.json5` (+64 synthetic OOD cases).

## Why this exists

Earlier tuning bought coverage but cost out-of-distribution safety: the OOD
abstain rate fell from 0.929 to 0.857. That regression could not be judged,
because the gold OOD set is 28 cases - one case moves the rate by 3.6 points.
This run fixes the measurement and then asks the question that matters:

    Does any (head, gate) hold precision >= 0.97 AND OOD abstain >= 0.95
    while routing a useful share of traffic?

## Measurement fix, with its own honest limit

The OOD surface is now 92 cases: the 28 gold OOD from the meept adversarial
corpus, plus 64 synthetic cases across the campaign's failure classes (poetry,
gibberish, non-English, personal smalltalk, meta/router questions,
injection-flavoured, hyper-short, unrelated domain). One case is now 1.1 points
instead of 3.6.

**The synthetic 64 are agent-authored, so they are probably easier than real
out-of-distribution traffic** - an obvious poem is obviously not a coding
request. They are reported separately from the gold 28 and never blended. Treat
the gold column as the trustworthy one and the synthetic column as a second
opinion that can be gamed.

## The frontier

Precision-first gate throughout: top-1 at or above its calibrated per-class
threshold, margin over the runner-up above 0.05, else abstain. `P` keeps
routed-but-wrong in the denominator. `E2E` = (route_correct + 0.868 x
abstained) / total.

| head | routes | P | C | E2E | OOD gold | OOD synthetic |
|---|---:|---:|---:|---:|---:|---:|
| centroid, temp 10 | 38 | **0.9737** | 0.105 | **0.8791** | 0.7714 | 0.7969 |
| centroid, temp 20 | 93 | 0.8925 | 0.258 | 0.8743 | 0.6929 | 0.6844 |
| LDA / Mahalanobis | 195 | 0.8256 | 0.540 | 0.8451 | 0.6357 | 0.5312 |
| ridge, penalty 1 | 15 | 0.9333 | 0.042 | 0.8707 | **0.9857** | 0.9844 |
| ridge, penalty 10 | 0 | - | 0 | 0.8680 | 1.0000 | 1.0000 |

**Feasible configurations (both bars met): 0.**

Three readings:

1. **Cosine-to-mean remains the best precision-holding head.** The step up I
   expected to help - a Mahalanobis / LDA head, which whitens the class
   distribution - routes far more traffic (195 cases, coverage 0.54) but at
   precision 0.826 and OOD 0.636. It is worse on both axes that matter. That
   hypothesis is refuted.
2. **You can have precision or OOD, not both, with this machinery.** The
   precision-holding head leaks OOD (0.771). The ridge head that rejects OOD
   well (0.986) routes 15 cases and misses the precision bar.
3. **Raising the precision target does nothing.** 38 routes at target 0.97,
   0.98, 0.99 and 0.995 - identical. `choose_thresholds` searches a 0.025
   quantile grid, which cannot resolve these levels on this data.

## The mechanism that actually fixes OOD - and what it costs

A **supervised OOD probe** (logistic classifier separating in-distribution from
out-of-distribution, trained on the train-fold gold plus the *other* folds' OOD
cases, evaluated on held-out OOD folds so it cannot memorise the eval set):

| configuration | routes | P | C | E2E | OOD gold | OOD synthetic |
|---|---:|---:|---:|---:|---:|---:|
| centroid temp 10, no probe | 38 | 0.9737 | 0.105 | 0.8791 | 0.7714 | 0.7969 |
| centroid temp 10, + probe | 20 | 0.9500 | 0.055 | 0.8725 | **1.0000** | **1.0000** |
| centroid temp 20, + probe | 72 | 0.8750 | 0.199 | 0.8694 | 0.9571 | 1.0000 |
| ridge penalty 1, + probe | 12 | 0.9167 | 0.033 | 0.8696 | 1.0000 | 1.0000 |

So a supervised probe does solve OOD - 0.771 to 1.000 on the gold set. It is not
free: precision falls to 0.950 and coverage halves (38 to 20 routes). The probe
removes correct routes as well as wrong ones.

For contrast, the **unsupervised** version - distance to the nearest training
example - barely moves the number at all: OOD 0.7714 to 0.7929, with no route
gain. That is the interesting negative: the OOD classes sit close to the
in-distribution manifold in embedding space ("write me a sonnet" and "write me
a design doc" are near neighbours), so novelty in embedding space cannot
separate them. OOD rejection here needs labels, not distance.

I also tried recalibrating the class thresholds on the probe-passing subset,
on the theory that the probe changes the population the thresholds should be
fitted to. That did not recover precision either (0.9524 to 0.9500).

## What this means for the classifier

- **No configuration on this corpus reaches precision 0.97 and OOD 0.95
  together.** The two requirements trade against each other through the same
  knob.
- **If OOD safety is the priority**, use a supervised OOD probe and accept
  precision 0.95 / coverage 0.055. That is what meept's existing TF-IDF veto
  does in spirit, and it is the only mechanism measured here that works.
- **If precision is the priority**, the plain cosine-centroid head at temp 10
  is the best available, and its OOD behaviour should be disclosed rather than
  assumed.
- **Two tooling defects to fix before further tuning:** `choose_thresholds`
  cannot resolve precision targets finer than its 0.025 quantile grid (so
  "target 0.97 vs 0.99" is a no-op), and class thresholds calibrated on the
  full train population overstate precision once any novelty gate is in front
  of them.

## Reproduction

```
python3 tools/eval/frontier_ood.py
```

Requires `data/e1_inputs.json` and `data/ood_synthetic_embedded.json`
(the latter is produced by embedding `testdata/eval-fixtures/ood_synthetic.json5`
with the local Qwen3-Embedding server).
