# Readout Training + Calibration - Implementation Leaf

> **For the implementing agent:** You are the implementer for this leaf.
> Implement ALL tasks below using TDD. Do NOT commit - the orchestrator
> handles all git operations after review. Do NOT use read_file on existing
> source files - explore with search_files or terminal cat. After writing a
> file, do NOT read it back to verify - write once and stop. After completing,
> report what you built, what files you touched, and any deviations.

## Meta

- **Parent:** ../master.md
- **Scope:** Offline Python trainer that trains ONLY the linear readout on
  frozen reservoir states, plus per-class threshold calibration.
- **Dependencies:** Contract 1, Contract 3
- **Estimated Context:** 55K
- **Concurrency Group:** A

## Goal

Given a `.fly` or an adjacency export and a training set of
`(embedding, label)` pairs, freeze the reservoir, run each embedding through
the recurrence, and fit a linear readout (ridge or logistic) plus per-class
route thresholds and margins. Emit the readout export Contract 3 defines.

Only the readout is trained. The reservoir stays fixed - that is the entire
point (cheap training, small-data robustness).

## Context

The classifier protocol is precision-first (from `../../RESEARCH.md` §1):
- A wrong direct route is worse than an abstention.
- Route only when the winning lane clears a per-class threshold AND the margin
  over the runner-up is large enough.
- Report coverage (C), precision (P), E2E, OOD abstain rate, and the ROUTE
  COUNT. At n=48 a one-case flip is noise; always disclose route counts.

Training must be CPU-only and deterministic (fixed seed, sorted inputs).

## Interface Contracts (From Parent)

### What This Leaf Exposes

```
tools/train/train_readout.py
  python3 tools/train/train_readout.py --fly <path> --pairs <pairs.jsonl> \
      --out <readout.json> [--lr 1e-3] [--epochs 200] [--calib quantile]
  # pairs.jsonl: one {"embedding":[...],"label":"coding"} per line

tools/train/calibrate.py
  # exposes choose_thresholds(scores, labels, target_precision) -> per-class
```

Readout export (Contract 3):

```json
{
  "classes": ["coding","debugging"],
  "W": [[...N...], ...K...],
  "bias": [...K...],
  "threshold": [...K...],
  "weight_scale": 0.0,
  "readout_scale": 0.0
}
```

### What This Leaf Consumes

```
Contract 1 header field names (classes, neurons, embed_dim, steps).
Contract 3 export shapes.
```

## Tasks

### Task 1: Reservoir state extraction

**Objective:** Reimplement the forward pass in Python for offline use (must
match the Go forward pass for the same artifact).

**Files:**
- Create: `tools/train/states.py`
- Test: `tools/train/test_states.py`

**Step 1: Write failing test** - load a tiny fixture `.fly`, run `states()` on
a fixed embedding, and assert the output length is `neurons` and values are in
`[-1,1]`. Include a golden vector captured from the Go implementation (the
orchestrator will supply it after 01 lands; until then use a self-consistent
golden generated in the test).

**Step 2: Run** `python3 -m pytest tools/train/test_states.py -q`
Expected: FAIL.

**Step 3: Implement** the same recurrence as the Go core: `s = tanh(decay*s +
W*s + W_in*x)`, dequantizing int8 with `weight_scale`. Import numpy if
available; otherwise use stdlib lists.

**Step 4: Run.** Expected: PASS.

### Task 2: Readout training

**Objective:** Fit a linear readout on frozen states with a head-appropriate
learning rate.

**Files:**
- Create: `tools/train/train_readout.py`
- Test: `tools/train/test_train_readout.py`

**Step 1: Write failing test** - on a separable synthetic task (two clusters
of embeddings mapped to two labels), training reaches >90% train accuracy, and
loss is NOT pinned at `ln(K)` (the underdose symptom: a fresh head trained at a
backbone learning rate never learns; use `1e-3` or a closed-form ridge solve).

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement.** Prefer a closed-form ridge solve
`W = (S^T S + lambda I)^-1 S^T Y` (deterministic, no lr tuning) with numpy if
present. If numpy is unavailable, implement logistic regression with a small
learning rate floor of `1e-3`.

**Step 4: Run.** Expected: PASS. Assert the train accuracy is well above chance
and the loss is below `ln(K)`.

### Task 3: Per-class threshold calibration

**Objective:** Choose per-class route thresholds from TRAIN-side score
quantiles targeting a precision floor.

**Files:**
- Create: `tools/train/calibrate.py`
- Test: `tools/train/test_calibrate.py`

**Step 1: Write failing test** - given scores where class A is never confused
and class B often is, `choose_thresholds(..., target_precision=0.97)` returns
a low threshold for A and a high threshold for B.

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement** quantile search per class over the training scores.
Never use a fixed absolute threshold tuned on another model family (it does not
transfer).

**Step 4: Run.** Expected: PASS.

### Task 4: End-to-end export

**Objective:** Produce the readout export and a calibration report.

**Files:**
- Modify: `tools/train/train_readout.py`

**Step 1:** Add `--out` writing the Contract 3 JSON, plus a
`readout_report.json` with train accuracy, per-class thresholds, and the
training pair count.

**Step 2:** Run it against the synthetic fixture pairs and confirm both files
are written and valid JSON.

**Step 3:** Print the report; do not print anything else.

**Step 4:** Confirm `python3 -m pytest tools/train -q` passes.

## Self-Verification Checklist

- [ ] State extraction matches the Go recurrence (golden if supplied)
- [ ] Readout training converges (loss below `ln(K)`; not underdosed)
- [ ] Per-class thresholds calibrated from train quantiles
- [ ] Export matches Contract 3 exactly
- [ ] `python3 -m pytest tools/train -q` passes
- [ ] Deterministic: same inputs -> same `W` (fixed seed / closed form)
- [ ] No deviations (or documented below)

**DO NOT COMMIT.** The orchestrator handles all git operations after review.

**Deviations from spec:** none

## Review Checklist (For Review Agent)

- [ ] Every Task implemented and tested
- [ ] The lr-underdose failure mode is explicitly tested against
- [ ] Threshold calibration uses train-side quantiles, not a constant
- [ ] No line-number corruption; no debug artifacts
- [ ] Export keys match Contract 3 exactly

Output: APPROVED or specific gaps with file + line references.

## Notes

- This leaf deliberately mirrors two lessons from the meept classifier
  campaign (see `../../RESEARCH.md` §1): the head needs its own learning rate
  (~1e-3), and thresholds must come from train-side quantiles.
- Do not train the reservoir itself. If you are tempted, stop - that is a
  different design and not this tree.
