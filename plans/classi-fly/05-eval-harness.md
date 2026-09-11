# Evaluation Harness + Baselines - Implementation Leaf

> **For the implementing agent:** You are the implementer for this leaf.
> Implement ALL tasks below using TDD. Do NOT commit - the orchestrator
> handles all git operations after review. Do NOT use read_file on existing
> source files - explore with search_files or terminal cat. After writing a
> file, do NOT read it back to verify - write once and stop. After completing,
> report what you built, what files you touched, and any deviations.

## Meta

- **Parent:** ../master.md
- **Scope:** The offline evaluation harness that scores the reservoir head and
  the judge mode against baselines, using the precision-first protocol.
- **Dependencies:** Contract 2 (`Load`, `Classify`), Contract 3 (readout)
- **Estimated Context:** 85K
- **Concurrency Group:** B

## Goal

Measure whether the reservoir adds signal. Two modes:
- **head mode:** the reservoir alone classifies each embedding.
- **judge mode:** route only when a primary head and the reservoir agree.

Report P (precision of direct routes), C (coverage), E2E, OOD abstain rate,
and the route count. Compare against baselines: nearest-centroid margin, kNN
unanimity, and a char n-gram TF-IDF logistic (the shapes the meept campaign
already measured).

## Context

Protocol (from `../../RESEARCH.md` §1 and §7):
- An incorrect direct route is worse than an abstention.
- E2E = `(gate_correct + 0.868 * abstained) / total` (the chain-credit
  convention). Any mixing of simulated credit uses the deterministic formula,
  never a random draw.
- Route-count disclosure is mandatory for any chain-credit score.
- 5-fold cross-evaluation, fixed fold assignment (seed 42).
- Embeddings come from an external OpenAI-compatible endpoint; pass its base
  URL in. Cache embeddings on disk keyed by text hash so nothing re-embeds.

## Interface Contracts (From Parent)

### What This Leaf Exposes

```
tools/eval/run_eval.py
  python3 tools/eval/run_eval.py --fly <path> --corpus <corpus.json5> \
      --embed-url http://127.0.0.1:8090/v1 --mode {head,judge,all} \
      --folds 5 --seed 42 --out <results.json>
  # prints a metrics table and writes results.json with per-case predictions
```

Corpus format (JSON5 or JSON): a list of
`{"text": "...", "intent": "...", "ood": false, "source": "..."}`.

Baselines live in `tools/eval/baselines.py`:
`centroid_margin`, `knn_unanimity`, `tfidf_logistic`, each with the same
`fit(states, labels)` / `predict(state) -> (label, conf, margin)` shape.

### What This Leaf Consumes

```
Contract 2: Load(path), Reservoir.Classify(embedding) -> Result{Class,...}
Contract 3: readout export (classes, W, bias, threshold).
```

## Tasks

### Task 1: Corpus loading + embedding cache

**Objective:** Load a corpus and embed every text once, cached by hash.

**Files:**
- Create: `tools/eval/corpus.py`
- Create: `tools/eval/embeddings.py`
- Test: `tools/eval/test_corpus.py`

**Step 1: Write failing test** - `load_corpus` reads a 3-item fixture and the
OOD flag; `cached_embed` calls the endpoint once for a repeated text (use a
fake endpoint that counts calls).

**Step 2: Run** `python3 -m pytest tools/eval/test_corpus.py -q`
Expected: FAIL.

**Step 3: Implement** OOD-aware loading and a disk cache keyed by
`sha256(model_id + text)`. Never send corpus text anywhere except the local
embedding endpoint.

**Step 4: Run.** Expected: PASS.

### Task 2: Baselines

**Objective:** Implement the three baseline heads.

**Files:**
- Create: `tools/eval/baselines.py`
- Test: `tools/eval/test_baselines.py`

**Step 1: Write failing test** - on separable synthetic states, each baseline
returns the right label with high confidence; centroid-margin returns a
positive margin for a clean case and a negative margin for an ambiguous one.

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement.** Centroid = mean of class states; margin = top1 cosine
minus top2 cosine. kNN = k=5 cosine, unanimity. TF-IDF logistic = char
n-grams 2-4 with a logistic fit (sklearn or a stdlib equivalent).

**Step 4: Run.** Expected: PASS.

### Task 3: Harness (head + judge modes)

**Objective:** Run the reservoir through the folds and score it.

**Files:**
- Create: `tools/eval/run_eval.py`
- Test: `tools/eval/test_run_eval.py`

**Step 1: Write failing test** - on a tiny fixture `.fly` + fixture corpus,
`run_eval(..., mode="head")` returns a dict with keys
`P, C, E2E, OOD_R, routes, total, predictions`, and `E2E` equals the
deterministic formula recomputed from the counts.

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement** fold splitting (seed 42), per-fold fit of the readout is
NOT done here - the `.fly` already carries the readout. Judge mode takes a
primary head (default: centroid-margin) and routes only on agreement, recording
the route count.

**Step 4: Run.** Expected: PASS.

### Task 4: Report + route-count disclosure

**Objective:** Print the table and write per-case predictions.

**Files:**
- Modify: `tools/eval/run_eval.py`
- Create: `tools/eval/report.md` (template written by the run)

**Step 1:** Emit a table: mode, P, C, E2E, OOD-R, routes/total, and a headline
sentence that names the ROUTE COUNT ("beat the 86.8% floor on N routes").

**Step 2:** Write `results.json` with per-case rows (text hash, gold, pred,
conf, margin, abstained).

**Step 3:** Run it end-to-end on the fixture and confirm the files exist.

**Step 4:** Confirm `python3 -m pytest tools/eval -q` passes.

## Self-Verification Checklist

- [ ] Corpus + embedding cache implemented and tested (no re-embedding)
- [ ] All three baselines implemented and tested
- [ ] Head and judge modes produce the required metric keys
- [ ] E2E uses the deterministic formula; recomputed in the test
- [ ] Route count printed and present in `results.json`
- [ ] No corpus text sent anywhere but the local embedding endpoint
- [ ] `python3 -m pytest tools/eval -q` passes
- [ ] No deviations (or documented below)

**DO NOT COMMIT.** The orchestrator handles all git operations after review.

**Deviations from spec:** none

## Review Checklist (For Review Agent)

- [ ] Every Task implemented and tested
- [ ] Read the SCORING LINES of `run_eval.py`, not just the summary table
  (a sampled draw where the protocol demands deterministic credit has flipped a
  verdict before)
- [ ] Route-count disclosure present
- [ ] OOD cases are scored, not skipped
- [ ] No line-number corruption; no debug artifacts

Output: APPROVED or specific gaps with file + line references.

## Notes

- Do not fake the 0.868 chain constant silently; make it a named constant with
  a comment that it is the measured LLM-chain baseline, and print it.
- If the fixture corpus is too small for 5 folds, fall back to a documented
  leave-one-out and say so in the report.
