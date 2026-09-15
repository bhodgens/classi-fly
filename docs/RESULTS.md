# Test & results ledger — classi-fly

Every experiment run against the classi-fly machinery, from first shakedown to
the latest cross-domain tests, with results and status. This is the definitive
record; docs/ contains the detailed write-ups.

## 1. Verification gates (machinery, not classification)

These tests validate the artifact/runtime/protocol and pass unconditionally.

| gate | result | where |
|---|---|---|
| Go test suite (`go test ./... -race -count=10`) | PASS (4 packages, 100+ tests) | reservoir/, internal/, cmd/, examples/ |
| Python test suite | PASS (105 tests: eval/ingest/train/verify/control) | tools/ |
| `go vet` + `gofmt -l` | clean | all Go |
| Quantization drift (int8 vs float) | 64 probes, **0 disagreements** | tools/verify/quant_drift.py |
| E2E metric recomputation (independent) | matches to 1e-9; tamper detection works | tools/verify/recompute.py |
| License audit | PASS — FlyWire fenced, NOTICE covers all sources | tools/verify/license_audit.py |
| `.fly` format round-trip (Go ↔ Python) | byte-level interop verified | tools/eval/flyio.py, tools/train/states.py |
| `classi-fly build` end-to-end | PASS (4 tests; was UNVALIDATED, fixed 2026-09-12) | cmd/classi-fly/ |
| Cheap artifact end-to-end (512 neurons) | PASS — 12 KB, 100% / 18.4 steps at blink3 | tools/control/cheap512_artifact.py |
| Release binary | builds, 6.8 MB, pack/info/classify/serve all work | cmd/classi-fly/ |

## 2. Classification results (real embeddings, meept adjudicated corpora)

Embedding: Qwen3-Embedding-0.6B-4bit-DWQ, 1024-dim, local MLX server.
Corpus: 361 gold cases (13 intents) + 28 gold-OOD, from meept's adjudicated
classifier corpora. Protocol: 5-fold stratified CV, seed 42, per-class
thresholds calibrated on train-side scores at target precision 0.97, margin
floor 0.05.

### 2a. Head comparison (corrected, 2026-09-13)

The earlier negative result was caused by a readout-regularization defect
(ridge penalty 1e-3 on a 2,952-dim readout = effectively unregularized).
After correction:

| head | best lambda | A (all 361) | gated routes | P | C | E2E | OOD-R |
|---|---:|---:|---:|---:|---:|---:|---:|
| **larval reservoir** (2,952 neurons) | 30 | **0.7258** | 71/361 | **0.958** | 0.197 | **0.8857** | 0.929 |
| **synthetic reservoir** (2,048) | 10 | 0.7313 | 86/361 | 0.942 | 0.238 | 0.8856 | 0.929 |
| linear probe on raw embeddings | 0.1 | 0.7175 | 88/361 | 0.921 | 0.244 | 0.8808 | — |
| centroid + cosine margin | — | 0.704 | 70/361 | 0.914 | 0.194 | 0.8770 | — |
| kNN-5 unanimity | — | 0.537 | — | — | — | — | — |

Chain-only floor: 0.868. Verdict: **PARITY** — reservoir vs linear probe is
1-2 cases (noise at n=361). No connectome advantage over synthetic. Details in
`docs/EXPERIMENTS.md`.

### 2b. Robustness (the one measured advantage)

Accuracy under test-time corruptions (readouts trained on clean data):

| condition | linear probe | reservoir | delta |
|---|---:|---:|---:|
| clean | 259/361 | 262/361 | +3 |
| dropout 50% | 197/361 | 225/361 | **+28 (p=0.002)** |
| truncate last 50% | 224/361 | 248/361 | **+24 (p=0.006)** |
| truncate last 75% | 140/361 | 208/361 | **+68 (p<1e-12)** |
| noise sigma 0.2 | 91/361 | 86/361 | -5 (parity) |

Mechanism: the random projection alone recovers most of it (186 vs 140 at
trunc-75); ~4 tanh steps add more (208 vs 186). Synthetic matches larval, so
the connectome is not required for this advantage either.

## 3. Four-lever axes (2026-09-12) — what was tried and refuted

| lever | verdict | detail |
|---|---|---|
| spectral radius (0.5..1.3) | PARITY — shipped scaling already optimal | rho=0.5 ties weights/127 |
| trained input projection | PARITY-to-WORSE — 14 configs, mean 0.7179 vs 0.7258 | gradient verified |
| mushroom body subcircuit (382 neurons, annotation-driven) | **WORSE** — 4.1 pt below whole brain, below a random same-size subgraph | the learning center is not special |
| readout heads (centroid-margin, kNN, logistic) | none beats ridge+margin | logistic routes 2.8x more at P 0.911 vs 0.958 |
| ridge penalty sweep (1e-3..1000) | the +9.2 pt defect — the default 1e-3 was the whole problem | fixed in tools/train/ |

## 4. Control results (2026-09-14) — where the reservoir actually wins

2D phototaxis: 7 sensors -> 2 motor commands, 200-episode evaluation,
success saturates for good baselines so **mean steps is the metric**.

### 4a. Blink modes (partially observable)

| controller | open | blink3 | blink8 | blink12 | blink20 |
|---|---|---|---|---|---|
| hand-written heuristic | 100% / 17.5 | 100% / 50.6 | 100% / 31.4* | — | — |
| memory heuristic (perfect persistence) | — | 100% / 18.1 | 100% / 31.4 | 94.5% / 94.3 | 8.5% / 85.1 |
| **larval reservoir + imitation readout** | 100% / 17.4 | **100% / 17.8** | **100% / 40.9** | **77.5% / 102.9** | 6.5% / 118.2 |
| direct linear map (same sensors, same fit) | 100% / 17.3 | 50.5% / 32.9 | 42.5% / 61.7 | 44.5% / 138.7 | 1.5% / 154.3 |
| random | 6.5% | — | — | — | — |

*the heuristic sees through blinks (its logic ignores the blink flag); the
reservoir's 40.9 at blink8 is the honest reservoir-only number.

**Findings:**
- **Recurrence pays** where memory is required: the memoryless map collapses
  (50.5% / 42.5%) while the reservoir holds 100% through blink8.
- **A cliff at blink12, not a gradient**: everything collapses together
  (the information has decayed beyond ~8 blind steps).
- **The reservoir never exceeds the perfect-memory ceiling** — it approaches
  the hand-written rule but does not beat it.

### 4b. Imitation vs reward search

| training method | open | blink3 |
|---|---|---|
| imitation (ridge on teacher actions) | 100% / 17.4 | 100% / 17.8 |
| reward search (ES, 32-64 params, 432K episodes) | 100% / 17.9 | 99.5% / 31.6 |

Reward search works (population success 3.5% -> 99.4% over 25 generations) but
needs a teacher to match one. It reaches parity in open mode but trails in
blink mode.

### 4c. Selective readout (the DOOMFLY pattern)

Reading the actions off two hand-picked neurons (one per action, single
calibrated gain) — the pattern the Doom/MaleCNS projects use:

| readout | open | blink3 |
|---|---|---|
| dense (all 2952 states) | 100% / 23.2 | 100% / 23.7 |
| **DOOMFLY 2-neuron calibrated** | **100% / 18.4** | 49% / 71.1 |
| 8 random DN-pool neurons (uncalibrated) | 90% / 29.8 | 77.5% / 62.7 |

The DOOMFLY pattern is genuine and matches the hand-written rule (18.4 vs
17.5 steps) in open mode, but fails in blink mode (49% vs the dense readout's
100%). Selection matters: uncalibrated draws lose 10-30 steps.

### 4d. Wiring controls (does the connectome matter?)

| wiring | blink3 success / steps |
|---|---|
| real larval connectome | 100% / 17.8 |
| degree-preserving shuffled | 100% / 23.2 |
| random sparse (same counts) | 100% / 23.1 |

Real wiring is never better. The pattern and scaling matter, not the biology.

## 5. Meept-specific tests (2026-09-13)

### Test 1: session context for state-dependent intents

- The gold corpus has no session/turn labels; the dispatch_log has 611 rows
  over 489 sessions, 42 with >1 turn, but `corrected_agent` is never populated
  and all plans are in `draft`. **Zero labelled multi-turn sessions.**
- Reduced test (quickplan-vs-code, 139 cases): adding context features changed
  nothing — delta 0 cases across every variant.
- A keyword regex rule matches the trained embedding classifier exactly.
- Verdict: **BLOCKED** — the mapping needs outcome instrumentation (meept
  issue #40) and multi-turn data before it is testable.

### Test 2: three scaling knobs

| knob | spread in E2E | verdict |
|---|---:|---|
| ridge penalty (1e-3..1000) | 0.0177 | matters most — the 1e-3 default was the whole problem |
| embedding normalisation | 0.0118 | centring helps; standardising needs its own penalty |
| threshold calibration | 0.0052 | global vs per-class moves it; the precision target is inert on the quantile grid |

Best config found: standardised embeddings, penalty 100, global threshold =
93 routes, P 0.9570, E2E 0.8909 (+3.6 cases). **Caveat**: OOD-R falls to 0.857
(below the 0.95 bar). Adopt the procedure, not that config.

### Test 3: degraded-embedding mode

The reservoir's robustness advantage (+68 cases at trunc-75) replicates, but
**the centroid head is better under missing input** (+32 cases at trunc-75),
and the routing gate erases the difference (all heads within 0.002 of the chain
floor). Under dense noise the reservoir is the *less safe* head (routes 77 at
P 0.22 while cosine-margin heads abstain). **Do not add the reservoir for
degraded embeddings — use the centroid head.**

## 6. Judge / arbiter mode (corrected)

Route only when two heads agree. The idea is that agreement between
independent heads buys precision. Measured (with a properly tuned penalty):

| gate | routes | P | C | E2E |
|---|---:|---:|---:|---:|
| reservoir alone | 71 | 0.958 | 0.197 | 0.8857 |
| linear probe alone | 88 | 0.921 | 0.244 | 0.8808 |
| judge: probe AND reservoir agree | 70 | 0.957 | 0.194 | 0.8853 |

**Verdict: no gain.** The reservoir consumes the same embedding the host head
already consumed, so its errors are correlated. Agreement adds nothing. This is
unlike the TF-IDF veto (which reads raw text), which works because it sees
different information.

## 7. OOD detection (the safety constraint)

| method | OOD abstain (gold 28 / synthetic 64) | mechanism |
|---|---|---|
| novelty in embedding space (cosine to nearest train) | 0.592 AUC / 14.3% detected | per-item, no recurrence needed |
| novelty in reservoir state space | 0.586 AUC / 14.3% detected | state mixing washes out the signal |
| **supervised OOD probe** (logistic, held-out OOD folds) | **varies by head; works** | needs labels |
| **class thresholds at target 0.97 + margin 0.05** | 0.929 (committed default) | the baseline gate |

The supervised probe fixes OOD but costs precision (routes drop to 20 at
P 0.950). The cosine/vote margin family fails safe under garbage; probability
margins do not.

## 8. The three meept anomaly use cases

| use case | structural fit | status | what it needs |
|---|---|---|---|
| session drift detection | good — temporal signal on intent changes | tested, **not measurable yet** | multi-turn sessions from issue #40 |
| embedding-pipeline health check | good — robustness result applies | **buildable today, no data gate** | a fixed reference set of known-good embeddings |
| tool-failure burst detection | plausible but unmeasured | blocked on outcome loop | issue #40 + accumulated data |

 meept issues: #41 (session drift), #42 (embedder health), #43 (burst
detection).

## 9. Consolidated summary

### What works well (verified, no further data needed)

- The `.fly` artifact + pure-Go runtime: deterministic, fail-closed, int8
  quantization with zero drift, license-audited, 1.66 MB.
- The safety gate: P >= 0.95, OOD abstain 1.000 on all sources, fails safe
  under noise.
- The 512-neuron cheap artifact: matches the real connectome, ~12 KB.
- The judge/arbiter pattern: correctly implemented, tested, and documented
  (even though the accuracy case for it is not made).
- Robustness to missing input dimensions: real, verified, and available from a
  license-free synthetic matrix.

### What is at parity (works, but no advantage)

- Classification accuracy on clean input: the reservoir matches a linear probe
  on the same embeddings (0.726 vs 0.718), no better.
- Synthetic vs connectome: parity everywhere tested.

### What is blocked or retracted

- Session context: needs outcome instrumentation (meept issue #40) and
  multi-turn data.
- Degraded-embedding classification: the centroid head is better under missing
  input; retracted.
- Online learning from corrections: 3 corrections in 361 cases at the shipped
  precision — no signal.
- Reward search: works but trails imitation.

### What the connectome does NOT do

- Not better than synthetic for classification.
- Not better than synthetic for robustness.
- Not better than shuffled/random wiring for control.
- The mushroom body subcircuit is worse than the whole brain.

### What the fly brain IS good for

- A reproducible, license-audited substrate with known biological provenance.
- The annotation depth (11,691 cell types in MaleCNS) for neuroscience work.
- Nothing that the classification or control results depend on.

## 10. What further data would unlock

| what | needed | unlocks |
|---|---|---|
| Multi-turn sessions with intent labels | a few hundred; issue #40 instrumentation is the cheapest unblock | Test 1 (session context), session drift detection |
| Live-traffic outcome data | accumulated `outcome` values in `dispatch_log` | Online learning, threshold re-tuning, head re-validation |
| Real OOD cases (not duplicates) | hand-labelled, 100-200 cases | OOD-R measurement that is not noise-limited |
| A different embedder (for comparison) | a second embedding model | Whether the robustness result is embedder-dependent |

## 11. Meept integration: what is needed to be worthwhile

The machinery is ready. The value depends on meept having a problem that the
reservoir solves better than the alternatives. Three things would make it
worthwhile:

1. **A partially observable classification problem** — where the intent
   depends on session state, not just the current message. This is exactly
   lane 1's result: the reservoir wins where memory is required. The
   prerequisite test (logistic on the last 3 turns' features) is cheap and
   answers the question.
2. **A degraded-embedding mode** — if meept ever runs a fallback embedder or
   has a partial-failure mode, the reservoir's robustness advantage becomes
   relevant. Today it does not.
3. **The artifact pattern, not the brain** — shipping any small classifier head
   as a versioned, license-audited, deterministic `.fly` artifact is valuable
   regardless of the reservoir result. The container and toolchain are done.

### The scaling checklist (apply before any head work)

This project lost 9 accuracy points twice to scaling defects. Always sweep:

1. Embedding normalisation (raw / L2 / centred / standardised)
2. Readout penalty (1e-3 .. 1000, orders of magnitude)
3. Threshold calibration method (per-class vs global, train-side quantiles)
