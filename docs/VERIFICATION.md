# Adversarial Verification Report (leaf 08)

> Independent recomputation of the eval harness's headline numbers, float-vs-int8
> quantization stress, license-provenance audit, and the empirical findings from
> a hostile review of everything currently landed in this repo.
>
> Rule: none of `tools/verify/` imports the harness's own aggregation code. Every
> number below is recomputed from per-case data or re-derived from first
> principles. Where this report says "verified", the exact command is given.

Date: 2026-09-11
Scope: `reservoir/` (Go), `internal/flyformat/` (Go), `cmd/classi-fly/` (Go),
`tools/eval/`, `tools/train/`, `tools/ingest/`, shipped binary `bin/classi-fly`,
fixture corpus `testdata/eval-fixtures/fixture_corpus.json5`, fixture artifact
`tools/eval/eval_fixture.fly`.

---

## 1. Findings (adversarial review)

### Finding 1 — Ridge readout orientation bug (CONFIRMED, severity: high)

`tools/train/train_readout.py` fits the readout as
`W = (SᵀS + λI)⁻¹ Sᵀ Y` over `Xb = [S | 1]` and returns `sol[:-1, :]` as `W`,
which has shape **[N][K]** — one row per NEURON, one column per class
(Contract 3's declared layout). But `score_readout` and `_class_scores` compute
logits with `zip(W, bias)`:

```python
logits = [sum(w * v for w, v in zip(row, s)) + b for row, b in zip(W, bias)]
```

which treats each ROW of `W` as one CLASS's weight vector over the state. The
two interpretations agree only when the fitted `W` is symmetric; in general,
including at N == K, they do not. **Every probability this trainer reports for
real (N > K) training sets is wrong, and `choose_thresholds` calibrates against
the wrong probabilities**, so any `.fly` trained via `tools/train` has an
untrustworthy readout and untrustworthy per-class thresholds.

The fit itself is sound: feeding the returned `W`/`bias` through the correct
orientation (`logits = h @ W + bias`) reproduces the closed form to
RMSE < 0.01. The bug is purely in the scoring/consumption orientation.

Empirical reproduction (also pinned by `tools/verify/test_ridge_orientation.py`
and printable via `tools/verify/ridge_orientation_repro.py`):

```
python3 tools/verify/ridge_orientation_repro.py
```

Output (synthetic frozen states, N=6 neurons, K=3 classes, 40 well-separated
samples, seed 99):

| metric | production (`zip(W, bias)`) | correct orientation (`h @ W + bias`) |
|---|---|---|
| train accuracy | **0.325** (chance level) | **1.000** |
| max probability divergence | 0.4052 | — |
| closed-form RMSE under correct orientation | 0.00953 (fit is fine) | — |

Sample row: gold one-hot `[1.0, 0.0, 0.0]` → production probabilities
`[0.390, 0.155, 0.455]` (argmax wrong); correct orientation
`[0.575, 0.213, 0.212]` (argmax right).

Second reproduction at **N == K = 4** (seed 5, 24 samples): production train
accuracy 0.75 vs 1.0 under the correct orientation. The orchestrator's hand-off
claimed "correct only when N == K"; **that exception is itself false** — the
hand-off's parenthetical was empirically wrong, though the bug it flagged is
real and worse than described. The confusion likely came from small
hand-symmetric fixtures.

Disposition: **FIXED 2026-09-11** (orchestrator). `train_readout._logits()`
now consumes W[N][K] neuron-major; `score_readout` and `_class_scores` both
route through it. The N≠K pin passes (production acc now equals the correct
orientation) and the N==K pin was flipped to assert parity. Full suite:
`python3 -m pytest tools/train tools/verify -q` -> 37 passed.

### Finding 2 — `classi-fly build` convenience path fails on scaffold pairs (REPRODUCED)

Command:

```
./bin/classi-fly build --synthetic --classes /tmp/classes.json --out /tmp/out.fly
```

(scaffold pairs generated from `--classes`, default seed 7, 64 neurons, 3 classes)

Observed output:

```
RuntimeError: readout failed to learn: mean loss 1.5410 >= ln(K) 1.0986 (lr underdose or degenerate data)
classi-fly build: .../tools/train/train_readout.py failed: exit status 1
build exit=1
```

Diagnosis: the provisional pack's `in_scale=0.1` saturates states across the
scaffold's embedded basis directions; the ridge fit then produces a readout
whose mean cross-entropy on (mis-oriented, see Finding 1) scoring exceeds
ln(K) and the trainer's learn-check aborts. Note the scaffold pairs themselves
are separable — the loss gate is firing on the combination of saturated states
and Finding 1's wrong scoring path.

**Disposition: RESOLVED 2026-09-11 — the `classi-fly build` path now
VALIDATED.** Root cause was Finding 1 (the trainer's mis-oriented scoring
made the learn-check fire on correct fits); after the Finding 1 fix, both
build end-to-end tests pass unchanged against the same provisional pack
(`TestBuildSyntheticEndToEnd`, `TestBuildScaffoldPairs`):

```
go test ./cmd/classi-fly/ -run TestBuild -v
--- PASS: TestBuildSyntheticEndToEnd (0.23s)
--- PASS: TestBuildScaffoldPairs (0.18s)
--- PASS: TestBuildMissingPython (0.00s)
--- PASS: TestBuildUsage (0.00s)
```

The supported shipping path remains `tools/ingest` -> `tools/train` with real
pairs -> `classi-fly pack`; `build` is the convenience wrapper and is now
covered by passing end-to-end tests.

### Finding 3 — No committed `results.json` (VERIFICATION SCOPE NOTE)

The repo contains **zero** committed `results.json` files (`git`-tracked tree
checked). The eval harness (leaf 05) produces them via `run_eval.py --out`,
but no real-corpus campaign run has been committed. Consequently the only
scored results that exist today are:

1. the harness's own unit-test fixtures (`tools/eval/test_run_eval.py`), and
2. the synthetic results file crafted for this leaf
   (`tools/verify/testdata/synthetic_results.json`), hand-checked against its
   per-case rows.

**There is therefore no real-corpus headline (no n=48 replay number, no
connectome judge-mode number) to verify.** Any future headline claim must be
re-run through `tools/verify/recompute.py` before it is believed; this report
gives the per-experiment verdicts accordingly (all UNVALIDATED on real data).

### Finding 4 — Quantization drift: none observed on twin artifacts (PASS)

See §3. On the float-vs-int8 twin pair, the int8 round trip is bit-exact with
the float reference (both representations store values already on the int8
grid), max state delta 0.0, max probability delta 0.0, 0 classification and 0
route disagreements across 64 probes. This verifies the **container +
reference forward pass** agreement, i.e. no reader-side drift is introduced
between the two storage modes. It does NOT yet verify a large production
artifact (none exists to test — see Finding 3).

### Finding 5 — License provenance clean in the shipped tree (PASS)

See §4. The only committed `.fly` (`tools/eval/eval_fixture.fly`) is
synthetic, license `CC0-1.0`, attribution carried in its header, and NOTICE
covers its source family. No non-commercial artifact is present, and the pack
path refuses FlyWire provenance without `--allow-noncommercial`
(`tools/ingest/test_license.py`, verified by its own suite).

### Finding 6 — Fixture corpus licensing (PASS)

`testdata/eval-fixtures/fixture_corpus.json5` is original fixture text
(`source: "eval-fixture"` on all 19 items), no third-party data. Classified
shippable/benchmark-irrelevant.

---

## 2. Headline recomputation (Task 1)

Tool: `tools/verify/recompute.py` — walks `predictions[]`, recounts
routes/abstentions/correctness/ood with its own formulas, and compares to the
harness's stored headline. Independent of `run_eval.py`'s aggregation.

Formulas (independent restatement):

- P = routed_correct / routes
- C = routes / total
- E2E = (routed_correct + chain_baseline × abstained) / total  — deterministic
  chain credit, never a sampled draw
- OOD_R = ood_abstained / ood_total

### 2.1 Synthetic results file (hand-built, counts verified by hand)

Fixture: `tools/verify/testdata/synthetic_results.json` (19 cases: 18 ID + 1
OOD; head routes all 19 with 2 routed-wrong; judge routes 14/19 with all 14
routed-correct and 5 abstains including the OOD case; chain credit 0.868).

Hand calculation: head P = 17/19 = 0.8947368, C = 19/19 = 1.0,
E2E = (17 + 0.868·0)/19 = 0.8947368; judge P = 14/14 = 1.0, C = 14/19 =
0.7368421, E2E = (14 + 0.868·5)/19 = 0.9652632, OOD_R = 1/1 = 1.0.

```
python3 tools/verify/recompute.py tools/verify/testdata/synthetic_results.json
```

Result: **all 8 metrics agree to 1e-9** with the hand calculation and with the
stored headline; exit 0. Tamper test: editing the stored `E2E` to 0.999 makes
the tool exit 1 with a `DISAGREEMENTS` block (covered by
`test_recompute.py::test_cli_fails_on_tampered_results`), and a cooked
headline (E2E/P inflated beyond what the rows support) is flagged
(`test_recompute_flags_a_cooked_headline`), as is a lying route count
(`test_recompute_flags_route_count_undisclosure`).

### 2.2 Real campaign results

None exist (Finding 3). Nothing to recompute; verdict UNVALIDATED.

### 2.3 Route-count discipline check on the judge-mode fixture

`recompute.verdict_lines` on the synthetic file:

```
judge routes/total: 14/19 - beats the floor by 1.36 cases on 14 routes
head  routes/total: 19/19 - beats the floor by 0.51 cases on 19 routes - SUB-ONE-CASE MARGIN, noise at this n
```

The margin-over-floor arithmetic (margin × routes, in cases) is the
disclosure instrument: any E2E win worth less than one correctly-routed case
is labeled noise. The success bars used are RESEARCH.md §7's: E2E > 87.35%
(existing veto), E2E > 86.80% (chain-only floor), P ≥ 97%, OOD_R ≥ 95%, route
count mandatory.

---

## 3. Quantization drift (Task 2)

Tool: `tools/verify/quant_drift.py`. Method (as amended by the orchestrator:
no Go `--trace` flag; Python reference only): craft two `.fly` twins with
identical topology/classes/readout — one seed-mode (float64 expansion of the
input projection, exactly as the reference reader expands it), one matrix-mode
with the same projection stored int8 and dequantized by `in_scale` — then run
both through the reference forward pass over 64 deterministic probes (basis
directions, negations, diagonal, uniform noise) and compare states,
probabilities, classes, and route/abstain decisions.

```
python3 tools/verify/quant_drift.py
```

Result (also asserted by `test_quant_drift.py`):

```
quant drift: 64 probes
  max |state| delta : 0.000e+00 (tol 5.0e-02) ok
  max prob delta    : 0.000e+00 (tol 1.0e-02) ok
  class disagreements: 0
  route disagreements: 0
  VERDICT: PASS - no classification changed
```

Interpretation: values that live on the int8 grid survive the
quantize/dequantize round trip exactly; the container's two input modes are
numerically interchangeable for grid-valued projections. Drift **can** only
enter when a producer stores non-grid-valued floats (impossible in this
container: everything is int8 by construction) — so the container+reader pair
is drift-free by design, and this test proves it. A production-scale
check against the Go binary remains possible future work once a real artifact
exists; the Go side carries its own parity tests
(`reservoir/fly_test.go`), which were executed as part of leaf 01/02 review
(`go test ./...`), not re-run here.

---

## 4. License audit (Task 3)

Tool: `tools/verify/license_audit.py`. It re-states the license policy
independently (does NOT import `tools/ingest/registry.py`), reads only the
container header of every `.fly` it can find under `testdata/` trees and
`tools/`, and classifies:

| header license | verdict |
|---|---|
| `none` / `synthetic` / `CC0-1.0` | SHIPPABLE |
| `CC-BY*` (any accepted variant) | SHIPPABLE **only with non-empty header attribution**, else UNKNOWN (fail-closed) |
| `CC-BY-NC*` (any variant) | BENCHMARK-ONLY |
| anything unrecognized | UNKNOWN (fail-closed — never shippable by default) |

It then cross-checks every distinct source family found against `NOTICE`
(Hemibrain / Larval / FlyWire / Synthetic paragraphs) and fails on gaps.

```
python3 tools/verify/license_audit.py
```

Result:

```
license audit: 1 artifact(s)
.../tools/eval/eval_fixture.fly   CC0-1.0   SHIPPABLE   notice=yes
distinct sources: synthetic-fixture
VERDICT: PASS - no non-commercial artifact is shippable and NOTICE covers every source
```

Test-verified policy behaviors (`test_license_audit.py`): a `CC-BY-NC-4.0`
header is BENCHMARK-ONLY and fails the audit; `synthetic`/`none`/`CC0`/`CC-BY`
are shippable; CC-BY without attribution is demoted to UNKNOWN; an
unrecognized license is UNKNOWN; a NOTICE missing the FlyWire paragraph is a
finding. The shipped tree passes with zero non-shippable artifacts.

---

## 5. Per-experiment verdicts

RESEARCH.md §7 bars, restated: judge-mode E2E must beat **87.35%** (existing
veto) **and 86.8%** (chain-only floor), with **P ≥ 97%**, **OOD-R ≥ 95%**, and
the **route count disclosed**; sub-one-case margins at n=48 are noise.

| experiment (RESEARCH.md §7) | claim under review | verdict | evidence line |
|---|---|---|---|
| E0 — synthetic `.fly` round trip, latency, determinism | container loads, classify deterministic | **SUPPORTED (fixture scale only)** | `tools/eval/test_run_eval.py::test_fixture_fly_loads_and_classifies` (8-neuron fixture); Go parity tests in `reservoir/fly_test.go`; §3 drift PASS. Real 3k-neuron latency/size numbers: never measured in-repo. |
| E1 — larval reservoir, readout held-out accuracy | head accuracy on held-out folds | **UNVALIDATED** | no trained artifact, no pairs file, no results.json in tree (Finding 3); additionally any such number produced today would be tainted by Finding 1. |
| E2 — judge mode vs 87.35%/86.8%/97%/95% bars on real traffic | E2E beats both bars | **UNVALIDATED** | same: no campaign results exist. The scoring formulas themselves are SUPPORTED as implemented (deterministic, auditable, recomputed to 1e-9 in §2.1) — but no real-data run exists to judge. |
| E2′ — judge mode on the 19-case synthetic fixture corpus | same bars | **UNSUPPORTED as a success claim** | `recompute.py` verdict on `tools/verify/testdata/synthetic_results.json`: judge clears E2E/P/OOD-R bars but on **14 routes of 19 synthetic fixture cases** — the corpus is a unit fixture, not the protocol's real-traffic replay; a "win" here is not a campaign result. Head mode additionally FAILS P≥97% and OOD-R≥95% and its floor margin is sub-one-case (noise) even on this toy set. |
| E3 — synthetic vs connectome parity | synthetic matches biology | **UNVALIDATED** | no connectome-trained artifact exists in-tree to compare. |
| E4 — FlyWire mushroom-body variant | optional, benchmark-only | **UNVALIDATED** (and correctly fenced) | no FlyWire artifact in tree; ingest refuses NC without ack (`tools/ingest/test_license.py::test_flywire_requires_ack`); §4 audit green. |
| E5 — reservoir as Stage-0 head | only if Tier 1 clears | **UNVALIDATED** | precondition not met; nothing to evaluate. |

Summary line for the final report writer: **no experiment has a SUPPORTED
real-data verdict.** The infrastructure (container, reader parity, scoring
protocol, license fence) is verified; every headline claim about accuracy is
either fixture-scale or does not yet exist.

---

## 6. Deviations and dispositions

| deviation | disposition |
|---|---|
| Task 2's spec suggested adding a Go `--trace` flag and driving the binary; the orchestrator's hand-off amended this to a pure-Python twin-artifact method via `flyio.py` | **accepted** (amended method implemented; see §3). The twin writer is local to `quant_drift.py` (independent of `tools/train/flybytes.py`) per the independence rule. |
| Orchestrator hand-off said the ridge bug is "correct only when N == K" | **refuted empirically** — N==K=4 also scores wrong (0.75 vs 1.0); the bug is unconditional except for contrived symmetric-W. Documented in Finding 1 and pinned by `test_square_case_is_also_wrong`. |
| Extra file beyond the leaf's interface contract: `tools/verify/ridge_orientation_repro.py` + `test_ridge_orientation.py` | **accepted** — the hand-off explicitly required an empirical, documented reproduction of Finding 1; the repro script is the exact command cited in this report. |
| `docs/VERIFICATION.md` reports on a synthetic results file rather than a real campaign | **forced** — no committed results.json exists (Finding 3). Not a silent substitution: the provenance of every number in §2 is named inline. |
| The `classi-fly build` UNVALIDATED marking (Finding 2) | **required by the hand-off**; reproduced with the exact command and output recorded above. |

No debug artifacts were left in the tree; all scratch files lived in `/tmp`
(outside the repo). No git operations were performed by this leaf.

---

## 7. Self-verification checklist (leaf 08)

- [x] Recomputation matches the harness to 1e-9 on every file it was given,
      and disagreements are *reported loudly* (tamper tests prove the tool
      refuses to bless a cooked headline)
- [x] Quantized vs float agreement tested (64 probes, 0 disagreements)
- [x] License audit refuses non-commercial artifacts (fail-closed on unknown
      licenses and attribution-less CC-BY too)
- [x] `docs/VERIFICATION.md` names the route count discipline (§2.3: margin
      × routes in cases; sub-one-case margins labeled noise) and the floor
      margins (87.35% veto / 86.8% chain floor from RESEARCH.md §7)
- [x] `python3 -m pytest tools/verify -q` → 25 passed
- [x] Deviations: all documented in §6 above
