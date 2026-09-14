# Tests of the three meept mappings

Run 2026-09-13. Each of the three mappings proposed in `docs/DIRECTIONS.md`
section 8 was tested against real data. Runners and raw rows:
`tools/eval/meept_test1_sequence.{py,_results.json}`,
`tools/eval/meept_test2_scaling.{py,_results.json}`,
`tools/eval/meept_test3_degraded.{py,_results.json}`.

Protocol everywhere: 5-fold stratified CV seed 42, per-class thresholds
calibrated on train-side scores, route only above threshold and margin 0.05,
P = route_correct/routes with routed-but-wrong kept in the denominator,
E2E = (route_correct + 0.868 x abstained) / total. 1 case = 0.28 pt at n=361.

---

## Test 1 - session context for the state-dependent intents: BLOCKED, and no gain in the reduced test

### Data audit (the headline)

The mapping was proposed on the theory that some intents depend on session
state. The data to test that does not exist yet:

| fact | measured |
|---|---|
| dispatch rows (`~/.meept/metrics.db` dispatch_log) | 611 |
| distinct sessions | 489 |
| sessions with more than one turn | 42 |
| sessions with three or more turns | 20 |
| rows where `corrected_agent` is populated | **0** |
| rows where `outcome` is not 'pending' | 6 |
| plans in `plans.db` | 640, **all `state='draft'`** |
| `plan_sessions` / `plan_signoffs` | **empty tables** |
| any table recording plan-active per turn | none |
| gold corpus cases carrying a session/turn field | **0** |

So: **zero labelled multi-turn sessions.** A sequence model cannot be trained,
and there is no recorded correction signal either - the same emptiness the
plasticity probe found (`tools/eval/plasticity_probe.py`).

### The reduced test that was still possible

On the quickplan-versus-code subset (58 + 81 = 139 cases):

| variant | correct / 139 | accuracy |
|---|---:|---:|
| per-message embedding classifier | 105 | 0.755 |
| + context features (cue, length, plan-active) | 105 | 0.755 |
| cue-only keyword rule | 105 | 0.755 |
| length-only rule | 101 | 0.727 |

Adding context changed **nothing: delta 0 of 139 cases**. The orchestration cue
fires on 32 of 58 quickplan cases and 8 of 81 code cases; out of fold it fixes
24 errors and breaks 24 - net zero.

### Verdict

- The mapping is **blocked on instrumentation, not on modelling.** The unblock
  is to populate `outcome` and `corrected_agent` and to harvest multi-turn
  sessions, not to build a sequence model.
- One incidental finding worth keeping: on this binary subset a keyword regex
  matches a trained embedding classifier exactly (105/139 both). Consistent
  with the campaign's conclusion that the quickplan-versus-code distinction is
  not a per-message text problem.

---

## Test 2 - the three scaling knobs: all real, small, and the procedure matters more than the config

| knob | spread in E2E | verdict |
|---|---:|---|
| penalty (ridge 1e-3 .. 1000) | 0.0177 | KNOB MATTERS - strongest |
| embedding normalisation | 0.0118 | KNOB MATTERS (centring helps; standardising needs its own penalty) |
| threshold calibration | 0.0052 | KNOB MATTERS, nominal (global vs per-class moves it; the precision target is inert on a 0.025 quantile grid) |

Best configuration found: linear probe on standardised embeddings, penalty 100,
a single global threshold.

| configuration | routes | P | C | A (all) | E2E |
|---|---:|---:|---:|---:|---:|
| committed probe baseline (penalty 0.1, per-class) | 88 | 0.9205 | 0.2438 | 0.7175 | 0.8808 |
| **best found (standardised, penalty 100, global threshold)** | **93** | **0.9570** | **0.2576** | 0.687 | **0.8909** |
| committed reservoir-states baseline (penalty 30) | 71 | 0.9577 | 0.1967 | 0.7258 | 0.8857 |

Delta versus the probe baseline: **+0.0101 E2E (about 3.6 cases), +5 routes,
+0.037 P**. Delta versus the states baseline: +0.0052 E2E (about 2 cases).

**Caveats that stop this being a win:** the OOD abstain rate falls from 0.929 to
**0.857**, below the 0.95 requirement; and a 3.6-case margin at n=361 is inside
noise. Treat the config as *unproven* and the procedure as *proven*.

**Verdict:** adopt the checklist - before any head work, sweep embedding
normalisation, the penalty, and global-versus-per-class thresholds. Do not adopt
this specific configuration without first re-checking OOD-R on the adjudicated
replay. This knob family has now cost two corrections (the 1e-3 penalty defect,
worth 9 points; and the projection scale in the control work).

---

## Test 3 - the degraded-embedding benefit: NO, and my earlier recommendation is retracted

Ungated accuracy (all 361 cases) and gated E2E:

| head | clean | trunc 50% | trunc 75% | dropout 50% | noise sigma 0.2 |
|---|---|---|---|---|---|
| (a) centroid + cosine margin | .704 / .874 | .673 / .871 | **.665** / .868 | **.643** / .869 | .224 / **.868** |
| (b) kNN-5 unanimity (ships) | .537 / .862 | .540 / .868 | .504 / .855 | .501 / .852 | .277 / .781 |
| (c) linear probe | .718 / .881 | .621 / .876 | .388 / .866 | .521 / .868 | .194 / .428 |
| (d) reservoir | **.726** / **.886** | **.687** / .870 | .576 / .868 | .573 / .870 | .188 / .730 |

Three findings:

1. **The robustness effect replicates.** Reservoir minus linear probe at 75%
   truncation is +0.188 accuracy = **+68 cases**, matching the earlier
   classification result. Reproduced independently by the orchestrator
   (centroid .665 vs reservoir .576 at trunc75 under the gate protocol).
2. **But it does not survive contact with meept's own proposed head.** Under
   missing input the centroid head is *better* than the reservoir: +32 cases at
   75% truncation, +25 at 50% dropout. Averaging the surviving dimensions beats
   pushing them through a fixed random projection.
3. **And the shipping gate erases whatever edge remains.** At 75% truncation
   every head lands within 0.002 of the 0.868 chain floor, with the reservoir
   routing zero cases - the gate correctly abstains, discarding the extra
   accuracy. The benefit exists only in a metric meept does not use.

**A safety finding worth more than the mapping:** under dense noise the
reservoir is the *less safe* head. At sigma 0.2 it routes 77 cases at precision
0.22 (E2E 0.730), because softmax-probability margins do not collapse when the
representation is garbage. The cosine-margin head abstains and keeps E2E 0.868.
**Cosine and vote margins fail safe; probability margins do not.** That is an
argument for the centroid head shape proposed in meept issue #39.

**Cost:** reservoir artifact ~1.7 MB compressed and ~30 MB resident, versus
13 KB for the int8 centroid set - about 130x - for no benefit.

**Verdict: do not add the reservoir for degraded embeddings.** The conditional
mapping in `docs/DIRECTIONS.md` section 8.3 is retracted.

---

## Consolidated verdict

| mapping | verdict | what would change it |
|---|---|---|
| 8.1 session context | **BLOCKED** - zero labelled multi-turn sessions; reduced test shows delta 0 | populate `outcome`/`corrected_agent`; harvest multi-turn sessions with labels |
| 8.2 scaling checklist | **ADOPT THE PROCEDURE** - all three knobs real; best config +0.0101 E2E but OOD-R regresses to 0.857 | a config that gains E2E without losing OOD-R, validated on the replay corpus |
| 8.3 degraded embeddings | **RETRACTED** - centroid beats the reservoir under missing input, and the gate erases the edge | a consumer that actually has a degraded-embedding path and no centroid head |

No mapping argues for adding a reservoir to meept. Two of the three produce
something usable anyway: an instrumentation fix for meept, and a tuning
checklist plus the "margins must fail safe" argument for the centroid head.
