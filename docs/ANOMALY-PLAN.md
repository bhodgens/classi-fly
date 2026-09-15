# Anomaly detection for meept streams — hierarchical plan

Companion to `docs/DIRECTIONS.md`, `docs/MEEPT-TESTS.md`, and meept issues
#40-#43. This plan covers the three anomaly-detection use cases and the
outcome-loop fix (meept issue #40).

## Owner decision: OOD policy = safety (2026-09-13)

## Status

| feature | status | verdict |
|---|---|---|
| outcome loop (issue #40) | ALIVE, gap is data volume | corrected diagnosis in issue #40 comment; loop works, 32 resolved outcomes |
| session drift (issue #41) | TESTED, NOT MEASURABLE | 15/42 flagged but the free rule captures the same signal |
| embedding health (issue #42) | TESTED, **per-item novelty wins** | 12/12 detected, 2.5-step latency; state-novelty is worse |
| failure-burst detection (issue #43) | blocked on outcome data | depends on #40 |
| cheap 512-neuron artifact | built and verified (12 KB) | tools/eval/cheap512_artifact.py |
| real-OOD well | EMPTY — 28/28 duplicates of the gold set | tools/eval/meept_drift_prototype_results.json |

## What remains actionable

The three anomaly features share a prerequisite (issue #40 outcome
instrumentation) and a common pattern (reservoir-based novelty detection on a
stream), but they differ in what data they need:

- **#42 embedding health**: buildable today. The per-item cosine distance to a
  fixed reference set is the right mechanism (2.5-step median latency, 0.83%
  false alarms). The reservoir's state-novelty is worse — use the simpler
  approach.
- **#41 session drift**: needs multi-turn sessions with labelled outcomes.
  The 42 existing multi-turn sessions have 2-3 turns and no outcome labels.
  Revisit when 100+ labelled sessions exist.
- **#43 burst detection**: same prerequisite as #41, plus the outcome loop
  needs to produce enough corrections to be informative. Currently 4
  corrections across 611 rows.

## Phase 1 (implementable now)

1. Implement the embedding-pipeline health check (meept issue #42) using
   cosine distance to a fixed reference set. This is the mechanism the
   classi-fly research proved works (12/12 detected, 2.5-step median latency).
2. Wire it into the daemon as a config-gated check.
3. Add a `/api/v1/embedder/health` endpoint.

## Phase 2 (after #40 instrumentation lands and data accumulates)

4. Re-run the session-drift prototype when 100+ multi-turn sessions exist.
5. Implement the tool-failure burst detector.

## Phase 3 (research / optional)

6. Test whether the rhythm-task result (the real connectome beat synthetic on
   anomaly detection) transfers to embedding streams — the lane 3 result
   suggested it might, but the embed-stream test showed state-novelty is worse
   than per-item novelty for this purpose.
