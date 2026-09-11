# Adversarial Verification + License Audit - Implementation Leaf

> **For the implementing agent:** You are the implementer for this leaf.
> Implement ALL tasks below. This leaf is primarily verification, not feature
> work. Do NOT commit - the orchestrator handles all git operations after
> review. Do NOT use read_file on existing source files - explore with
> search_files or terminal cat.

## Meta

- **Parent:** ../master.md
- **Scope:** Independently reproduce the harness numbers, audit every shipped
  artifact's license provenance, and stress the quantized model.
- **Dependencies:** all other leaves COMPLETE
- **Estimated Context:** 70K
- **Concurrency Group:** C

## Goal

Do not trust the harness's own summary. Recompute the headline numbers from the
per-case data, prove the quantized `.fly` matches the float reference within
tolerance, and prove no non-commercial data can reach a shippable artifact.

This leaf exists because a prior campaign found a headline that was a
measurement artifact (a sampled draw where the protocol required deterministic
credit) and a 2-route win presented as a victory. Verification is the point.

## Context

Read `../../RESEARCH.md` §7 (success bars) and §8 (risks). The success bar for
judge mode: E2E beats 87.35% (the existing veto) and 86.8% (chain-only floor),
with P >= 97% and OOD-R >= 95%, and the route count disclosed. A sub-one-case
margin at n=48 is noise and must be labeled as such.

## Interface Contracts (From Parent)

### What This Leaf Exposes

```
tools/verify/recompute.py      # recompute P/C/E2E/OOD-R from results.json
tools/verify/license_audit.py  # walk every testdata/*.fly and check provenance
tools/verify/quant_drift.py    # float vs int8 output agreement
docs/VERIFICATION.md           # the findings, with evidence
```

### What This Leaf Consumes

```
results.json from 05-eval-harness.md; .fly artifacts; the .fly header.
```

## Tasks

### Task 1: Recompute the headline from per-case data

**Files:**
- Create: `tools/verify/recompute.py`
- Test: `tools/verify/test_recompute.py`

**Step 1: Write failing test** - given a hand-built results fixture with known
counts, `recompute()` yields the same P, C, E2E, OOD-R as a hand calculation,
and E2E uses the deterministic formula.

**Step 2: Run** `python3 -m pytest tools/verify/test_recompute.py -q`
Expected: FAIL.

**Step 3: Implement** the recomputation, and print a table comparing the
harness's reported numbers to the recomputed ones. Any difference is a finding.

**Step 4: Run.** Expected: PASS.

### Task 2: Quantization drift

**Files:**
- Create: `tools/verify/quant_drift.py`
- Test: `tools/verify/test_quant_drift.py`

**Step 1: Write failing test** - a float reference and its int8 quantization
produce classification results that agree on all fixture embeddings, and the
max state difference is within the documented tolerance.

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement** by loading the `.fly`, running the Go forward pass via a
small `classi-fly classify --trace` mode (add a `--trace` flag that prints the
final state vector), and comparing to a float reference. If drift changes any
classification, that is a FAIL finding.

**Step 4: Run.** Expected: PASS.

### Task 3: License audit

**Files:**
- Create: `tools/verify/license_audit.py`
- Test: `tools/verify/test_license_audit.py`

**Step 1: Write failing test** - a `.fly` whose header `license` is
`CC-BY-NC-4.0` is reported as NON-SHIPPABLE; a `synthetic` one as shippable;
the repo `NOTICE` lists every distinct source found.

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement** by walking `testdata/` and the release directory, reading
each `.fly` header, and classifying shippable vs benchmark-only. Cross-check
against `NOTICE`.

**Step 4: Run.** Expected: PASS.

### Task 4: Adversarial review of the claims

**Files:**
- Create: `docs/VERIFICATION.md`

**Step 1:** For every number in the final report, state: the source artifact,
the exact command that produced it, the recomputed value, and whether they
agree.

**Step 2:** For the judge-mode result specifically, state the ROUTE COUNT and
the margin over the floor, and label sub-one-case margins as noise at n=48.

**Step 3:** List every deviation found and its disposition (fixed / accepted).

**Step 4:** Give a verdict per experiment: SUPPORTED, UNSUPPORTED, or
UNVALIDATED, with the evidence line that supports it.

## Self-Verification Checklist

- [ ] Recomputation matches the harness to the last digit, or the difference is
  reported as a finding
- [ ] Quantized vs float agreement tested
- [ ] License audit refuses non-commercial artifacts
- [ ] `docs/VERIFICATION.md` names the route count and the floor margin
- [ ] `python3 -m pytest tools/verify -q` passes
- [ ] No deviations (or documented below)

**DO NOT COMMIT.** The orchestrator handles all git operations after review.

**Deviations from spec:** none

## Review Checklist (For Review Agent)

- [ ] Every Task implemented and tested
- [ ] The verification is INDEPENDENT: it must not call the harness's summary
  function to compute the headline (that would be self-confirmation)
- [ ] Any disagreement between harness and recomputation is reported, not hidden
- [ ] No line-number corruption; no debug artifacts

Output: APPROVED or specific gaps with file + line references.

## Notes

- The honest outcome may be "the reservoir adds nothing measurable." That is a
  valid, valuable result. Report it plainly; do not massage it.
- If the synthetic reservoir matches the connectome, say so - it makes the
  license question moot and simplifies shipping.
