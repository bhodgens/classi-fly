# Synthetic Reservoir Generator - Implementation Leaf

> **For the implementing agent:** You are the implementer for this leaf.
> Implement ALL tasks below using TDD. Do NOT commit - the orchestrator
> handles all git operations after review. Do NOT use read_file on existing
> source files - explore with search_files or terminal cat. After writing a
> file, do NOT read it back to verify - write once and stop. After completing,
> report what you built, what files you touched, and any deviations.

## Meta

- **Parent:** ../master.md
- **Scope:** A seed-driven generator for a mushroom-body-shaped sparse
  reservoir. No connectome data, no license constraints. The license-clean
  default artifact.
- **Dependencies:** none at author time; consumes Contract 1
- **Estimated Context:** 40K
- **Concurrency Group:** A

## Goal

Generate a fixed sparse recurrent matrix with the architectural shape of the
mushroom body: a few "projection" inputs expand into many sparsely connected
internal units, which feed a small output fan-in. The output is a CSR adjacency
in the same shape Contract 1 expects, plus fixed input and output neuron
indices, so the packer can build a `.fly` with zero third-party data.

Rationale (from `../../RESEARCH.md` §4): the mushroom body's principle - sparse
expansion of a low-dimensional code - is not copyrightable. A seed-generated
matrix with that shape gives the same computation with no license risk.

## Context

Shape parameters (all from the seed):
- N internal units (Kenyon-like), default 2048.
- Fan-in per internal unit: small (default 6), drawn without replacement.
- A fraction of inhibitory edges (default 20%), sign applied to the weight.
- Weights drawn from a fixed distribution scaled so the spectral radius stays
  below 1 (reservoir stability). Use a scale factor and document it.

## Interface Contracts (From Parent)

### What This Leaf Exposes

```
tools/ingest/synthetic.py
  python3 tools/ingest/synthetic.py --seed 7 --neurons 2048 --fan-in 6 \
      --out <adjacency.json>
  # emits the CSR JSON with:
  #   source:"synthetic", license:"none",
  #   attribution:"seed-generated (classi-fly); no third-party data"
```

Determinism: identical `--seed` and parameters must yield a byte-identical
`indptr`/`indices`/`weights`. Use a seeded PRNG (`random.Random(seed)`) and
never rely on set/dict iteration order.

### What This Leaf Consumes

```
Contract 1 field names from ../../master.md.
```

## Tasks

### Task 1: Deterministic adjacency generation

**Objective:** `generate(seed, n, fan_in, inhib_frac) -> dict` producing CSR.

**Files:**
- Create: `tools/ingest/synthetic.py`
- Test: `tools/ingest/test_synthetic.py`

**Step 1: Write failing test**

```python
def test_deterministic():
    a = generate(seed=7, n=64, fan_in=4, inhib_frac=0.2)
    b = generate(seed=7, n=64, fan_in=4, inhib_frac=0.2)
    assert a["indptr"] == b["indptr"]
    assert a["indices"] == b["indices"]
    assert a["weights"] == b["weights"]

def test_different_seed_differs():
    a = generate(seed=1, n=64, fan_in=4, inhib_frac=0.2)
    b = generate(seed=2, n=64, fan_in=4, inhib_frac=0.2)
    assert a["indices"] != b["indices"]

def test_csr_shape():
    a = generate(seed=3, n=32, fan_in=5, inhib_frac=0.2)
    assert len(a["indptr"]) == 33
    assert len(a["indices"]) == len(a["weights"]) == a["edges"]
    assert a["indptr"][0] == 0 and a["indptr"][-1] == a["edges"]
```

**Step 2: Run** `python3 -m pytest tools/ingest/test_synthetic.py -q`
Expected: FAIL.

**Step 3: Implement** `generate` using `random.Random(seed)`, per-row sample of
`fan_in` distinct targets, weight sign from `inhib_frac`, magnitude from a
fixed distribution. Return the CSR dict with license/attribution filled.

**Step 4: Run.** Expected: PASS.

### Task 2: Stability check

**Objective:** Reject a generated matrix whose spectral radius is too high.

**Files:**
- Modify: `tools/ingest/synthetic.py`
- Modify: `tools/ingest/test_synthetic.py`

**Step 1: Write failing test** - `spectral_radius_estimate(csr)` is below 0.99
for defaults, and `generate` raises if asked for a scale that pushes it over.

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement** a power-iteration estimate (stdlib only; sparse). Apply
a global scale so the estimate stays below a target (default 0.9); if the
caller forces an unsafe scale, raise.

**Step 4: Run.** Expected: PASS.

## Self-Verification Checklist

- [ ] `generate` deterministic for a fixed seed; differs across seeds
- [ ] CSR shape matches Contract 1
- [ ] Spectral-radius guard tested
- [ ] `python3 -m pytest tools/ingest/test_synthetic.py -q` passes
- [ ] No third-party data referenced anywhere in output or code
- [ ] No deviations (or documented below)

**DO NOT COMMIT.** The orchestrator handles all git operations after review.

**Deviations from spec:** none

## Review Checklist (For Review Agent)

- [ ] Every Task implemented and tested
- [ ] Output keys match Contract 1 exactly; `license` is `none`
- [ ] Determinism proven by the test
- [ ] No line-number corruption; no debug artifacts

Output: APPROVED or specific gaps with file + line references.

## Notes

- This leaf is the license-clean path. If the harness later shows the real
  connectome beats synthetic, the synthetic stays the shipped default unless
  the owner says otherwise (OPEN-QUESTIONS Q4).
- Keep generation O(N * fan_in). For N=2048 and fan_in=6 this is trivial.
