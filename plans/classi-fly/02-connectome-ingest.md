# Connectome Ingest - Implementation Leaf

> **For the implementing agent:** You are the implementer for this leaf.
> Implement ALL tasks below using TDD. Do NOT commit - the orchestrator
> handles all git operations after review. Do NOT use read_file on existing
> source files - explore with search_files or terminal cat. After writing a
> file, do NOT read it back to verify - write once and stop. After completing,
> report what you built, what files you touched, and any deviations.

## Meta

- **Parent:** ../master.md
- **Scope:** Offline Python tooling that turns a fly connectome into an
  adjacency export usable by the `.fly` packer, with license enforcement.
- **Dependencies:** none at author time; consumes Contract 1 (header names)
- **Estimated Context:** 45K
- **Concurrency Group:** A

## Goal

Build `tools/ingest/` scripts that read a connectome source, collapse it to a
neuron-by-neuron signed adjacency matrix, and write a JSON adjacency export
with provenance. Support two sources: the larval connectome (a local file) and
the hemibrain (via neuPrint). Refuse to mark anything FlyWire-derived as
shippable.

## Context

Sources and licenses (from `../../RESEARCH.md` §4):
- Larval, Winding et al. 2023, Science - CC BY 4.0 - shippable. Data mirror:
  `github.com/tingshanL/BPU`, file `data/signed_connectivity_matrix.csv`.
- Hemibrain, Janelia FlyEM - CC-BY - shippable, reached via the neuPrint API.
- FlyWire (FAFB/BANC/MCNS/MANC) - CC BY-NC 4.0 - NON-COMMERCIAL, benchmark only.

Offline tooling is Python 3.12 (see `../SHARED-CONVENTIONS.md` §Language Split).
Runtime is Go and is NOT this leaf's concern.

## Interface Contracts (From Parent)

### What This Leaf Exposes

```
tools/ingest/larval.py
  python3 tools/ingest/larval.py --csv <path> --out <adjacency.json>
  # reads the signed connectivity CSV, emits:
  # {"name","neurons","edges","indptr":[..],"indices":[..],"weights":[..],
  #  "source":"larval-connectome","license":"CC-BY-4.0",
  #  "attribution":"Winding et al. 2023, Science 379:eadd9330"}

tools/ingest/hemibrain.py
  python3 tools/ingest/hemibrain.py --token <neuPrint token> --region MB \
      --out <adjacency.json>
  # source:"hemibrain", license:"CC-BY",
  # attribution:"Janelia FlyEM Hemibrain (CC-BY)"

tools/ingest/flywire.py
  python3 tools/ingest/flywire.py ... --out <adjacency.json>
  # source:"flywire", license:"CC-BY-NC-4.0"  # benchmark only; never shipped
```

Output is column-oriented (CSR) to match Contract 1 so the packer is a
straight copy: `indptr` length `neurons+1`, `indices` length `edges`,
`weights` length `edges` (ints; polarity applied as sign).

### What This Leaf Consumes

```
Contract 1 header field names from ../../master.md (source/license/attribution).
```

## Tasks

### Task 1: Larval CSV -> CSR adjacency export

**Objective:** Parse the larval signed connectivity CSV and emit the CSR JSON.

**Files:**
- Create: `tools/ingest/larval.py`
- Test: `tools/ingest/test_larval.py`

**Step 1: Write failing test** - build a tiny CSV fixture inline (a 3x3 signed
matrix), run the parser function on it, assert `neurons==3`, `edges==` the
non-zero count, `indptr` starts at 0, and `weights` carry the sign.

**Step 2: Run** `python3 -m pytest tools/ingest/test_larval.py -q`
Expected: FAIL - module missing.

**Step 3: Implement** a pure function `csv_to_csr(text: str) -> dict` plus a
`main()` that reads the file and writes JSON. Handle a comma-delimited square
matrix; skip zeros; keep the sign.

**Step 4: Run.** Expected: PASS.

### Task 2: Hemibrain via neuPrint

**Objective:** Query neuPrint for a region's adjacency and emit the same CSR
shape.

**Files:**
- Create: `tools/ingest/hemibrain.py`
- Test: `tools/ingest/test_hemibrain.py` (does NOT call the network; tests the
  transform on a recorded fixture response)

**Step 1: Write failing test** - given a recorded neuPrint-style response
dict, `response_to_csr()` returns a CSR dict with the hemibrain license and
attribution.

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement** the transform, plus a `--token`, `--region`, `--out`
CLI that issues the real query only when `--live` is passed.

**Step 4: Run.** Expected: PASS.

### Task 3: FlyWire guard (non-commercial)

**Objective:** Make it impossible to silently produce a shippable artifact
from FlyWire data.

**Files:**
- Create: `tools/ingest/flywire.py`
- Create: `tools/ingest/registry.py` (source -> license/attribution map)
- Test: `tools/ingest/test_license.py`

**Step 1: Write failing test**

```python
def test_flywire_requires_ack(tmp_path):
    # building a flywire export without allow_noncommercial=True must raise
```

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement** a `guard(source, allow_noncommercial)` used by every
ingest path; it raises `NonCommercialError` for `flywire` unless explicitly
acknowledged, and sets `license` accordingly in the output.

**Step 4: Run.** Expected: PASS.

## Self-Verification Checklist

- [ ] All three ingest paths produce the CSR shape Contract 1 expects
- [ ] License/attribution strings are exact and come from one registry
- [ ] FlyWire guard raises without acknowledgement; tested
- [ ] `python3 -m pytest tools/ingest -q` passes
- [ ] `ruff check tools/ingest` clean (if ruff present)
- [ ] No network calls in tests
- [ ] No deviations (or documented below)

**DO NOT COMMIT.** The orchestrator handles all git operations after review.

**Deviations from spec:** none

## Review Checklist (For Review Agent)

- [ ] Every Task implemented and tested
- [ ] Output JSON keys match Contract 1 exactly
- [ ] FlyWire guard tested and cannot be bypassed silently
- [ ] No line-number corruption; no debug artifacts; no hardcoded tokens
- [ ] Cross-references resolve (the CSV path and repro names exist)

Output: APPROVED or specific gaps with file + line references.

## Notes

- Do not download the connectome in this leaf's tests. Fixtures only. The real
  download is gated by OPEN-QUESTIONS Q3 and an owner approval.
- The larval CSV is large (~35 MB). Stream it; do not slurp into a dense
  Python list of lists if avoidable. A row-by-row scan is enough.
