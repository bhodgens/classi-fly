# CLI + Packaging - Implementation Leaf

> **For the implementing agent:** You are the implementer for this leaf.
> Implement ALL tasks below using TDD. Do NOT commit - the orchestrator
> handles all git operations after review. Do NOT use read_file on existing
> source files - explore with search_files or terminal cat. After writing a
> file, do NOT read it back to verify - write once and stop. After completing,
> report what you built, what files you touched, and any deviations.

## Meta

- **Parent:** ../master.md
- **Scope:** The `classi-fly` command-line tool: pack an artifact, inspect it,
  and classify an embedding. Plus module packaging and release hygiene.
- **Dependencies:** Contract 1, Contract 2
- **Estimated Context:** 70K
- **Concurrency Group:** B

## Goal

Ship a single static binary that a consumer can run without Go. Subcommands:
`pack` (assemble a `.fly` from an adjacency export + readout export),
`build` (convenience: synthetic + train in one step), `info`, and `classify`.
Also finalize the Go module (README, NOTICE, versioning).

## Context

The runtime library is `reservoir` (Contract 2). Offline tooling writes the
intermediate JSON exports (Contract 3). `pack` is the bridge from those exports
to a `.fly`. All Go; no CGO.

## Interface Contracts (From Parent)

### What This Leaf Exposes

```
classi-fly pack --adjacency a.json --readout r.json --out model.fly
classi-fly info model.fly
classi-fly classify model.fly --embedding-file vec.json
classi-fly classify model.fly --embedding 0.1,0.2,0.3   # comma-separated
```

Exit codes: 0 ok, 1 usage error, 2 load/validation error, 3 non-commercial
artifact refused without `--allow-noncommercial`.

`info` prints the header JSON verbatim to stdout.
`classify` prints `{"class":...,"confidence":...,"margin":...,"abstained":...}`.

### What This Leaf Consumes

```
Contract 1: .fly format (WriteFile, Load).
Contract 2: reservoir.Load, Reservoir.Classify, Reservoir.Info.
Contract 3: adjacency + readout JSON shapes.
```

## Tasks

### Task 1: `pack`

**Objective:** Read the two exports and write a valid `.fly`.

**Files:**
- Create: `cmd/classi-fly/main.go`
- Create: `cmd/classi-fly/pack.go`
- Test: `cmd/classi-fly/pack_test.go`

**Step 1: Write failing test** - with fixture `adjacency.json` and
`readout.json`, `pack` writes a `.fly` that `reservoir.Load` reads back with
matching `Info()`. A FlyWire-licensed adjacency without
`--allow-noncommercial` exits 3.

**Step 2: Run** `go test ./cmd/classi-fly/ -run TestPack -v`
Expected: FAIL.

**Step 3: Implement** JSON readers for the two shapes, validation (dimension
agreement: `readout.classes == header.classes`, `adjacency.neurons ==
readout.W rows`), the license gate, and a call to `reservoir.WriteFile`.

**Step 4: Run.** Expected: PASS.

### Task 2: `info` and `classify`

**Objective:** Inspect and use an artifact.

**Files:**
- Create: `cmd/classi-fly/info.go`
- Create: `cmd/classi-fly/classify.go`
- Test: `cmd/classi-fly/classify_test.go`

**Step 1: Write failing test** - `info` on a fixture emits JSON containing
`"format":"fly-reservoir"`; `classify` on a fixture embedding emits a JSON
object with the four keys; a wrong-length embedding exits 2.

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement** using `reservoir.Load` and `Classify`.

**Step 4: Run.** Expected: PASS.

### Task 3: `build` (synthetic convenience)

**Objective:** One command to go from a seed to a `.fly`.

**Files:**
- Create: `cmd/classi-fly/build.go`

**Step 1: Write failing test** - `build --synthetic --seed 7 --classes
fixture_classes.json --pairs fixture_pairs.jsonl --out /tmp/x.fly` produces a
loadable `.fly` (it shells to the Python tooling; if Python is absent the
command reports a clear error and exits 1).

**Step 2: Run.** Expected: FAIL.

**Step 3: Implement** by invoking `tools/ingest/synthetic.py` and
`tools/train/train_readout.py` in a temp dir, then packing. Document that this
subcommand requires Python on PATH; it is a convenience, not the shipping path.

**Step 4: Run.** Expected: PASS.

### Task 4: Packaging hygiene

**Objective:** Make the module consumable and version-stamped.

**Files:**
- Create: `README.md`
- Create: `NOTICE`
- Create: `Makefile`
- Create: `LICENSE` (or confirm the intended license with the orchestrator)

**Step 1:** README documents: what it is, the `.fly` format, the Go API
(snippet), the sidecar face, the license rules, and how to build.

**Step 2:** `NOTICE` lists hemibrain (CC-BY) and larval (CC BY 4.0) sources and
states synthetic is license-free.

**Step 3:** `Makefile`: `build`, `test`, `lint`, `fmt`, `release`.

**Step 4:** `go build -trimpath -ldflags "-s -w" -o bin/classi-fly ./cmd/classi-fly`
and report the resulting binary size and the size of a fixture `.fly`.

## Self-Verification Checklist

- [ ] `pack`, `info`, `classify`, `build` implemented and tested
- [ ] License gate returns exit 3 for non-commercial sources
- [ ] Output JSON keys exact
- [ ] `go build ./...`, `go vet ./...`, `gofmt -l .` clean
- [ ] `go test ./... -race -count=10` passes
- [ ] README + NOTICE + Makefile present
- [ ] No deviations (or documented below)

**DO NOT COMMIT.** The orchestrator handles all git operations after review.

**Deviations from spec:** none

## Review Checklist (For Review Agent)

- [ ] Every Task implemented and tested
- [ ] Exit codes match the contract
- [ ] `info` output is the header JSON verbatim (no reformatting that loses keys)
- [ ] No meept references anywhere in this leaf's output
- [ ] No line-number corruption; no debug artifacts

Output: APPROVED or specific gaps with file + line references.

## Notes

- The binary must stay dependency-light. Do not add a CLI framework unless the
  standard `flag` package becomes unwieldy; stdlib `flag` is fine for four
  subcommands.
- Report the compressed `.fly` size in the leaf summary; the target is under
  5 MB, expected under 1 MB for the synthetic default.
