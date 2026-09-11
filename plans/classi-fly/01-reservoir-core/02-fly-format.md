# `.fly` Format (load/save + quantize) - Implementation Leaf

> **For the implementing agent:** You are the implementer for this leaf.
> Implement ALL tasks below using TDD. Do NOT commit - the orchestrator
> handles all git operations after review. Do NOT use read_file on existing
> source files - explore with search_files or terminal cat. After writing a
> file, do NOT read it back to verify - write once and stop. After completing,
> report what you built, what files you touched, and any deviations.

## Meta

- **Parent:** ../orchestrator.md
- **Scope:** The versioned `.fly` binary container: encode, decode, quantize,
  and the public `Load`/`Info` API.
- **Dependencies:** 01-sparse-core.md (`csr`, `newCSR`, `core`).
- **Estimated Context:** 75K
- **Concurrency Group:** B

## Goal

Implement reading and writing of the `.fly` artifact defined in
`../../master.md` §Interface Contracts §Contract 1. The file is a magic +
JSON header + packed int8/uint32 arrays, zstd-compressed. Loading dequantizes
into the `csr` and `core` types from sibling 01, validates everything, and
exposes `Load` and `Info`.

## Context

Sibling 01 defines (inline, do not re-read the file):

```go
type csr struct { n int; indptr []uint32; indices []uint32; vals []float64 }
func newCSR(n int, indptr []uint32, indices []uint32, vals []float64) (*csr, error)
func (m *csr) spmv(x []float64, out []float64)

type core struct { W *csr; win []float64; decay float64; steps int }
func (c *core) forward(x []float64) []float64
```

Use these as-is. Do not modify sibling 01's file.

## Interface Contracts (From Parent)

### What This Leaf Exposes

```go
package reservoir

type Info struct {
    Name        string
    Neurons     int
    Edges       int
    EmbedDim    int
    Steps       int
    Classes     []string
    License     string
    Attribution string
}

func Load(path string) (*Reservoir, error)
func (r *Reservoir) Info() Info
// Encode/write side (used by the CLI packer 06-cli-package.md):
func WriteFile(path string, a Artifact) error

type Artifact struct {
    Info    Info
    WeightScale float64
    Source  string
    CreatedUTC string
    // graph + readout fields per Contract 1
}
```

### What This Leaf Consumes

```
From 01-sparse-core.md: csr, newCSR, spmv, core, core.forward.
From ../../master.md: Contract 1 header JSON field names (frozen).
```

## Tasks

### Task 1: Header encode/decode

**Objective:** Marshal and unmarshal the frozen header JSON with all required
fields, rejecting unknown/missing required fields.

**Files:**
- Create: `internal/flyformat/header.go`
- Test: `internal/flyformat/header_test.go`

**Step 1: Write failing test**

```go
package flyformat

import "testing"

func TestHeaderRoundTrip(t *testing.T) {
    h := Header{Format: "fly-reservoir", Version: 1, Name: "t",
        Neurons: 8, Edges: 10, EmbedDim: 4, Steps: 3,
        Classes: []string{"a", "b"}, WeightScale: 0.01,
        Source: "synthetic", License: "none", Attribution: "n/a",
        CreatedUTC: "2026-09-11T00:00:00Z"}
    b, err := MarshalHeader(h)
    if err != nil { t.Fatal(err) }
    got, err := UnmarshalHeader(b)
    if err != nil { t.Fatal(err) }
    if got.Neurons != 8 || got.WeightScale != 0.01 || len(got.Classes) != 2 {
        t.Fatalf("mismatch: %+v", got)
    }
}

func TestHeaderRejectsBadVersion(t *testing.T) {
    if _, err := UnmarshalHeader([]byte(`{"format":"fly-reservoir","version":99}`)); err == nil {
        t.Fatal("expected version rejection")
    }
}
```

**Step 2: Run test to verify failure**

Run: `go test ./internal/flyformat/ -run TestHeader -v`
Expected: FAIL - package undefined.

**Step 3: Write minimal implementation** - a `Header` struct with JSON tags
matching Contract 1 exactly (`format`, `version`, `name`, `neurons`, `edges`,
`embed_dim`, `steps`, `classes`, `weight_scale`, `source`, `license`,
`attribution`, `created_utc`), plus `MarshalHeader`/`UnmarshalHeader` that
reject `format != "fly-reservoir"` or `version != 1`.

**Step 4: Run test to verify pass**

Run: `go test ./internal/flyformat/ -run TestHeader -v`
Expected: PASS.

### Task 2: Container encode/decode with zstd

**Objective:** Wrap the header + payload in magic + lengths, zstd-compress,
and decompress on read.

**Files:**
- Create: `internal/flyformat/container.go`
- Test: `internal/flyformat/container_test.go`

**Step 1: Write failing test** - marshal a small payload, write to a temp
file via the container writer, read it back, assert the magic is `FLYRES01`
and the payload bytes are identical after decompression.

**Step 2: Run** `go test ./internal/flyformat/ -run TestContainer -v`
Expected: FAIL.

**Step 3: Implement** using `github.com/klauspost/compress/zstd`. Before
adding the dependency, read its LICENSE under `go env GOMODCACHE`; if it is
GPL/AGPL, STOP and report. (klauspost/compress is BSD-3; confirm, do not
assume.) If the module cache has nothing yet, add it to `go.mod` and run
`go mod tidy`.

**Step 4: Run** the test. Expected: PASS.

### Task 3: `Load` and `Info` (int8 dequantization)

**Objective:** Load a `.fly` into a `*Reservoir`, dequantize weights, validate
shapes, and expose `Info`. Fail closed.

**Files:**
- Create: `reservoir/fly.go`
- Create: `reservoir/classify.go` (the `Reservoir`, `Result`, `Info` types;
  the `Classify` method may be a stub returning an error here and is completed
  by 05/06 - keep the type definitions exact).
- Test: `reservoir/fly_test.go`

**Step 1: Write failing tests**

```go
func TestLoad_MissingFileFailsClosed(t *testing.T) {
    if _, err := Load("/no/such/file.fly"); err == nil {
        t.Fatal("expected error for missing file")
    }
}

func TestLoad_RoundTripDims(t *testing.T) {
    // Build a tiny Artifact, WriteFile to t.TempDir()/x.fly, Load it back.
    // Assert Info().Neurons, Edges, Classes match, and that a second Load
    // yields an identical in-memory matrix (compare a checksum).
}
```

**Step 2: Run** `go test ./reservoir/ -run TestLoad -v`
Expected: FAIL.

**Step 3: Implement** `WriteFile` (in `fly.go`) and `Load` (validate magic,
header, array lengths; dequantize `vals[i] = float64(weights[i]) * weight_scale`;
construct `newCSR`; build the `core`; build the readout). `Info()` returns the
header fields. Any inconsistency -> error, never a partial reservoir.

**Step 4: Run** the test. Expected: PASS.

### Task 4: Quantization tolerance test

**Objective:** Prove int8 round-trip stays within the documented tolerance.

**Files:**
- Modify: `reservoir/fly_test.go`

**Step 1:** Write a test that quantizes a known float matrix to int8 with a
chosen scale, loads it back, and asserts each element is within `scale` of the
original.

**Step 2:** Run it. Expected: PASS.

**Step 3:** Document the tolerance in the `WriteFile` doc comment.

**Step 4:** Run `go test ./reservoir/ -race -count=10`. Expected: PASS.

## Self-Verification Checklist

- [ ] Header encode/decode with exact Contract 1 field names
- [ ] Container magic `FLYRES01` + zstd round-trip
- [ ] `Load`, `WriteFile`, `Info`, `Reservoir`, `Result` present and typed exactly
- [ ] Load is fail-closed; corrupt/truncated file -> error
- [ ] int8 quantization within documented tolerance
- [ ] `go build ./...`, `go vet ./...`, `gofmt -l .` clean
- [ ] `go test ./... -race -count=10` passes
- [ ] Dependency license checked (zstd lib is permissive, confirmed)
- [ ] No deviations (or documented below)

**DO NOT COMMIT.** The orchestrator handles all git operations after review.

**Deviations from spec:** none

## Review Checklist (For Review Agent)

- [ ] Every Task implemented and tested
- [ ] Contract 1 field names match master.md EXACTLY (diff the strings)
- [ ] Contract 2 signatures match exactly
- [ ] Fail-closed behavior verified by the missing-file test
- [ ] No line-number corruption; no debug artifacts
- [ ] New dependency license read and permissive

Output: APPROVED or specific gaps with file + line references.

## Notes

- The `Reservoir` type's `Classify` may be minimally implemented here (state +
  readout argmax + threshold). The eval harness and CLI consume it. If you
  implement it fully, that is fine; do not leave it as an untested stub.
- Keep `WriteFile` symmetric with `Load` so the round-trip test is meaningful.
