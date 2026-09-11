# Sparse Core (CSR + SpMV + forward pass) - Implementation Leaf

> **For the implementing agent:** You are the implementer for this leaf.
> Implement ALL tasks below using TDD. Do NOT commit - the orchestrator
> handles all git operations after review. Do NOT use read_file on existing
> source files - explore with search_files or terminal cat. After writing a
> file, do NOT read it back to verify - write once and stop. After completing,
> report what you built, what files you touched, and any deviations.

## Meta

- **Parent:** ../orchestrator.md
- **Scope:** Pure-Go CSR sparse matrix, SpMV, and the reservoir forward pass.
- **Dependencies:** none
- **Estimated Context:** 70K (exploration + generation + iteration + overhead)
- **Concurrency Group:** A

## Goal

Implement the numeric core of classi-fly: a compressed-sparse-row matrix type
with a matrix-vector product, and a reservoir forward pass that runs T steps of
a fixed sparse recurrence over an input vector. Pure Go. No CGO. Deterministic.

## Context

This is a new, empty repository. Module path is
`github.com/caimlas/classi-fly`. The runtime package is `reservoir`. This leaf
creates `reservoir/sparse.go` and `reservoir/reservoir.go` and their tests.

Projects to understand: none exist yet. You are the first leaf.

## Interface Contracts (From Parent)

### What This Leaf Exposes

```go
// File: reservoir/sparse.go
package reservoir

// csr is a compressed-sparse-row matrix with float64 values.
type csr struct {
    n     int       // rows (== cols; the reservoir matrix is square)
    indptr []uint32  // len n+1
    indices []uint32 // len nnz
    vals   []float64 // len nnz
}

func newCSR(n int, indptr []uint32, indices []uint32, vals []float64) (*csr, error)
func (m *csr) spmv(x []float64, out []float64) // out = m * x
```

```go
// File: reservoir/reservoir.go
package reservoir

// statefulForward runs steps of: s = tanh(decay*s + W*s + win*x)
// where W is the reservoir matrix and win is a fixed input projection.
type core struct {
    W      *csr
    win    []float64 // len n, fixed random projection (seed-derived)
    decay  float64
    steps  int
}

func (c *core) forward(x []float64) []float64
```

### What This Leaf Consumes

```
Nothing from sibling leaves. Sibling 02-fly-format.md will construct a *csr
via newCSR() and a *core via a constructor you define here.
```

## Tasks

### Task 1: CSR type and SpMV

**Objective:** Implement `csr` with a correct, deterministic matrix-vector
product.

**Files:**
- Create: `reservoir/sparse.go`
- Test: `reservoir/sparse_test.go`

**Step 1: Write failing test**

```go
package reservoir

import "testing"

func TestCSR_Spmv(t *testing.T) {
    // 3x3 matrix:
    //  2 0 1
    //  0 3 0
    //  4 0 5
    m, err := newCSR(3,
        []uint32{0, 2, 3, 5},
        []uint32{0, 2, 1, 0, 2},
        []float64{2, 1, 3, 4, 5},
    )
    if err != nil { t.Fatalf("newCSR: %v", err) }
    out := make([]float64, 3)
    m.spmv([]float64{1, 1, 1}, out)
    want := []float64{3, 3, 9}
    for i := range want {
        if out[i] != want[i] { t.Fatalf("out[%d]=%v want %v", i, out[i], want[i]) }
    }
}

func TestCSR_RejectsBadShape(t *testing.T) {
    if _, err := newCSR(3, []uint32{0, 2}, []uint32{0}, []float64{1}); err == nil {
        t.Fatal("expected error for indptr length mismatch")
    }
}
```

**Step 2: Run test to verify failure**

Run: `go test ./reservoir/ -run TestCSR -v`
Expected: FAIL - package does not build.

**Step 3: Write minimal implementation**

Implement `csr`, `newCSR` (validate `len(indptr)==n+1`, `len(indices)==len(vals)`,
and `indptr` non-decreasing), and `spmv` as the standard inner-loop:

```go
func (m *csr) spmv(x []float64, out []float64) {
    for i := 0; i < m.n; i++ {
        var sum float64
        for k := m.indptr[i]; k < m.indptr[i+1]; k++ {
            sum += m.vals[k] * x[m.indices[k]]
        }
        out[i] = sum
    }
}
```

**Step 4: Run test to verify pass**

Run: `go test ./reservoir/ -run TestCSR -v`
Expected: PASS.

### Task 2: Reservoir forward pass

**Objective:** Implement `core.forward` running `steps` of
`s = tanh(decay*s + W*s + win*x)`.

**Files:**
- Create: `reservoir/reservoir.go`
- Test: `reservoir/reservoir_test.go`

**Step 1: Write failing test**

```go
package reservoir

import "testing"

func TestCore_ForwardShapes(t *testing.T) {
    m, _ := newCSR(2, []uint32{0, 1, 2}, []uint32{1, 0}, []float64{1, 1})
    c := &core{W: m, win: []float64{1, 0}, decay: 0.9, steps: 4}
    got := c.forward([]float64{0.5})
    if len(got) != 2 { t.Fatalf("len=%d", len(got)) }
    for _, v := range got {
        if v != v || v > 1 || v < -1 { t.Fatalf("state out of range: %v", v) }
    }
}

func TestCore_ForwardDeterministic(t *testing.T) {
    m, _ := newCSR(3, []uint32{0,1,2,3}, []uint32{1,2,0}, []float64{0.5,0.5,0.5})
    c := &core{W: m, win: []float64{1,0.5,0.25}, decay: 0.8, steps: 6}
    a := c.forward([]float64{0.3})
    b := c.forward([]float64{0.3})
    for i := range a {
        if a[i] != b[i] { t.Fatalf("nondeterministic at %d: %v vs %v", i, a[i], b[i]) }
    }
}
```

**Step 2: Run test to verify failure**

Run: `go test ./reservoir/ -run TestCore -v`
Expected: FAIL - `core` undefined.

**Step 3: Write minimal implementation**

```go
func (c *core) forward(x []float64) []float64 {
    n := c.W.n
    s := make([]float64, n)   // zero initial state
    recur := make([]float64, n)
    drive := make([]float64, n)
    for i := 0; i < n && i < len(c.win); i++ {
        drive[i] = c.win[i] * x[0] // scalar input in the fixture; see Notes
    }
    for t := 0; t < c.steps; t++ {
        c.W.spmv(s, recur)
        for i := 0; i < n; i++ {
            s[i] = math.Tanh(c.decay*s[i] + recur[i] + drive[i])
        }
    }
    return s
}
```

**Step 4: Run test to verify pass**

Run: `go test ./reservoir/ -run TestCore -v`
Expected: PASS.

### Task 3: Determinism and race guard

**Objective:** Confirm no map iteration or float-equality instability.

**Files:**
- Modify: `reservoir/reservoir_test.go`

**Step 1:** Add a test that calls `forward` 100 times and asserts identical
output, then run `go test ./reservoir/ -race -count=10`.

**Step 2:** Run: `go test ./reservoir/ -race -count=10`
Expected: PASS, no race reports.

**Step 3:** If it fails, the cause is a map iteration in an accumulation or a
`!=` float compare - fix by sorting keys and using an epsilon.

**Step 4:** Re-run until PASS.

## Self-Verification Checklist

- [ ] `csr`, `newCSR`, `spmv` implemented and tested
- [ ] `core`, `core.forward` implemented and tested
- [ ] `go build ./...`, `go vet ./...`, `gofmt -l .` clean
- [ ] `go test ./reservoir/ -race -count=10` passes
- [ ] No CGO; no map-iteration nondeterminism
- [ ] No deviations from spec (or documented below)

**DO NOT COMMIT.** The orchestrator handles all git operations after review.

**Deviations from spec:** none

## Review Checklist (For Review Agent)

- [ ] Every Task above is implemented
- [ ] Every test in the task is present and passing
- [ ] Interface signatures match exactly (`newCSR`, `spmv`, `core.forward`)
- [ ] Code follows `../SHARED-CONVENTIONS.md`
- [ ] No bugs, no scope creep

Output: APPROVED or specific gaps with file + line references.

## Notes

- The input projection `win` maps the embedding to the reservoir. For the
  tiny N in tests, a scalar input is enough. The real input handling (a
  D-dimensional embedding projected into N neurons) is the responsibility of
  the `Classify` wrapper in `classify.go` (owned by 02/05 as noted in
  master.md). Keep this leaf focused on the recurrence itself.
- `decay` is a leak term in (0,1). Use 1.0 only if you document it.
