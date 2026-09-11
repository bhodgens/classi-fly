# Reservoir Core - Implementation Orchestrator (branch)

> **For the executing agent:** You are the orchestrator for this branch.
> Dispatch its leaf implementation agents, review their work in-session,
> re-dispatch if incomplete, and track completion. Do NOT implement code
> yourself.

## Meta

- **Role:** Branch
- **Parent:** ../master.md
- **Children:** 2 leaves
- **Scope:** The pure-Go sparse core and the versioned `.fly` artifact container
  that the whole project depends on.

## Goal

Deliver the runtime foundation: a CSR sparse matrix with a fast
matrix-vector product, a reservoir forward pass over it, and the `.fly`
artifact reader/writer with int8 quantization. This is the de-risk milestone -
if this branch works, every other leaf has a stable substrate.

## Architecture

A reservoir is a fixed sparse recurrent matrix `W` (N x N). One forward step
is `state = tanh(W_state + W_in * input)` plus a decay term. We store `W` in
CSR with int8 weights and a single global scale. The `.fly` container is a
small magic + JSON header + packed arrays, then zstd-compressed. Loading
dequantizes into `[]float64` once and keeps the CSR in memory.

## Interface Contracts

### Contract 1: `.fly` format v1

See `../master.md` §Interface Contracts §Contract 1 for the exact byte layout
and header JSON field names. This branch owns it. Do not rename fields.

### Contract 2: Go package API

See `../master.md` §Interface Contracts §Contract 2. This branch owns the
types and the loader; `05-eval-harness.md` and `07-integration.md` consume
them. Exact signatures:

```go
package reservoir

func Load(path string) (*Reservoir, error)
func (r *Reservoir) Classify(embedding []float32) (Result, error)
func (r *Reservoir) Info() Info
```

`Classify` lives in `classify.go`; `sparse.go` owns `csr` and `spmv`;
`fly.go` owns `Load` and decode.

## Child Index

| # | Document | Type | Dependencies | Est. Context | Concurrency |
|---|----------|------|-------------|-------------|-------------|
| 01 | 01-sparse-core.md | leaf | none | 70K | A |
| 02 | 02-fly-format.md | leaf | 01 (uses `csr`) | 75K | B |

**Concurrency groups:** 01 before 02 - 02 compiles against the `csr` type
01 defines. Both share the frozen header field names from Contract 1.

## Dispatch Protocol

### Phase 1: Dispatch 01-sparse-core.md

1. **Read** `01-sparse-core.md` and dispatch via `delegate_task`:
   - Goal: "Implement all tasks from 01-sparse-core.md"
   - Context: full leaf text + Contract 1 (header names) + Contract 2
     (signatures) + `../SHARED-CONVENTIONS.md` §Coding Conventions
     §Determinism Rules INLINED.
   - Include: "Do NOT commit. Do NOT run git add. Write code, run tests,
     report results only."
   - Include: "Do NOT use read_file on existing source files - explore with
     search_files or terminal cat. After writing a file, do NOT read it back."

### Phase 2: Review 01, then Dispatch 02

1. Orchestrator reviews 01 in-session: `go build ./...`, `go vet ./...`,
   `gofmt -l .`, `go test ./reservoir/ -race -count=10`. Grep each exported
   symbol for a production caller.
2. If gaps: re-dispatch 01 with findings (max 3 cycles).
3. Commit 01's files. Status -> REVIEWED.
4. **Read** `02-fly-format.md` and dispatch via `delegate_task`:
   - Goal: "Implement all tasks from 02-fly-format.md"
   - Context: full leaf text + Contract 1 + Contract 2 +
     `../SHARED-CONVENTIONS.md` §Determinism Rules §License Rules INLINED +
     the exact `csr` field names written by 01 (inline the struct).
   - Include: the do-not-commit and no-read_file lines above.

### Phase 3: Review 02 and Integrate

1. Review 02 in-session (same commands).
2. Round-trip test: write a fixture `.fly`, `Load` it, check every header
   field and the matrix, and confirm byte-identical reload.
3. Normalize `gofmt -w .`; check line-number corruption.
4. Commit 02. Mark both leaves REVIEWED/COMPLETE. Report COMPLETE to
   `../master.md`.

## Review Checklist

- [ ] `csr` and `spmv` implemented with the exact field names used by 02
- [ ] `Load` is fail-closed (missing/corrupt file -> error, never zero value)
- [ ] `Classify` returns `Abstained=true` below the per-class threshold
- [ ] int8 quantization round-trips within the documented tolerance
- [ ] Deterministic: reload produces an identical matrix; `-count=10` clean
- [ ] No CGO; `gofmt -l` empty; `go vet` clean
- [ ] Every exported symbol has a production caller (grep)
- [ ] No debug artifacts, no TODOs, no placeholder values
- [ ] No line-number corruption

Output: APPROVED, or specific gaps with file + line references.

## Coding Conventions

See `../SHARED-CONVENTIONS.md` §Coding Conventions §Determinism Rules.

## Completion Tracking Table

| Child | Status | Iterations | Review Notes |
|-------|--------|------------|-------------|
| 01-sparse-core | PENDING | 0 | |
| 02-fly-format | PENDING | 0 | |

Status values: PENDING | IN_PROGRESS | IMPLEMENTED | REVIEWED | COMPLETE | BLOCKED

## Integration Test Plan

```bash
go test ./reservoir/... ./internal/flyformat/... -race -count=10
# Round-trip: a fixture written by the test helper must reload identically.
go test ./reservoir/ -run TestFlyRoundTrip -v
```

Expected: all pass; round-trip is byte-stable; no `-race` warnings.

## Open Questions

None specific to this branch. See `../OPEN-QUESTIONS.md` for forest-level
decisions (notably Q1, the consumer interface, which fixes the public API).

## Notes

- Keep N small in fixtures (e.g. 8 neurons) so tests are fast and the checked-in
  `.fly` stays a few hundred bytes.
- The reservoir matrix is fixed at load. Do not add any training path here;
  training is offline (`04-training.md`).
