# classi-fly - Shared Conventions

Sibling to `master.md`. Every document in this forest references this file
instead of restating conventions. When a contract changes, edit it here once.

## Project Layout

Standalone repository. Module path `github.com/caimlas/classi-fly`. No import
of any consumer project (meept or otherwise).

```
classi-fly/
  go.mod                          module github.com/caimlas/classi-fly
  cmd/classi-fly/main.go          CLI entry point
  reservoir/                      runtime library (pure Go, no CGO)
    sparse.go                     CSR type + SpMV
    reservoir.go                  Reservoir struct + Forward
    fly.go                        .fly load/save
    classify.go                   Classify + Result + Info
  internal/flyformat/             low-level encode/decode of the .fly container
  tools/                          offline tooling (Python 3.12, NOT shipped)
    ingest/                       connectome -> adjacency export
    train/                        readout trainer -> readout export
    eval/                         evaluation harness
  testdata/                       fixtures (tiny, checked in)
  plans/                          this forest
  RESEARCH.md                     motivation + source inventory
```

## Language Split

- **Runtime (shipped):** Go only. The library and CLI must be pure Go, no CGO,
  so a consumer can `go get` it anywhere.
- **Offline tooling:** Python 3.12 is permitted for ingest, training, and the
  eval harness (mirrors the existing convention in sibling projects, e.g.
  `build_tfidf_veto.py`). These scripts are not part of the shipped binary.
- OPEN-QUESTION Q3 asks whether the user wants Go-only end-to-end; until
  answered, assume Python tooling is acceptable.

## Frozen Contracts (canonical copy lives in master.md §Interface Contracts)

- **Contract 1** - `.fly` artifact format v1 + exact header JSON field names.
- **Contract 2** - Go package API: `Load`, `Classify`, `Info`, `Result`, `Info`.
- **Contract 3** - offline tooling export shapes (adjacency, readout, pack).
- **Contract 4** - sidecar HTTP face: `/classify`, `/info`, `/healthz`.

Never rename a header field or a function signature in a leaf. If a contract
must change, it is a `master.md` edit (and a note in OPEN-QUESTIONS.md), never
silent leaf drift.

## Coding Conventions

- **Language:** Go 1.22+. No CGO. `gofmt` clean before reporting.
- **Naming:** exported = PascalCase, unexported = camelCase. Package `reservoir`.
- **Imports:** grouped stdlib / third-party / local; no unused imports.
- **Errors:** wrap with `%w`; return early; never `panic` in the library
  (panics allowed only in `main` for startup misuse). Loaders are fail-closed.
- **Testing:** stdlib `testing`; table-driven; `_test.go` beside the source.
  testify is allowed only if already in `go.mod`.
- **No debug artifacts:** no stray `fmt.Println`, no `TODO`, no placeholder
  values, no commented-out code in reported files.
- **Docs:** exported symbols carry a doc comment starting with the symbol name.

## Determinism Rules

The classifier must return byte-identical results for identical input.

- Never iterate a map to accumulate floats. Extract keys, sort, then iterate.
- Never compare floats with `!=` in a sort comparator. Use an epsilon
  (`d > 1e-9 || d < -1e-9`).
- Tests that rank or sort by a float score must run with
  `go test -count=10 -race`.
- The `.fly` loader must produce the same in-memory matrix every time.

## License Rules

- **FlyWire (FAFB, BANC, MCNS, MANC):** CC BY-NC 4.0 - NON-COMMERCIAL.
  Benchmark and research only. Never export a `.fly` derived from FlyWire
  unless the header `license` is `CC-BY-NC-4.0` AND the caller passes an
  explicit `--allow-noncommercial` flag. Shippable artifacts must refuse.
- **Hemibrain (Janelia FlyEM):** CC-BY - shippable with attribution.
- **Larval connectome (Winding 2023, Science):** CC BY 4.0 - shippable with
  attribution.
- **Synthetic (seed-generated):** no third-party license - the default
  license-clean artifact.
- Every `.fly` header carries `license` and `attribution` strings, and the
  repo ships a `NOTICE` file listing each source. `08-verification.md` audits
  this.

## Test Strategy

- Runtime: Go unit tests, `-race -count=10` for anything float-ranked.
- Fixtures: a tiny synthetic `.fly` (a few neurons) checked into `testdata/`
  so tests never depend on a real connectome download.
- Eval: Python harness reads a `.fly` and a corpus, prints P / C / E2E /
  OOD-R / route-count. No network except the embedding endpoint, which is
  passed in as a base URL.
- Every leaf reports: files touched, exact commands run, observed output, and
  any deviation from spec.

## Commit Policy

Only orchestrators commit. Implementation agents write code, run tests, and
report - they never run `git add` or `git commit`. The orchestrator stages the
leaf's explicit file paths and commits after review passes.
