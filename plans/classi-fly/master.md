# classi-fly - Implementation Orchestrator (root)

> **For the executing agent:** You are the orchestrator for this tree node.
> Your job: (1) dispatch implementation agents, (2) review their work,
> (3) re-dispatch if incomplete, (4) track completion.
> Do NOT implement code yourself. All implementation happens in leaf agents.

## Meta

- **Role:** Root
- **Parent:** none
- **Children:** 1 branch (01-reservoir-core) + 7 top-level leaves
- **Scope:** Build classi-fly - a standalone, reusable Go reservoir classifier
  driven by a fly connectome - and verify it as a second-opinion judge over a
  text-embedding intent classifier. No dependency on meept or any consumer.

## Goal

classi-fly is a self-contained Go library and CLI that classifies a fixed-width
embedding vector into a small set of labels using a frozen recurrent
"reservoir". The reservoir is derived from a fruit-fly connectome (or, in the
license-free mode, generated from a seed). Only a small linear readout is
trained.

The deliverable has no knowledge of meept. Any program that can produce an
embedding vector - meept, another agent, or any service - consumes classi-fly
through one of two identical faces: an in-process Go package, or a tiny HTTP
sidecar.

The research that motivates this tree is `RESEARCH.md` at the repo root. This
tree implements and measures what that document proposes.

## Architecture

Layered, bottom-up:

1. A pure-Go sparse core (CSR matrix, SpMV, reservoir forward pass). No CGO.
2. A versioned binary artifact (`.fly`, zstd-compressed) that carries the
   reservoir, the input projection, the readout, and licensing/attribution
   metadata. Quantized to int8 to stay small.
3. Offline tooling (ingest + train) that produces the artifact from a
   connectome source or a seed, and a training set of (embedding, label) pairs.
4. An evaluation harness that measures the classifier against baselines using
   the same precision-first, route-count-disclosed protocol the meept campaign
   already agreed to.
5. A CLI, and two thin consumer faces: Go import, and HTTP sidecar.

Data flow: connectome or seed -> build adjacency -> `.fly` artifact ->
`Load()` -> `Classify(embedding)` -> `{class, confidence, margin, abstained}`.

## Interface Contracts

### Contract 1: `.fly` artifact format v1

Binary, then zstd-compressed as the outer container. All integers
little-endian.

```
magic          [8]byte   "FLYRES01"
header_len     uint32    length of the JSON header that follows
header         JSON      (see fields below)
-- payload, offsets relative to end of header --
indptr         uint32[N+1]
indices        uint32[E]
weights        int8[E]        dequant: float64(w) * header.weight_scale
input_mode     uint8         0 = seed, 1 = matrix
  if seed:    seed uint64
  if matrix:  in_w int8[D*S], in_row_len uint32 (S), in_scale float64
readout        int8[N*K]
readout_scale  float64
bias           float32[K]
threshold      float32[K]     per-class route floor
```

Header JSON fields (frozen names):

```
{
  "format": "fly-reservoir",
  "version": 1,
  "name": "larval-v1",
  "neurons": 3016,
  "edges": 65000,
  "embed_dim": 1024,
  "steps": 8,
  "classes": ["coding","debugging","analysis","search","chat","platform",
              "git","scheduling","planning","review","reporting","recall",
              "quickplan"],
  "weight_scale": 0.001,
  "source": "larval-connectome",
  "license": "CC-BY-4.0",
  "attribution": "Winding et al. 2023, Science 379:eadd9330",
  "created_utc": "2026-09-11T00:00:00Z"
}
```

Owner: `01-reservoir-core/02-fly-format.md`.
Consumers: `02-connectome-ingest.md`, `03-synthetic-reservoir.md`,
`04-training.md`, `05-eval-harness.md`, `06-cli-package.md`,
`07-integration.md`, `08-verification.md`.

### Contract 2: Go package API (`github.com/caimlas/classi-fly/reservoir`)

```go
package reservoir

type Result struct {
    Class      string
    Confidence float64 // winning lane probability, 0..1
    Margin     float64 // top1 - top2 probability
    Abstained  bool    // true when the caller should fall through
}

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

type Reservoir struct { /* unexported */ }

func Load(path string) (*Reservoir, error)
func (r *Reservoir) Classify(embedding []float32) (Result, error)
func (r *Reservoir) Info() Info
```

`Load` is fail-closed: a missing or corrupt file returns an error, never a
zero-valued reservoir that silently misclassifies. `Classify` returns
`Abstained=true` (with zero confidence) when the winning lane is below its
threshold - the caller then falls through. Dimension mismatch is an error.

Owner: `01-reservoir-core/01-sparse-core.md` (types) and `03-forward-pass`
folded into it; `Classify` wrapper owner `05-eval-harness.md` consumer side.
Consumers: `06-cli-package.md`, `07-integration.md`, `08-verification.md`.

### Contract 3: Offline tooling output (producer side)

`tools/` scripts (Python 3.12, offline only) emit:
- an adjacency export: `{name, neurons, edges, indptr, indices, weights, source, license, attribution}`
- a readout export: `{classes[], W[N][K], bias[K], threshold[K], weight_scale, readout_scale}`
- and the assembled `.fly` file (via `06-cli-package.md`'s `classi-fly pack`).

Owner: `02-connectome-ingest.md`, `03-synthetic-reservoir.md`,
`04-training.md`. Consumer: `06-cli-package.md`.

### Contract 4: Sidecar HTTP face (optional, identical semantics)

```
POST /classify   {"embedding": [float,...], "trace": false}
              -> {"class": "...", "confidence": 0.91, "margin": 0.22,
                  "abstained": false}
GET  /info    -> the header JSON of the loaded artifact
GET  /healthz -> 200 "ok" once the artifact is loaded
```

Owner: `07-integration.md`. This face is a thin wrapper; it must not
re-implement the core.

## Child Index

| # | Document | Type | Dependencies | Est. Context | Concurrency |
|---|----------|------|-------------|-------------|-------------|
| 01 | 01-reservoir-core/orchestrator.md | branch | none | 5K (branch) | A |
| 02 | 02-connectome-ingest.md | leaf | none (needs Contract 1 names) | 45K | A |
| 03 | 03-synthetic-reservoir.md | leaf | none | 40K | A |
| 04 | 04-training.md | leaf | Contract 1, Contract 3 | 55K | A |
| 05 | 05-eval-harness.md | leaf | Contract 2, Contract 3 | 85K | B |
| 06 | 06-cli-package.md | leaf | Contract 1, Contract 2 | 70K | B |
| 07 | 07-integration.md | leaf | Contract 2, Contract 4 | 60K | B |
| 08 | 08-verification.md | leaf | all | 70K | C |

**Concurrency groups:** documents in the same letter have no
inter-dependencies and may be dispatched together (max 3 per batch,
`delegation.max_concurrent_children`).

Group A (parallel): 01, 02, 03, 04. Group B (after A): 05, 06, 07.
Group C (after B): 08.

Shared conventions and the open-questions ledger are siblings of this file:
`SHARED-CONVENTIONS.md`, `OPEN-QUESTIONS.md`. Every document references them.

## Dispatch Protocol

For each concurrency group, in dependency order.

### Phase 1: Dispatch Concurrency Group A

Dispatch these simultaneously (no inter-dependencies):

1. **Read** `01-reservoir-core/orchestrator.md` and dispatch it as a branch
   (it dispatches its own two leaves).
   - This is the de-risk milestone: CSR + SpMV + forward pass + `.fly` format.
2. **Read** `02-connectome-ingest.md` and dispatch via `delegate_task`:
   - Goal: "Implement all tasks from 02-connectome-ingest.md"
   - Context: full leaf text + Contract 1 (exact `.fly` header field names) +
     `SHARED-CONVENTIONS.md` §Project Layout §Coding Conventions §License Rules
     INLINED + the exact larval/hemibrain ingest requirements from RESEARCH.md §4.
   - Include: "Do NOT commit. Do NOT run git add. Write code, run tests,
     report results only."
   - Include: "Do NOT use read_file on existing source files - explore with
     search_files or terminal cat. Never feed read output into write_file."
3. **Read** `03-synthetic-reservoir.md` and dispatch via `delegate_task`:
   - Goal: "Implement all tasks from 03-synthetic-reservoir.md"
   - Context: full leaf text + Contract 1 + `SHARED-CONVENTIONS.md`
     §Coding Conventions §Determinism Rules INLINED.
   - Include: the do-not-commit and no-read_file lines above.
4. **Read** `04-training.md` and dispatch via `delegate_task`:
   - Goal: "Implement all tasks from 04-training.md"
   - Context: full leaf text + Contract 1 + Contract 3 + `SHARED-CONVENTIONS.md`
     INLINED. Dependency: may run before 01 finishes because the Python trainer
     consumes a `.fly` file lazily; if the `.fly` loader is not yet built,
     the leaf must be verified against a fixture `.fly` generated by its own
     test helper.
   - Include: the do-not-commit and no-read_file lines above.

### Phase 2: Review and Commit Each Child

After each implementation agent returns, the orchestrator reviews in-session
(the main model, NOT a delegated subagent - delegate_task children inherit
`delegation.model`):

1. Read the changed files from the implementer's file list.
2. Check against the leaf spec + Contract 1/2/3/4 + `SHARED-CONVENTIONS.md`.
3. For Go leaves: `go build ./...`, `go vet ./...`, `gofmt -l .`, run the
   leaf's tests. For each exported function the leaf adds, grep for a
   production caller (no isolated islands).
4. If gaps: re-dispatch with the specific findings, max 3 cycles, then
   escalate.
5. If it passes: commit only the leaf's explicit paths, then set the tracking
   table status to REVIEWED.

### Phase 3: Integration Review

After all group-B children reach REVIEWED:

1. Run `go test ./... -race -count=10` on the whole module.
2. Verify each Contract exactly (header field names, function signatures,
   HTTP shapes) - read the artifacts, do not trust summaries.
3. Cross-check: a `.fly` produced by 02/03 loads in 06's CLI and classifies
   in 05's harness.
4. Normalize formatting: `gofmt -w .`; `ruff format tools/` if present.
5. Verify no line-number corruption:
   `grep -rcE '^\s+[0-9]+\|' --include='*.go' --include='*.py' --include='*.md' .`
   must return zero hits.
6. Commit integration changes; mark children COMPLETE; report COMPLETE.

## Review Checklist

The orchestrator (main model) verifies each child in-session:

- [ ] All tasks from the leaf document are implemented
- [ ] Interface contracts from this orchestrator are satisfied exactly
- [ ] All specified files created/modified at exact paths
- [ ] Tests written and passing; `go vet` clean; `gofmt -l` empty
- [ ] Code follows `SHARED-CONVENTIONS.md` (naming, errors, determinism)
- [ ] No scope creep; no meept import anywhere in the module
- [ ] No debug artifacts: no stray prints, TODOs, placeholder values
- [ ] No line-number corruption (no `     N|` prefixes)
- [ ] For every exported function: a production caller exists (grep)
- [ ] License rules honored: no FlyWire-derived data in a shippable `.fly`

Output: APPROVED, or a list of specific gaps with file + line references.

## Coding Conventions

See `SHARED-CONVENTIONS.md` §Coding Conventions. Summary: Go 1.22+, pure Go
(no CGO), gofmt, `errors.Is/As` and `%w` wrapping, fail-closed loaders,
deterministic output (sorted map keys, epsilon float compares), stdlib
`testing` (testify allowed if already vendored), no debug artifacts.

## Completion Tracking Table

| Child | Status | Iterations | Review Notes |
|-------|--------|------------|-------------|
| 01-reservoir-core | IN_PROGRESS | 0 | 01-sparse-core dispatched (wave A) |
| 02-connectome-ingest | IN_PROGRESS | 0 | dispatched (wave A) |
| 03-synthetic-reservoir | IN_PROGRESS | 0 | dispatched (wave A) |
| 04-training | IN_PROGRESS | 0 | dispatched (wave A) |
| 05-eval-harness | IN_PROGRESS | 0 | dispatched early (file-disjoint, own fixtures) |
| 06-cli-package | PENDING | 0 | |
| 07-integration | PENDING | 0 | |
| 08-verification | PENDING | 0 | |

Status values: PENDING | IN_PROGRESS | IMPLEMENTED | REVIEWED | COMPLETE | BLOCKED

## Integration Test Plan

Run from the repo root after all group-B leaves are REVIEWED:

```bash
go build ./... && go vet ./... && gofmt -l .
go test ./... -race -count=10
# End-to-end: build a .fly from the synthetic reservoir, classify a fixture
./bin/classi-fly build --synthetic --seed 7 --classes testdata/classes.json \
    --out /tmp/test.fly
./bin/classi-fly info /tmp/test.fly
./bin/classi-fly classify /tmp/test.fly --embedding-file testdata/embed_example.json
# Harness: judge mode against baselines on the fixture corpus
python3 tools/eval/run_eval.py --fly /tmp/test.fly --corpus testdata/fixture_corpus.json5
```

Expected: build/vet/fmt clean; tests pass under `-race -count=10`; `info`
prints the header JSON; `classify` returns a class + confidence + margin;
the harness prints P, C, E2E, OOD-R and the route count.

## Open Questions

See `OPEN-QUESTIONS.md` (sibling to this file). Three items need an owner
decision before execution: Q1 consumer interface, Q2 Python tooling, Q4
synthetic-vs-connectome default. The rest have safe defaults.

## Notes

- The whole point of this tree is reuse. Nothing here may import or mention
  meept except `07-integration.md`, which is an example consumer.
- FlyWire is CC BY-NC 4.0. It must never end up in a shipped `.fly`. See
  `SHARED-CONVENTIONS.md` §License Rules and `08-verification.md`.
- Keep the de-risk path first: a synthetic reservoir that classifies a trivial
  fixture proves the plumbing before any connectome is touched.
