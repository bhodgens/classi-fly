# Integration Surface (Go library + sidecar) - Implementation Leaf

> **For the implementing agent:** You are the implementer for this leaf.
> Implement ALL tasks below using TDD. Do NOT commit - the orchestrator
> handles all git operations after review. Do NOT use read_file on existing
> source files - explore with search_files or terminal cat. After writing a
> file, do NOT read it back to verify - write once and stop. After completing,
> report what you built, what files you touched, and any deviations.

## Meta

- **Parent:** ../master.md
- **Scope:** The two consumer faces that make classi-fly reusable: the in-process
  Go package (already exposed by the core) and an optional HTTP sidecar. Plus a
  worked example showing how a host app plugs it in. This is the only leaf that
  names a specific consumer.
- **Dependencies:** Contract 2, Contract 4
- **Estimated Context:** 60K
- **Concurrency Group:** B

## Goal

Make classi-fly trivially consumable by anything that has embeddings, with no
knowledge of the consumer. Two faces, one core:
1. In-process: `import "github.com/caimlas/classi-fly/reservoir"` and call
   `Load` + `Classify`. Nothing to build here beyond a usage doc and an example.
2. Sidecar: `classi-fly serve --fly model.fly --addr 127.0.0.1:8091` speaking
   `POST /classify`, `GET /info`, `GET /healthz`.

Plus a worked example (`examples/host_judge.go`) showing the intended "second
opinion" pattern: a host keeps its own head, calls classi-fly, and routes only
on agreement. The meept case is one instance of this pattern, described but not
required.

## Context

The design principle: the reservoir is a fixed, tiny, dependency-free second
opinion. It never replaces the host's routing; it either agrees (route) or
disagrees (fall through). This mirrors the existing "veto" pattern where a
cheap independent checker gates a confident decision.

## Interface Contracts (From Parent)

### What This Leaf Exposes

```
HTTP:
  POST /classify   {"embedding":[float,...],"trace":false}
                -> {"class":"...","confidence":0.9,"margin":0.2,"abstained":false}
  GET  /info    -> header JSON
  GET  /healthz -> 200 "ok" once loaded, 503 before

Go example:
  examples/host_judge.go
  // shows: load reservoir, embed a text, compare with the host head, route on agree
```

Sidecar must wrap the core; it must not re-implement SpMV or the readout.

### What This Leaf Consumes

```
Contract 2: Load, Classify, Info, Result, Info.
Contract 4: the HTTP shapes above.
```

## Tasks

### Task 1: Sidecar server

**Objective:** A minimal stdlib HTTP server over the loaded artifact.

**Files:**
- Create: `cmd/classi-fly/serve.go`
- Test: `cmd/classi-fly/serve_test.go`

**Step 1: Write failing test** - start the server with `httptest`, POST a
valid embedding, assert the four JSON keys; POST a wrong-length embedding,
assert 400; GET `/healthz` returns 503 before load and 200 after; GET `/info`
returns the header JSON.

**Step 2: Run** `go test ./cmd/classi-fly/ -run TestServe -v`
Expected: FAIL.

**Step 3: Implement** with `net/http` only. Load the artifact once at startup;
`/healthz` reflects load state. No third-party router.

**Step 4: Run.** Expected: PASS.

### Task 2: Worked consumer example (judge pattern)

**Objective:** A compiling example showing the intended integration.

**Files:**
- Create: `examples/host_judge.go`
- Test: `examples/host_judge_test.go`

**Step 1: Write failing test** - a fake host head returns label X with high
confidence; the reservoir returns X -> example routes; the reservoir returns Y
-> example abstains. Test the decision function directly.

**Step 2: Run** `go test ./examples/ -run TestJudge -v`
Expected: FAIL.

**Step 3: Implement** `func judgeDecide(hostLabel string, hostConf float64,
res reservoir.Result) (route bool, label string)` embodying: route only when
the host is above its own threshold AND the reservoir agrees AND the reservoir
is not abstained.

**Step 4: Run.** Expected: PASS.

### Task 3: Integration documentation

**Objective:** Document both faces and the license rules for a consumer.

**Files:**
- Create: `docs/INTEGRATION.md`

**Step 1:** Section "In-process": the exact import path, a code snippet, and
the note that a missing `.fly` makes the feature a no-op (fail-closed, the
host keeps working).

**Step 2:** Section "Sidecar": the three endpoints, a curl example, and a
health-check note.

**Step 3:** Section "Licensing": which `.fly` files may ship; how to check via
`classi-fly info`.

**Step 4:** Section "Worked example: a host with an existing head" - the judge
pattern, described generically, with the meept case as an illustrative
instance (the host keeps its own kNN/centroid head and uses the reservoir as a
tie-breaking second opinion). Keep it an example, not a dependency.

## Self-Verification Checklist

- [ ] Sidecar serves all three endpoints; health reflects load state
- [ ] Judge example compiles and its decision function is tested
- [ ] `docs/INTEGRATION.md` documents both faces and licensing
- [ ] The Go package requires no consumer-specific import (grep the module for
  consumer names: only this leaf's doc/example may mention meept)
- [ ] `go build ./...`, `go vet ./...`, `gofmt -l .` clean
- [ ] `go test ./... -race -count=10` passes
- [ ] No deviations (or documented below)

**DO NOT COMMIT.** The orchestrator handles all git operations after review.

**Deviations from spec:** none

## Review Checklist (For Review Agent)

- [ ] Every Task implemented and tested
- [ ] Sidecar wraps the core (grep: no duplicate SpMV/readout logic)
- [ ] `grep -rn "meept" --include='*.go' .` returns hits ONLY in
  `examples/` docs this leaf authored - the library itself stays generic
- [ ] No line-number corruption; no debug artifacts

Output: APPROVED or specific gaps with file + line references.

## Notes

- The sidecar is optional. If the host embeds Go libraries directly, the
  in-process face is enough. Build both because the cost is small.
- Do not add auth to the sidecar; it binds loopback and is a local
  second-opinion service. Document that it must not be exposed publicly.
