# classi-fly

A fly-reservoir classifier: a tiny, frozen, deterministic classifier that
mimics the insect mushroom body — a sparse fixed "reservoir" transforms an
embedding through a few tanh recurrence steps, and a small trained linear
readout picks the class (or abstains). The whole classifier ships as a single
compressed `.fly` artifact (well under 1 MB) loadable by a static Go binary,
so consumers need no Python, no model server, and no heavyweight ML runtime.

## Status

Implementation tree COMPLETE (2026-09-11). All 8 plan leaves implemented,
reviewed, and committed. Research arc COMPLETE (2026-09-15): classification,
control, robustness, OOD detection, and temporal tasks all measured. Full
results: [docs/RESULTS.md](docs/RESULTS.md). Verification:
[docs/VERIFICATION.md](docs/VERIFICATION.md). Integration:
[docs/INTEGRATION.md](docs/INTEGRATION.md).

| Gate | Result |
|---|---|
| `go test ./... -race -count=10` | PASS (4 packages) |
| Python test suite | PASS (134 tests: eval/ingest/train/verify/control) |
| `go vet` + `gofmt -l` | clean |
| Quantization drift (int8 vs float) | 64 probes, 0 disagreements |
| License audit | PASS; FlyWire fenced as benchmark-only |
| E2E metric recomputation | matches to 1e-9; tamper detection works |
| `classi-fly build` end-to-end | PASS (4 tests; was UNVALIDATED, fixed 2026-09-12) |
| Safety-first gate (P 0.958, OOD 1.000, noise fail-safe) | PASS, 5 tests |
| Cheap 512-neuron artifact (12 KB) | 100% / 18.4 blink3, 100% / 39.9 blink8 |
| Classification accuracy (clean) | **PARITY** with a linear probe (0.726 vs 0.718) — no advantage |
| Robustness to missing input dims | **+18.8 pt at 75% truncation (p<1e-12)** — real, replicates |
| Connectome vs synthetic (all tasks) | **PARITY** — the wiring is not the active ingredient |
| Session drift on real meept data | **NOT MEASURABLE** at current data volume |
| Embedding-stream state-novelty | **WORSE** than per-item cosine novelty |

## Honest summary

The machinery is production-ready and well-tested. The fly brain does not
improve classification accuracy — on clean input it is at parity with a linear
probe, and the connectome shows no advantage over a synthetic matrix in any
task tested (classification, control, robustness, OOD, rhythm, anomaly
detection). The one measured advantage is **graceful degradation when input
dimensions go missing**, which a license-free synthetic matrix reproduces.
The substrate's real value is **temporal integration under partial
observability**: a recurrent reservoir holds 100% success on a control task
where a memoryless map collapses to 50%, and the larval connectome detected
period-change anomalies that a synthetic matrix missed entirely (152/200 vs
0/200) — though that advantage did not generalize to other anomaly types.

Measured cost vs meept's current Stage-0: 1.66 MB on disk (smaller) but
30.6 MB resident and 1.90 ms per call (~3x); the input projection, not the
recurrence, is the cost driver.

## What is in this repository

- `reservoir/` — the pure-Go runtime library (load `.fly`, classify).
- `cmd/classi-fly/` — the `classi-fly` CLI (`pack`, `info`, `classify`,
  `build`, `serve`).
- `tools/` — offline Python tooling (ingest, train, eval, verify, control).
  **Not shipped**; only `pack`-ready exports and the CLI are part of the
  artifact path. Only the `build` subcommand shells out to it.
- `tools/control/` — lane 1/3 control experiments (phototaxis harness,
  rhythm task, anomaly task, size co-sweep, hardening sweeps).
- `examples/host_judge.go` — the second-opinion "judge" integration pattern.
- `plans/classi-fly/` — the hierarchical implementation plan tree and the
  frozen interface contracts (authoritative for the `.fly` format).
- `RESEARCH.md` — the study that motivated this project: fly connectomes as
  reservoirs, dataset licensing, and the experiment plan (E0-E5).

## Documentation

| document | what it covers |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | **complete test and results ledger** — every experiment, every verdict |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | classification results + correction |
| [docs/AXES-2026-09-12.md](docs/AXES-2026-09-12.md) | four-lever axis results (spectral radius, projection, MB, robustness) |
| [docs/LANE3-AND-SIZE.md](docs/LANE3-AND-SIZE.md) | temporal task + size co-sweep results |
| [docs/MEEPT-TESTS.md](docs/MEEPT-TESTS.md) | the three meept mapping tests (context, scaling, degraded) |
| [docs/MEEPT-INTEGRATION.md](docs/MEEPT-INTEGRATION.md) | how it would wire into meept, with measured resource comparison |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | findings ledger with evidence |
| [docs/ANOMALY-PLAN.md](docs/ANOMALY-PLAN.md) | anomaly detection implementation plan |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | consumer integration guide (Go import + sidecar) |
| [RESEARCH.md](RESEARCH.md) | the study: fly connectomes as reservoirs, licensing, experiment tiers |

## The `.fly` format

A `.fly` artifact is `FLYRES01` magic + a length-prefixed JSON header +
zstd-compressed little-endian binary arrays: the sparse reservoir graph
(CSR), the input projection, and the int8-quantized readout. The header
carries the frozen field names `format`, `version`, `name`, `neurons`,
`edges`, `embed_dim`, `steps`, `classes`, `weight_scale`, `source`,
`license`, `attribution`, `created_utc`, plus the `decay` extension the
runtime recurrence reads. The authoritative definition is
`plans/classi-fly/master.md` (Contract 1), implemented by the `reservoir`
package (`reservoir.Load`, `reservoir.WriteFile`).

## Go API

```go
import "github.com/caimlas/classi-fly/reservoir"

r, err := reservoir.Load("model.fly")
if err != nil {
    return err // fail-closed: missing or corrupt artifacts never misclassify silently
}
result, err := r.Classify(embedding) // embedding: []float32 of header.embed_dim
if err != nil {
    return err
}
info := r.Info() // Name, Neurons, Edges, EmbedDim, Steps, Classes, License, Attribution

fmt.Println(result.Class, result.Confidence, result.Margin, result.Abstained)
```

`Result.Abstained` is true when the winning lane is below its calibrated
per-class route threshold; the caller should fall through to another system
in that case.

## CLI

```
classi-fly pack --adjacency a.json --readout r.json --pack p.json --out model.fly [--decay 0.8] [--allow-noncommercial]
classi-fly info model.fly
classi-fly classify model.fly --embedding 0.1,0.2,0.3
classi-fly classify model.fly --embedding-file vec.json
classi-fly build --synthetic --classes classes.json --pairs pairs.jsonl --out model.fly
classi-fly serve --fly model.fly --addr 127.0.0.1:8091
```

- `pack` assembles a `.fly` from two offline-tooling exports plus a pack
  block. Exit codes: `0` ok, `1` usage error, `2` load/validation error,
  `3` non-commercial source refused without `--allow-noncommercial`.
- `info` prints the header JSON **verbatim** (bytes identical to what is
  stored in the file).
- `classify` prints `{"class":...,"confidence":...,"margin":...,"abstained":...}`.
- `build` is a convenience that runs the synthetic generator and the trainer
  (both Python) in a temp dir and packs the result. It requires `python3` on
  PATH and a checkout containing `tools/`; the shipping path is `pack` + a
  pre-built export pair. If Python is missing it prints a clear error and
  exits `1`.
- `serve` is the HTTP sidecar: `POST /classify`, `GET /info`,
  `GET /healthz`. Loopback-only by design; do not expose it publicly.

### Pack inputs

The two offline exports follow Contract 3
(`plans/classi-fly/master.md`):

- **adjacency export** (`tools/ingest/*.py`):
  `{name, neurons, edges, indptr, indices, weights, source, license, attribution}`
- **readout export** (`tools/train/train_readout.py`):
  `{classes[], W[N][K], bias[K], threshold[K], weight_scale, readout_scale}`

Neither export carries the artifact's input contract, so `pack` also reads a
small **pack block** JSON (extension field `--pack`; extension, not part of
the frozen contract):

```json
{
  "embed_dim": 8,
  "steps": 8,
  "input_mode": "matrix",
  "in_w": [ "...embed_dim*neurons int8 values..." ],
  "in_scale": 0.1
}
```

`input_mode` is `"matrix"` (payload carries `in_w` + `in_scale`) or `"seed"`
(payload carries `seed`; the runtime expands it deterministically). `decay`
is written into the header (`--decay`, default `0.8`) so the runtime
recurrence and the offline trainer agree on the leak term — the trainer reads
it from the header with default `0.8`, so changing it here after training
changes runtime behavior; keep them in sync.

## Offline tooling

Python 3.12+ (runs on 3.14; stdlib + numpy only, no installs required).

```
python3 -m pytest tools -q                     # whole offline suite
python3 tools/ingest/synthetic.py --seed 7 --neurons 2048 --out adjacency.json
python3 tools/train/train_readout.py --fly provisional.fly --pairs pairs.jsonl --out readout.json
python3 tools/eval/run_eval.py --fly model.fly --corpus corpus.json5 --embed-url http://127.0.0.1:8090/v1 --mode all --out results.json
python3 tools/eval/embed_health_server.py --ref data/embed_health_ref.npz --port 8092
```

- `tools/ingest/` — connectome -> CSR adjacency. `larval.py` (CC BY 4.0),
  `hemibrain.py` (CC-BY, `--live` gated), `flywire.py` (CC BY-NC 4.0,
  refuses without `--allow-noncommercial`), `synthetic.py` (license-free
  default). Fixture-only tests; no network.
- `tools/train/` — ridge readout + per-class quantile calibration.
  Precision-first: a wrong direct route is worse than an abstention.
  Ridge penalty auto-selected by deterministic inner CV (or `--ridge-lambda`).
- `tools/eval/` — head and judge modes vs baselines (centroid-margin,
  kNN-unanimity, TF-IDF logistic). `E2E = (correct + 0.868*abstained)/total`
  is a deterministic named constant (`CHAIN_BASELINE`); route counts are
  always disclosed. Also: safety gate, drift prototype, stream monitor,
  frontier with OOD as a constraint.
- `tools/verify/` — independent recomputation, quant-drift twins, license
  audit. Never trusts the harness's own summaries.
- `tools/control/` — the phototaxis control harness (lane 1), rhythm task,
  anomaly task, size co-sweep, lane-1 hardening sweeps.

## Connecting an embedding source

The classifier consumes embedding vectors; it does not embed text itself.
Point it at any OpenAI-compatible embedding endpoint (e.g. a local
Qwen3-Embedding server) via the eval harness `--embed-url`, or compute
embeddings in-process and call `Classify` directly. The `embed_dim` in the
pack block must match the embedding model's output dimension.

## License rules

- **FlyWire (FAFB, BANC, MCNS, MANC)** — CC BY-NC 4.0: **non-commercial**.
  `pack` refuses such sources (exit 3) unless `--allow-noncommercial` is
  passed. Use for benchmark/research only.
- **Hemibrain (Janelia FlyEM)** — CC-BY 4.0: shippable with attribution.
- **Larval connectome (Winding 2023, Science)** — CC BY 4.0: shippable with
  attribution.
- **Synthetic** — license-free (`license: "none"`): the default shippable
  artifact, no third-party data.

Every `.fly` header carries `license` and `attribution`; `NOTICE` lists the
sources.

## Build

Requires Go 1.22+. No CGO.

```
make build     # bin/classi-fly
make test      # go test ./... -race -count=10
make lint      # go vet ./... + gofmt check
make fmt       # gofmt -w
make release   # stripped, trimpath binary + size report
```

Directly: `go build -trimpath -ldflags "-s -w" -o bin/classi-fly ./cmd/classi-fly`

The offline Python tooling is invoked by `classi-fly build` only; it is not
needed to build the binary itself.

## Roadmap

1. ~~Classification~~ **CLOSED (parity)** — docs/EXPERIMENTS.md. The
   reservoir matches a linear probe; do not wire it into the chain.
2. ~~Robustness~~ **CONFIRMED** — docs/AXES-2026-09-12.md. +18.8 pt at 75%
   truncation, independently reproduced.
3. ~~Temporal computation~~ **OPENED** — docs/LANE3-AND-SIZE.md and
   docs/DIRECTIONS.md section 7. The reservoir wins rhythm (200/200) and
   anomaly (152/200, 5-step latency) tasks. The connectome beat synthetic on
   anomaly detection (152 vs 0) — the first connectome advantage. Replication
   with three new anomaly types confirmed the result is seed-robust but showed
   it does not generalize to non-period anomalies. The lane-1 hardening sweep
   identified a cliff at blink12 and found that 512 neurons fail at default
   scaling but 512 at ρ=0.5/lognormal matches the real connectome.
4. ~~Cheap artifact~~ **DONE** — 512 neurons / ρ=0.5 / lognormal matches the
   real connectome (100% / 18.2 steps).
5. ~~Embedder health check~~ **DONE** — cosine distance to a reference set,
   implemented in Go (internal/agent/embed_health.go), default-enabled,
   24 tests, wired into the prefilter path. Closes meept issue #42.
6. Consumer wiring: the first host to use classi-fly as a second-opinion
   judge in production — the machinery is ready, but the accuracy case for
   doing so is not made.
7. Outcome-loop data accumulation — meept issue #40. The re-route detector
   is alive and working (32 resolved outcomes, 4 corrections) but needs more
   multi-turn sessions before session-aware classification is trainable.
