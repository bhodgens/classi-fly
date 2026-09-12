# classi-fly

A fly-reservoir classifier: a tiny, frozen, deterministic classifier that
mimics the insect mushroom body — a sparse fixed "reservoir" transforms an
embedding through a few tanh recurrence steps, and a small trained linear
readout picks the class (or abstains). The whole classifier ships as a single
compressed `.fly` artifact (well under 1 MB) loadable by a static Go binary,
so consumers need no Python, no model server, and no heavyweight ML runtime.

## Status

Implementation tree COMPLETE (2026-09-11). All 8 plan leaves implemented,
reviewed, and committed. Verification results live in
[docs/VERIFICATION.md](docs/VERIFICATION.md); integration guidance for
consumers in [docs/INTEGRATION.md](docs/INTEGRATION.md).

| Gate | Result |
|---|---|
| `go test ./... -race -count=10` | PASS (4 packages) |
| Python test suite | PASS (94 tests: eval/ingest/train/verify) |
| `go vet` + `gofmt -l` | clean |
| Quantization drift (int8 vs float) | 64 probes, 0 disagreements |
| License audit | PASS; FlyWire fenced as benchmark-only |
| E2E metric recomputation | matches to 1e-9; tamper detection works |

**Honest limits** (from docs/VERIFICATION.md): the classifier *machinery* is
verified; the *accuracy claims* are not. Experiments E1-E5 (does the reservoir
beat baselines on real embeddings?) are UNVALIDATED - no real-corpus run has
happened yet. The `classi-fly build` convenience path is UNVALIDATED
(state saturation on scaffold pairs); use `pack` with a properly trained
readout instead.

## What is in this repository

- `reservoir/` — the pure-Go runtime library (load `.fly`, classify).
- `cmd/classi-fly/` — the `classi-fly` CLI (`pack`, `info`, `classify`,
  `build`, `serve`).
- `tools/` — offline Python tooling (ingest, train, eval, verify). **Not
  shipped**; only `pack`-ready exports and the CLI are part of the artifact
  path. Only the `build` subcommand shells out to it.
- `examples/host_judge.go` — the second-opinion "judge" integration pattern.
- `plans/classi-fly/` — the hierarchical implementation plan tree and the
  frozen interface contracts (authoritative for the `.fly` format).
- `RESEARCH.md` — the study that motivated this project: fly connectomes as
  reservoirs, dataset licensing, and the experiment plan (E0-E5).

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
  exits `1`. UNVALIDATED - see Status.
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
python3 tools/verify/ridge_orientation_repro.py   # documents Finding 1 (fixed); exits 1 post-fix
```

- `tools/ingest/` — connectome -> CSR adjacency. `larval.py` (CC BY 4.0),
  `hemibrain.py` (CC-BY, `--live` gated), `flywire.py` (CC BY-NC 4.0,
  refuses without `--allow-noncommercial`), `synthetic.py` (license-free
  default). Fixture-only tests; no network.
- `tools/train/` — ridge readout + per-class quantile calibration.
  Precision-first: a wrong direct route is worse than an abstention.
- `tools/eval/` — head and judge modes vs baselines (centroid-margin,
  kNN-unanimity, TF-IDF logistic). `E2E = (correct + 0.868*abstained)/total`
  is a deterministic named constant (`CHAIN_BASELINE`); route counts are
  always disclosed.
- `tools/verify/` — independent recomputation, quant-drift twins, license
  audit. Never trusts the harness's own summaries.

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

1. **E1-E5 experiments** (RESEARCH.md §7): run the judge-mode evaluation on a
   real corpus against the baselines. Needs the ~35 MB larval connectome
   download (owner-gated) and a local embedding server.
2. Validate or fix the `build` scaffold path (Finding 2).
3. Consumer wiring: the first host to use classi-fly as a second-opinion
   judge in production.
