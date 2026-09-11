# Integrating classi-fly

Two ways to consume the same classifier core:

1. **In-process** — import the Go package and call `Load` + `Classify`.
2. **Sidecar** — run `classi-fly serve` and speak HTTP to it.

Both faces are stateless per request: send an embedding vector, get
`{class, confidence, margin, abstained}` back. The reservoir is a fixed,
tiny, dependency-free **second opinion**. It never replaces the host's own
routing decision; it either agrees (route) or disagrees / abstains (fall
through).

---

## In-process (Go library)

Import path:

```go
import "github.com/caimlas/classi-fly/reservoir"
```

Snippet:

```go
// Load once at startup. Fail-closed: a missing or corrupt .fly returns an
// error, never a silently broken classifier.
res, err := reservoir.Load("model.fly")
if err != nil {
    log.Printf("second-opinion judge disabled (%v); host head only", err)
    // keep running: the feature is a no-op, the host is unaffected
}

// Per request, with the embedding your own head already produced:
result, err := res.Classify(embedding) // embedding []float32
if err != nil {
    // e.g. dimension mismatch against the artifact's embed_dim
    return err
}
_ = result // reservoir.Result{Class, Confidence, Margin, Abstained}
```

`Result.Abstained == true` means "fall through": the winning lane was below
its per-class route floor. Treat abstain and disagreement the same way —
neither ever upgrades a host decision.

**Missing-`.fly` semantics:** the feature is a no-op that fails closed. If
`Load` fails, disable the judge and keep the host's own path; the host keeps
working. Never ship a zero-value `Reservoir` or guess a class.

Metadata for logs/telemetry comes from `res.Info()`
(`Name`, `Neurons`, `Edges`, `EmbedDim`, `Steps`, `Classes`, `License`,
`Attribution`).

---

## Sidecar (HTTP)

For consumers that are not Go programs, run the sidecar:

```
classi-fly serve --fly model.fly --addr 127.0.0.1:8091
```

- `--addr` defaults to `127.0.0.1:8091` (**loopback only**).
- The artifact is loaded once at startup, before the listener opens. A bad
  artifact exits with code 2 — the sidecar never serves without a model.

### Endpoints

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/classify` | POST | `{"embedding":[...],"trace":false}` | `{"class":"...","confidence":0.9,"margin":0.2,"abstained":false}` |
| `/info` | GET | — | the artifact's header JSON |
| `/healthz` | GET | — | `200 "ok"` once loaded, `503` before |

A wrong-length embedding, an empty embedding, or invalid JSON is a `400`
with `{"error":"..."}`; `/classify` also answers `503` if hit before the
artifact finished loading.

### curl examples

```sh
# health: wire this into your supervisor / orchestrator probe
curl -fsS http://127.0.0.1:8091/healthz          # -> ok

# what is loaded
curl -fsS http://127.0.0.1:8091/info             # -> header JSON

# classify (embedding width must equal the artifact's embed_dim)
curl -fsS -X POST http://127.0.0.1:8091/classify \
  -H 'Content-Type: application/json' \
  -d '{"embedding":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],"trace":false}'
# -> {"class":"coding","confidence":0.91,"margin":0.22,"abstained":false}
```

**Exposure:** the sidecar has no authentication. It binds loopback by
default and must stay loopback-only; do not expose it on a public interface
or through a proxy. If another machine needs it, front it with your own
authenticated gateway.

---

## Licensing

Every `.fly` carries `license` and `attribution` in its header. Rules:

| Source | License | May ship? |
|--------|---------|-----------|
| Synthetic (seed-generated) | none (license-free) | yes — the default |
| Larval connectome (Winding 2023) | CC BY 4.0 | yes, with attribution |
| Hemibrain (Janelia FlyEM) | CC-BY | yes, with attribution |
| FlyWire (FAFB/BANC/MCNS/MANC) | CC BY-NC 4.0 | **no** — non-commercial only |

Before shipping or deploying an artifact, check it:

```sh
classi-fly info model.fly
```

and read the `license` and `attribution` fields. A FlyWire-derived `.fly`
must not be packed for distribution at all (packing requires an explicit
`--allow-noncommercial` escape hatch, for local research only). If you
redistribute a CC BY artifact, you must reproduce its attribution — keep the
`NOTICE` file with your distribution.

---

## Worked example: a host with an existing head

`examples/host_judge.go` shows the intended pattern for a program that
**already has a classifier head** (kNN, centroid, linear probe — anything)
and wants a cheap independent second opinion before acting:

1. The host head produces `(label, confidence)` for an embedding.
2. If `confidence` is below the host's own threshold → fall through to the
   host's slow/interactive path. The reservoir is never consulted.
3. Otherwise call classi-fly (in-process or via the sidecar).
4. Route **only if** the reservoir agrees (`res.Class == label`) **and** did
   not abstain. Disagreement or abstain → fall through.

The whole decision is one function:

```go
func judgeDecide(hostLabel string, hostConf float64, res reservoir.Result) (route bool, label string)
```

The reservoir can only *downgrade* a host decision to fall-through; it never
overrides the host's label and never rescues a low-confidence host decision.
This is the classic veto pattern: a confident action must survive an
independent, differently-built checker.

As an illustrative instance (kept generic here): an agent daemon that
classifies user intents could run its own head for routing and use this
reservoir as a second-opinion veto on high-stakes routes, falling back to a
slower deliberate path on disagreement. The library contains no knowledge of
any such consumer — it only ever sees embedding vectors.
