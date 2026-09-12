# Would MaleCNS (male CNS, 166,691 neurons) beat the larval brain in classi-fly?

Assessment, 2026-09-12. Question raised by the owner. Answer: **no measurable
gain expected, at roughly 100x the cost.** Everything below is either a
verified fact from the source or a measurement run for this document.

## 1. The dataset (verified)

From the Janelia project page, the download page, and the manuscript (Berg et
al. 2025, bioRxiv 2025.10.09.680999; Cell 2026):

| Property | Value |
|---|---|
| Neurons | **166,691** (full male CNS: central brain, optic lobes, ventral nerve cord) |
| Connectome graph | **25.6M edges** between 166,391 neurons |
| Cell types | 11,691 |
| Released | v1.0, 2026-06-08 |
| **License** | **CC-BY** - shippable with attribution, same class as the hemibrain |
| Access | neuPrint API (`dataset='male-cns:v1.0'`, needs an account + token); bulk download `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`; the connection graph is `connectome-weights-male-cns-v1.0-minconf-0.5.feather`, **1.1 GB**; annotations 13 MB; also a neo4j database dump |

Compared with what classi-fly uses today (larval: 2,952 neurons, 63,545 edges =
21.5 edges per neuron), MaleCNS is **56x the neurons and 403x the edges**, at
**7.2x the density** (154 edges per neuron).

Licensing is not the obstacle - CC-BY is fine. The obstacles are size, cost,
and the measurements below.

## 2. Does size help? Measured - no (saturation near 32K)

Synthetic reservoirs of matched density (fan-in 20, ~4 tanh steps, same folds,
same corpus, dual-form ridge with the penalty swept per size):

| neurons | edges | projection MB | clean (of 361) | truncate-75% (of 361) |
|---:|---:|---:|---:|---:|
| 512 | 10,240 | 2.1 | 232 | 91 |
| 2,048 | 40,960 | 8.4 | 255 | 128 |
| 8,192 | 163,840 | 33.6 | 260 | 177 |
| 32,768 | 655,360 | 134.2 | **264** | 205 |
| 131,072 | 2,621,440 | 536.9 | 261 | 172 |
| **166,000** | 3,320,000 | **679.9** | **256** | 187 |
| *larval reference (2,952 real neurons)* | *63,545* | *12.1* | *262* | ***208*** |

Clean accuracy saturates at 2,048-32,768 neurons and then drifts **down**.
Robustness peaks in the same band and also falls off. At 166,000 neurons the
synthetic reservoir is *worse* than the real larval connectome at 2,952 - on
both metrics - while needing a 680 MB projection matrix.

## 3. Does density help? Measured - no

MaleCNS's 154 edges per neuron is 7x the larval density, which could have been
a mechanism the size sweep missed. It is not:

| neurons | fan-in | edges | clean | truncate-75% |
|---:|---:|---:|---:|---:|
| 8,192 | 20 | 163,840 | 262 | 164 |
| 8,192 | 60 | 491,520 | 259 | 161 |
| 8,192 | **154** (MaleCNS density) | 1,261,568 | **255** | 215 |
| 8,192 | 400 | 3,276,800 | 258 | 180 |
| 32,768 | 154 | 5,046,272 | 257 | 206 |
| *larval reference* | *21.5* | *63,545* | *262* | *208* |

Clean accuracy is flat in density (255-262, no trend). Robustness moves
erratically (161-215) with no monotone relationship to density. Density buys
nothing, at 3x to 20x the edge count.

## 4. Why this was predictable from the existing results

Five levers have now been measured on this task, and the connectome is not
load-bearing in any of them:

| lever | verdict |
|---|---|
| topology (mushroom body vs whole brain vs random same-size) | no - MB is worse than random |
| size (512 to 166,000 neurons) | no - saturates near 32K |
| density (20 to 400 edges/neuron) | no - flat |
| spectral radius (0.5 to 1.3) | no - the shipped scaling is already optimal |
| trained input projection (vs fixed random) | no - parity to worse |

What actually delivers is the **random projection plus about four tanh steps**,
and both are already present at 2,952 neurons. A bigger connectome cannot add
information that the 1024-dimensional embedding does not already contain.

## 5. What MaleCNS would cost in classi-fly (projected)

At 166,391 neurons and 25.6M edges, using the measured throughput of the
current Go runtime (3.3M multiply-adds per call takes 1.9 ms):

| | today (larval) | MaleCNS | factor |
|---|---|---|---|
| artifact, compressed | 1.66 MB | ~200-250 MB (raw ~300 MB) | ~130x |
| resident memory (Go, float64) | 30.6 MB | ~1.6 GB (float32 projection: ~0.9 GB) | ~50x |
| multiply-adds per call | 3.3M | ~272M (drive 170M + 4 x 25.6M) | ~82x |
| latency per call | 1.90 ms | ~100-160 ms | ~50-80x |
| load time | 7.2 ms | ~1.5-3 s | ~200-400x |
| data to fetch | 34 MB CSV | 1.1 GB feather (+ new ingest path) | 32x |
| license | CC BY 4.0 | CC-BY | same class |

The input projection alone (1024 x 166,391) is 170 MB as int8 and 1.36 GB as
float64 - and it is the *dominant* cost, not the connectome.

## 6. Recommendation

**Do not port MaleCNS.** It is license-clean and technically accessible, but
every measured axis says it would cost roughly 100x more (artifact, RAM,
latency) for no accuracy or robustness gain, and the closest synthetic analogue
tested at the same scale (166,000 neurons) scored slightly worse than the
current larval reservoir.

If MaleCNS is ever wanted for a different reason - its annotations are the
richest available, including 11,691 cell types and dimorphic cell catalogues -
that is a data-science use, not a classifier use.

The one thing worth keeping from this assessment: the size sweep shows the
current choice (2,952 neurons) already sits in the useful band, and that a
**larger** reservoir would be a regression. If the artifact ever needs to be
smaller, the sweep says 512-2,048 neurons is the range to try - at 2.1-8.4 MB
of projection and a fraction of the latency.

## Reproduce

- Size and density sweeps: `data/size_sweep.json` plus the inline runs recorded
  in this document (synthetic matrices, fan-in as labelled, fan-in 20 at
  matched density for the size axis).
- Method notes and the reference numbers: `docs/EXPERIMENTS.md`,
  `docs/AXES-2026-09-12.md`.
- Source facts: <https://male-cns.janelia.org/>, the download page, and Berg et
  al. 2025 (166,691 neurons; 25.6M edges; CC-BY).
