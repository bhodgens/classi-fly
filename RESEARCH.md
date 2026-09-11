# classi-fly - Research: a fly-connectome reservoir for meept's intent classifier

Date: 2026-09-11
Status: research + experiment proposal. No code written. Awaiting gate.

Scope: can a fruit-fly connectome, used as a fixed recurrent "reservoir",
improve the meept intent classifier - either as a direct head in the
classification chain, or as an independent judge/arbiter? Result must ship as
a small compressed package and be implemented in Go (not Python).

---

## 0. The one-paragraph answer

Yes, this is testable, and the cost of testing is low. Real evidence exists
that a fixed biological connectome works as a reusable recurrent core with
only a tiny trained readout. The decisive practical constraint is not compute
or accuracy - it is licensing: the best-known dataset (FlyWire) is
non-commercial, so it cannot ship in a product, while two other fly datasets
can. The classifier today is already the right shape for a drop-in swap (a
small head in front of the LLM chain). The main honest risk is that for text
the input is already a dense embedding, so the reservoir may add little; the
experiment is cheap enough that measuring it beats arguing about it.

---

## 1. What meept's classifier is today (verified, file:line)

The routing cascade is 3 stages, cheapest first:

1. Stage-0 embedding gate - `internal/agent/embedding_prefilter.go`.
   Embeds the raw text (OpenAI-compatible endpoint, Qwen3-Embedding-0.6B),
   then runs a kNN vote: k=5, all 5 nearest examples must agree on one
   intent, membership floor cosine 0.70 (`DefaultPrefilterThreshold`,
   `defaultPrefilterK`). On any miss it returns nil and the chain runs.
2. Stage-0.5 ModernBERT probe - runs only if Stage-0 abstains.
3. Stage-3 LFM-8B LLM chain - the 100% backstop.

Two facts about Stage-0 matter for this research:

- A second-opinion checker already exists: `internal/agent/tfidf_veto.go`.
  It is a char n-gram TF-IDF logistic classifier. Door 1 routes only when
  the kNN vote and this veto agree. It is ~2 MB, ~0.4 ms, pure stdlib, no
  network, and disabled silently when its model file is missing
  (`loadTfidfVeto` returns nil,nil on IsNotExist). This is the exact template
  for a "judge/arbiter" - the plug point already exists.
- The head is the known weak link. `docs/plans/classifier-iteration/master.md`
  records that the shipped kNN-unanimity head measures precision 90.9-91.7%
  at coverage 9.6%, while the campaign's best head (centroid + cosine margin)
  reaches coverage 34% at precision 98.5%. The shipped head is behind the
  measured frontier.

The scoring rules the campaign already agreed to (from `master.md`), which any
new head must obey:

- An incorrect classification is worse than a fall-through (abstain).
- Precision of direct routes P >= 97% (must not regress).
- OOD abstain rate OOD-R >= 95% (out-of-scope input must fall through).
- Headline metric E2E = `(gate-correct + 0.868 * abstained) / total`.
- Latency budget 500 ms for embed+decide.

Intents: 12 in `tools/classifier-eval/intent_descriptions.json5`
(coding, debugging, analysis, search, chat, platform, git, scheduling,
planning, review, reporting, recall), plus `quickplan` as the 13th
(`internal/agent/intent.go`, `IntentQuickPlan`).

---

## 2. What a "reservoir" is, and why the fly brain is a candidate

Plain terms:

- A reservoir is a fixed, random-looking recurrent network. You do NOT train
  its internal weights. You feed input in, let activity echo around for a few
  time steps, read the activity vector, and train only a small linear output
  layer. Training is therefore tiny and cheap.
- The reservoir is a feature transform. It takes an input vector, expands it
  into a higher-dimensional, nonlinear, history-dependent state, and hands
  that state to a simple classifier.

Why a fly connectome is a natural reservoir:

- The fly's learning center, the mushroom body, does exactly one thing well:
  it takes a low-dimensional sensory signal, expands it through many sparsely
  active "Kenyon cells", and reads out associations through a small number of
  output neurons. That is expansion coding plus a sparse associative memory -
  structurally the same job a reservoir does.
- The adult hemibrain contains roughly 25,000 neurons and more than 20 million
  connections. The larval brain is far smaller: 3,016 neurons and 548,000
  synapses. Both are small enough to hold in memory as a sparse matrix.
- The connectome gives real, evolution-selected topology and weights instead
  of a random matrix. Two independent studies find the biological matrix
  beats a random one - and, more useful here, that it is markedly more
  resistant to overfitting.

---

## 3. Evidence that this actually works (verified sources)

| Work | What it did | Result |
|---|---|---|
| BPU, arXiv 2507.10951 (2025) | Whole larval connectome as a fixed recurrent core; only input/output projections trained. 3,000 neurons, 65,000 weights. | 98% MNIST, 58% CIFAR-10, beats size-matched MLPs. GNN-BPU 60% chess-move accuracy; CNN-BPU beats parameter-matched Transformers. |
| "The Drosophila Connectome as a Computational Reservoir", Biomimetics 2025, doi 10.3390/biomimetics10050341 | Built an Echo State Network whose reservoir is the real connectome matrix. Compared connectome topology+weights vs random variants. | Connectome reservoir significantly more resilient to overfitting; full-connectome run stayed below 2% normalized error. Both topology and weights contribute. |
| Morra & Daley, arXiv 2201.09359 | Replaced an ESN's random reservoir with a hemibrain-derived connectivity matrix ("Fruit Fly ESN"). | 100-fold reduction in performance variance; equal or better accuracy. |
| Arena et al., IJCNN 2015 / Neural Networks 2015 | Mushroom-body-inspired spiking network for classification and sequence learning. | MB structure used directly as a classifier substrate. |

The repeated theme that matters for meept: the advantage shows up as
robustness and variance reduction / overfitting resistance, not as a large raw
accuracy jump. That is exactly the regime meept is in - a small gold corpus
(~369 cases) where a freely-trained head can overfit.

---

## 4. Datasets and licensing (the decisive constraint)

This is the part that decides the architecture. Verified:

| Dataset | Size | License | Ships in a product? |
|---|---|---|---|
| FlyWire FAFB v783 (adult female brain) | 139,255 neurons, 3,732,460 connections | CC BY-NC 4.0 | NO - non-commercial only |
| FlyWire BANC / MCNS / MANC | 158K-166K neurons | CC BY-NC 4.0 | NO |
| Janelia Hemibrain (adult, includes mushroom body) | ~25,000 neurons, >20M connections | CC-BY | YES (attribution only) |
| Drosophila larval connectome, Winding et al. 2023, Science | 3,016 neurons, 548,000 synapses | CC BY 4.0 (article) | YES (attribution only) |

Consequences:

- The FlyWire adult brain - the one most publicized, and the one the source
  chat leads with - is CC BY-NC. Weights derived from it can be used for
  research and benchmarking, but not shipped inside a commercial product.
  This kills FlyWire for the direct-head path that ships.
- The license-clean options that ship are the hemibrain (CC-BY, includes the
  mushroom body) and the larval connectome (CC BY 4.0).
- The larval connectome is the best first target: it is the smallest (3,016
  neurons), it has a published working implementation (BPU repo,
  github.com/tingshanL/BPU), its signed adjacency matrix is a single ~35 MB
  CSV, and the paper already proves it works as a fixed recurrent core.

Design note that removes the license problem entirely for the shippable path:
the *architectural principle* of the mushroom body (sparse expansion coding
from a few projection neurons to many sparsely active Kenyon cells, then a
small readout) is not copyrightable. A reservoir with that same shape,
generated from a fixed random seed, needs no connectome data at all and has no
license constraint. So there are two clean tiers (section 7).

---

## 5. Two ways it could plug into meept

Both use the same artifact. They differ only in what the dispatcher does with
the reservoir's output.

### Option A - direct head (replaces/augments Stage-0)

Path: text -> embedding -> reservoir(dynamics T steps) -> linear readout ->
intent + confidence, gated by a precision-first threshold and margin, exactly
like the current head.

- Plugs into `internal/agent/embedding_prefilter.go`, replacing or running
  alongside the `vote()` head.
- Honest caveat: this is the harder sell. The embedding is already a
  high-dimensional semantic code, so the reservoir is expanding an expansion.
  Its plausible edge is small-data robustness, not raw accuracy.

### Option B - judge / arbiter (recommended first experiment)

Path: keep the current head; add the reservoir as an independent second
opinion; route only when both agree. This is exactly what `tfidf_veto.go` does
today.

- Plugs in as a sibling of `tfidf_veto.go` (`agrees`-shaped API, loaded from a
  compressed file, missing-file-disabled).
- Advantage: zero risk to the current head, and it directly attacks the worst
  failure mode (confidently-wrong direct routes). It also gives a second,
  differently-built signal for OOD abstention.
- The reservoir is a good arbiter because it is structurally unlike both the
  kNN head and the TF-IDF head, so its errors should be less correlated.

Recommendation: build Option B first (cheap, safe, measurable), then decide
from data whether to try Option A.

---

## 6. Go implementation and small compressed package

Feasibility: fully feasible in pure Go, no CGO.

- Reservoir step = sparse matrix-vector multiply (SpMV). Go libraries exist:
  `github.com/james-bowman/sparse` (CSR/CSC, `MulVecTo`, Sparse BLAS
  `Dusmv`), or a hand-written CSR SpMV, which is ~40 lines.
- The embedding endpoint is already in the path (`PrefilterEmbedder`), so no
  new runtime dependency is introduced.

Artifact layout (proposed `.fly` package):

```
fly-reservoir.v1
  header:   version, neurons, edges, embed_dim, T_steps, encode
  reservoir CSR:  indptr (uint16 x N+1), indices (uint16 x edges), weights (int8 x edges)
  input projection:  seed (uint64) OR a stored int8 matrix
  readout:  weights (int8 x N x K) + bias (float32 x K) + per-class thresholds
```

Size estimate for the larval connectome (estimates, to be confirmed by the
spike):

| Part | Raw | Notes |
|---|---|---|
| Reservoir CSR | ~0.2 MB | 65,000 edges x (2 + 1) bytes + 3,017 x 2 byte row pointers |
| Input projection | ~0 bytes | store the PRNG seed, not the matrix (fixed random projection) |
| Readout | ~0.04 MB | 3,016 x 13 int8 = 39 KB, plus thresholds |
| **Total raw** | **~0.25 MB** | |
| **Compressed (zstd)** | **~0.1-0.15 MB** | |

Even the adult hemibrain mushroom-body subcircuit (a few thousand Kenyon
cells, order 1-2M edges) lands at a few MB raw, ~2-3 MB compressed. The
"small compressed package" goal is easy to meet; target < 5 MB, likely < 1 MB.

Latency estimate: one classification = T steps x edges multiply-adds. With
T=8 and 65,000 edges that is ~520K multiply-adds, well under 1 ms in Go. This
is the same order as the existing kNN head (which scans hundreds of example
vectors x 1024 dims). So the reservoir is not a latency regression; the 500 ms
budget is dominated by embedding, which already exists.

Go file layout (matching meept conventions):

- `internal/agent/flyreservoir.go` - load `.fly`, SpMV forward pass, predict
  intent + confidence. Mirrors `tfidf_veto.go` (missing file = disabled).
- `internal/agent/flyreservoir_test.go` - determinism, dimension mismatch,
  disable-on-missing-model.
- `internal/config` - one config block (`fly_reservoir: { enabled, path,
  assert_only }`), default OFF, per the campaign's "gate fast paths behind
  config, default OFF, assert-only first" rule.

Training (offline, may stay Python in `tools/` like the current
`build_tfidf_veto.py`): ridge regression or logistic on the frozen reservoir
states. Only the readout is trained, so this is seconds of CPU.

---

## 7. Proposed experiments (with success criteria)

Cheap, ordered, each answers one question. Uses the existing offline harness
(`tools/classifier-eval/eval_harness.py`) and the adjudicated replay ruler.

Tier 0 - can we build it at all (no connectome license needed):
- E0. Generate a synthetic mushroom-body-shaped reservoir (seed-based) in Go,
  export a `.fly` file, load it, run one forward pass. Measure artifact size
  and single-call latency.
- Success: artifact < 5 MB, latency < 5 ms, deterministic output.

Tier 1 - does a reservoir add signal as a judge (license-clean, larval CC BY):
- Start with the LARVAL brain, not the hemibrain. Reasons: (1) it is a
  complete circuit with real sensory input and motor output neurons, so input
  and output projections attach cleanly - the hemibrain is central brain only
  and lacks the optic lobes and the ventral nerve cord; (2) it is 8x smaller
  (3,016 vs ~25,000 neurons), so sweeps are faster and the package is smaller;
  (3) it is one clean downloadable file with no auth, while the hemibrain is
  reached through the neuPrint API; (4) it is the only one with a published
  working fixed-recurrent-core implementation (BPU) and its adjacency matrix
  already in a repo. The hemibrain is the upgrade path, not the starting point.
- E1. Build the larval-connectome reservoir. Train only the readout on the
  campaign's train folds. Report held-out accuracy of the standalone head.
- E2. Judge test: route only when the current head and the reservoir agree.
  Report P, C, E2E on the 5-fold set and on the 48-case real-traffic replay.
- Success bar (from master.md): E2E must beat the current veto's 87.35% and
  the chain-only 86.8% floor, with P >= 97% and OOD-R >= 95%, and - mandatory -
  disclose the number of direct routes (a 2-route win is noise).

Tier 2 - does real biology beat a synthetic reservoir (research only):
- E3. Same as E2 but with the synthetic Tier-0 reservoir. If synthetic matches
  connectome, the product ships the license-clean synthetic version.
- E4. (optional, benchmark only, NOT shipped) Repeat E1/E2 with the FlyWire
  mushroom-body subcircuit, to see whether the adult non-commercial data is
  materially better. If it is not, the license question is moot.

Tier 3 - direct head:
- E5. Only if Tier 1 clears its bar: try the reservoir as the Stage-0 head
  itself (Option A), measured against the centroid-margin champion frontier.

---

## 8. Risks and honest caveats

- Transfer is unproven. Every published reservoir/BPU result is on images,
  time series, or chess - not text intent classification. This experiment is
  the first test of that transfer.
- Embedding already expands. The mushroom body's core trick (sparse expansion
  of a low-dim code) may be redundant on top of a 1024-dim embedding. This is
  the single biggest reason the gain could be small.
- Small-sample ceiling. The gold corpus is ~369 cases and the real-traffic
  ruler is 48 cases. At n=48, a one-case flip is noise. Any win must be
  reported with route counts and treated as unvalidated until live outcome
  data (`harvest_outcomes.py`) accumulates - exactly the lesson the campaign
  already learned from the TF-IDF veto.
- License hygiene. FlyWire data (CC BY-NC) must never enter a shipped
  artifact. Keep the hemibrain/larval (CC BY) or synthetic path for anything
  that ships; keep FlyWire strictly in the offline benchmark tier.
- Attribution. Hemibrain (CC-BY) and larval (CC BY 4.0) require attribution in
  the shipped package (a NOTICE/manifest field), same as any third-party asset.
- Quantization risk. int8 weights on a reservoir with recurrence can drift
  over T steps. Validate the quantized model against the float model on the
  replay before shipping (the campaign's "re-validate under the honest
  protocol" rule applies).

---

## 9. Recommendation

1. Run Tier 0 and Tier 1 first. They are cheap, license-clean, and answer the
   real question (does the reservoir add signal as a judge).
2. Prefer Option B (judge/arbiter) over Option A (direct head) for the first
   cut - it is the existing `tfidf_veto.go` pattern, so it is low-risk and its
   effect is easy to measure against the known 87.35% bar.
3. Default the feature OFF behind config, run assert-only first, and gate any
   routing change on real-traffic precision - the campaign's standing rule.
4. Treat FlyWire as a benchmark-only curiosity. Ship the larval connectome
   (CC BY 4.0) or a synthetic MB-shaped reservoir.

Decisions needed from you before any code:
- (a) Confirm the judge-first approach, or ask for the direct-head approach.
- (b) Approve downloading the larval connectome (~35 MB CSV from
  github.com/tingshanL/BPU) and the Go sparse dependency decision (hand-rolled
  CSR vs `james-bowman/sparse`).
- (c) Say whether the synthetic (license-free) reservoir is acceptable as the
  default shipped artifact if it matches the connectome in E3.

---

## 10. Sources

- BPU paper: https://arxiv.org/abs/2507.10951
- BPU code: https://github.com/tingshanL/BPU
- Drosophila connectome as a reservoir: https://doi.org/10.3390/biomimetics10050341
- Morra & Daley, connectome ESN: https://arxiv.org/abs/2201.09359
- Larval connectome (Winding 2023, Science): https://doi.org/10.1126/science.add9330
- Larval connectome data mirror: https://github.com/brain-networks/larval-drosophila-connectome
- Hemibrain (CC-BY): https://www.janelia.org/project-team/flyem/hemibrain
- FlyWire Codex (CC BY-NC): https://codex.flywire.ai
- FlyWire license: https://flywire.ai/guidelines
- Go sparse matrix library: https://github.com/james-bowman/sparse
