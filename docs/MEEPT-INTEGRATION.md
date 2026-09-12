# classi-fly x meept — integration design and measured benefit

Status: design complete; benefit measured. Numbers from
`docs/EXPERIMENTS.md` + `tools/eval/e1_judge_corrected.json`.
Verdict: **the machinery is a good fit; the classifier is not.**

---

## 1. Where it would plug into meept's chain

meept routes a `ClassifyAndRoute` call through a 3-stage cascade:

| Stage | Component | File |
|---|---|---|
| 0 | embedding prefilter: kNN k=5 unanimity (floor 0.70) + TF-IDF veto agreement | `internal/agent/embedding_prefilter.go`, `internal/agent/tfidf_veto.go` |
| 0.5 | ModernBERT probe (only if Stage 0 abstains) | campaign `docs/plans/classifier-iteration/` |
| 3 | LFM-8B LLM chain (100% backstop) | `internal/agent/llm_classifier.go` |

classi-fly has exactly two possible slots, and both already exist as patterns
in that repo (so wiring effort is small):

**Slot A - Stage-0 head replacement.** The `PrefilterEmbedder` already
produces the embedding; the reservoir head consumes that vector and returns
`{class, confidence, margin, abstained}`. It is a drop-in for the `vote()`
head: same input, same output shape, same config surface
(`enabled`, `assert_only`, model path, missing-file-disabled).

**Slot B - second-opinion arbiter.** A sibling of the TF-IDF veto: route only
when the existing head and classi-fly agree. Same `agrees()` shape, same
missing-model-is-disabled contract.

## 2. Measured numbers for each role

Same corpus (meept's own adjudicated corpora: 361 gold cases, 13 intents, 28
OOD), same protocol (5-fold, seed 42, per-class train-side calibration at
target precision 0.97, margin floor 0.05, route counts disclosed), with the
readout penalty swept properly:

| Role | Routes | P | C | E2E | vs 0.868 chain floor |
|---|---:|---:|---:|---:|---:|
| Slot A: reservoir as Stage-0 head | 71/361 | 0.958 | 0.197 | **0.8857** | +1.8 pt |
| Slot A alternative: linear probe on the same embeddings | 88/361 | 0.921 | 0.244 | 0.8808 | +1.3 pt |
| Slot B: judge (existing head AND reservoir agree) | 70/361 | 0.957 | 0.194 | 0.8853 | +1.7 pt |
| Slot B variant: either head may route (OR) | 89/361 | 0.921 | 0.247 | 0.8812 | +1.3 pt |
| Reservoir alone (baseline row) | 71/361 | 0.958 | 0.197 | 0.8857 | +1.8 pt |

## 3. Why Slot B does not work

**Arbitration only helps if the two heads are informationally independent.**
The existing TF-IDF veto works because it reads raw character n-grams - a
different information source from the embedding. The reservoir consumes *the
same embedding vector the host head already consumed*, so its errors are
correlated with the host's. Measured: judge-AND (0.8853) is within one case of
the reservoir alone (0.8857), and OR (0.8812) is within one case of the probe
alone (0.8808). Agreement adds no information; it only re-labels the same
decision.

If you want a useful arbiter for meept, it must see something the embedding
does not: raw text (TF-IDF already does this), the conversation/turn history,
the session's plan state, or tool-call context.

## 4. What the machinery is genuinely good for

These are real properties, all verified by the test suite:

- **Pure Go, no CGO, no Python, no model server at inference.** One static
  binary plus a `.fly` artifact. meept's existing subprocess supervisor is not
  even needed for the in-process face.
- **Small.** The artifact carries a 2,952-node sparse graph plus int8 weights;
  the target was <5 MB and the fixture artifacts are tens of KB. Quantization
  showed zero drift across 64 probes.
- **Deterministic and fail-closed.** Identical input gives identical output;
  a missing or corrupt artifact returns an error and the host keeps working
  (`Load` never yields a silently broken classifier).
- **License-clean by construction.** Synthetic artifacts carry no third-party
  data; the larval connectome (CC BY 4.0) and hemibrain (CC-BY) are shippable
  with attribution; FlyWire (CC BY-NC 4.0) is fenced and refuses to pack
  without an explicit flag.
- **Two consumer faces.** Go import or loopback sidecar (`POST /classify`),
  both wrapping one core.
- **No per-host training.** The readout is baked into the artifact; a host
  just loads and calls.

## 5. The honest recommendation for meept

1. **Do not replace meept's Stage-0 head with this reservoir.** At C 19.7% /
   P 0.958 it is *worse* than the campaign's measured champion head
   (centroid + cosine margin: C 34% at P 98.5% on the campaign ruler) and it
   does not beat a linear probe on the same embeddings (C 24.4% / P 0.921).
   It also misses the campaign's pre-registered 97% precision gate.
2. **Do not add it as an arbiter.** Measured: correlated inputs, no gain.
3. **If meept ever wants a Stage-0 gate that is not the shipped kNN head,**
   the cheapest correct option is a centroid or a linear probe over the
   embeddings already computed - a 1024x13 matrix, ~13 KB, trivial in Go, no
   connectome, no reservoir, and it scores the same or better.

**Where the machinery could still earn a place in meept:** as the *container*
pattern, not the brain. The `.fly` format + `Load`/`Classify` + sidecar +
`assert_only` gate is a reusable shape for shipping any small classifier head
as a self-contained, license-audited, deterministic Go artifact. That is
useful independent of the reservoir research result.

## 6. Conditions that would change this answer

- A corpus large enough to resolve sub-2-point differences (this corpus cannot:
  every head lands within ~2 cases of the others).
- The untested axes on the connectome side: trained input projection and
  proper echo-state spectral-radius scaling (axis A of the 2026-09-11 run),
  mushroom-body-only subcircuit (axis B), alternative readout heads (axis C),
  and robustness under corrupted embeddings (axis D). If any of those shows a
  real gain, revisit Slot A.
- Any head that consumes information the embedding does not (session state,
  turn history, raw text) - that is where an arbiter can actually help.
