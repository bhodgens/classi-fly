# Directions

Exploration notes, started 2026-09-12. This is a working document: it records
what we are considering, what has already been measured, and what each option
would cost. Update it in place as lanes open and close.

Related: `docs/EXPERIMENTS.md` (classification results and corrections),
`docs/AXES-2026-09-12.md` (the four levers), `docs/MALECNS-ASSESSMENT.md`
(why a bigger connectome does not help), `docs/MEEPT-INTEGRATION.md` (how any
of this would attach to meept).

---

## 1. What the "fly plays Doom / Beat Saber / Mario" projects actually do

Summary: **not trained, and only marginally learning.** Four separable layers,
and only the last one adapts.

| Layer | What it is | Trained? |
|---|---|---|
| Wiring (the connectome) | fixed graph, 25.6M connections in MaleCNS v1.0 | No - it is data, not parameters |
| Input encoding (pixels to retinal stimulation) | hand-built transducer; "inferred retinal inputs", retinal geometry / brightness / tonic lamina current "assumed" | No - hand-designed and calibrated |
| Output decoding (activity to keystrokes) | fixed; four descending neurons (DNp20 turns, DNpe017 rate) "chosen after visual-response calibration" | No - hand-picked, calibrated |
| Adaptation | damage stimulates two PPL101 dopamine cells; plus "KC adaptation and modeled chemical modulation" | Yes - local plasticity with a hand-wired reward channel |

Points that matter more than the demos:

- The Doom project's own description calls it "reconstructed wiring with
  approximate neuron dynamics, not a validated fly emulation", and states "No
  enemy coordinates, navigation policy or additional aiming policy enters the
  decoder."
- Its earlier baseline "produced controls with black input" - the agent moved
  with no visual signal. It also reports the biological walking and mouthpart
  readouts were **silent** in the initial test, i.e. the connection to
  behaviour was not established.
- "Useful vision remains unvalidated."
- Data-quality caveats from the same source: 42 unmapped photoreceptors,
  3,718 uncertain transmitter signs.
- The sibling demos are early: Mario repeatedly walks into a wall, and one
  project describes itself as "100% vibe coded ... just for fun".

So the reusable invention is **not** the connectome. It is the loop around it:
a stimulus transducer, a fixed decoder picked by calibration, and a
dopamine-style reward wire into local plasticity.

## 2. Where a fly map plausibly helps (ranked)

The one measured advantage we have is robustness to missing input dimensions
(+18.8 pt at 75% truncation, +7.8 pt at 50% dropout; `docs/AXES-2026-09-12.md`),
and it comes from a random projection plus about four tanh steps - not from
biology. Classification is therefore the wrong target. The lanes below are
ordered by how well they match the substrate.

1. **Fixed-graph control loop.** The substrate's intended job. Our Go runtime
   does 1.9 ms per forward pass (~500 Hz) and a 512-1,024-neuron reservoir
   would be several times faster, so a real-time loop is feasible in the
   language and artifact shape we already ship. First experiment worth running:
   a 2D light-seeking or looming-avoidance task (behaviours the fly actually
   implements) with a hand-built transducer and a fixed decoder, benchmarked
   against a two-line heuristic controller. Cheap and decisive.
2. **Plasticity instead of a frozen readout.** The one thing every classi-fly
   test so far has held constant. Explored below - the naive version loses.
3. **Temporal and rhythmic tasks.** Beat Saber is a timing task, and timing is
   what a static head lacks. Stream anomaly detection, event ordering, and
   interval/rhythm sensing fit a fixed recurrent substrate, and our robustness
   result applies directly to streams with dropped frames.
4. **Geometry and motion, using the CNS-male specifically.** It carries optic
   lobes and the ventral nerve cord in one graph - a full sensorimotor loop,
   which is what makes it interesting for control (not classification). The
   hard part is always the transducer, not the graph.
5. **A reproducible fixed substrate for benchmarking** implementations (our Go
   runtime, CUDA, browser). Low glamour, real value, already half-built.

## 3. Lane 2 exploration: does online adaptation help?

### The question

The Doom pattern is a fixed graph plus a hand-wired reward channel plus local
plasticity. Every classi-fly result so far used a frozen readout. So: if a
deployed head may update itself from corrections, does it get better?

### What was measured

`tools/eval/plasticity_probe.py`, results in
`tools/eval/plasticity_probe_results.json`. Setup: larval reservoir states
(2,952 neurons), 361 gold cases, 5 folds, precision-first routing gate (top-1
at or above its calibrated per-class threshold, margin > 0.05), corrections
applied only where the head was wrong.

| variant | routes | P | C | E2E |
|---|---:|---:|---:|---:|
| **frozen readout (what classi-fly ships)** | 71 | 0.958 | 0.197 | **0.8857** |
| online delta-rule eta=0.05 | 68 | 0.956 | 0.188 | 0.8846 |
| online delta-rule eta=0.2 | 66 | 0.955 | 0.183 | 0.8838 |
| online delta-rule eta=1.0 | 102 | 0.902 | 0.283 | 0.8776 |
| threshold-only adaptation, step 0.02 | 71 | 0.958 | 0.197 | 0.8857 |
| threshold-only adaptation, step 0.10 | 64 | 0.953 | 0.177 | 0.8831 |
| anchored weight drift (EWC-lite) | 68 | 0.956 | 0.188 | 0.8846 |

E2E = (route_correct + 0.868 x abstained) / total, the campaign convention.

A separate, more aggressive run (higher learning rates, replay buffer) was
worse still: eta=0.2 dropped accuracy 5.5 points and replay made it 21 points
worse, with the second half of the stream consistently worse than the first -
classic forgetting.

### The finding that matters

**The correction stream is almost empty: 3 wrong routes out of 361 cases at
P 0.958.** A precision-first gate emits so few errors that there is nothing to
learn from. Every adaptation mechanism tested either does nothing (identical
numbers) or costs a fraction of a point, and aggressive weight updates actively
degrade the head.

This is structural, not a tuning failure: the better the gate, the less
correction signal exists. Adaptation and precision pull against each other.

### Verdict

- Lane 2 as **per-example online weight learning: closed, negative.** Do not
  build it.
- Lane 2 as **calibration-only adaptation** (adjust per-class abstention
  thresholds from observed corrections, never touch what the head predicts):
  neutral in this data, low risk, plausible only over much longer horizons or
  much higher error rates. Not worth building yet.
- Lane 2 as **scalar-reward reinforcement** (what the Doom project actually
  does - a reward scalar, no label): untested. High variance, and in meept's
  setting the payoff for a routing decision is delayed and sparse, so I would
  rate this the weakest of the three.
- The real value of this exploration is diagnostic: it explains why a **frozen
  head plus periodic batch retraining** (what meept already does via harvest ->
  corpus -> rebuild index) is the right architecture at these error rates. Batch
  retraining uses all the data; online learning would use only the 3 errors.

## 4. What this means for meept

- meept already has a *better* signal than the Doom project: `dispatch_log`
  carries `outcome` (pending/ok/corrected/failed_replan), `corrected_agent`,
  `margin`, `input_hash`, `turn_no`, and `harvest_outcomes.py` turns corrections
  into adjudication candidates. That is a **label**, not just a reward scalar.
- It is still too sparse to drive online learning, and the existing batch loop
  is strictly better value. So: no change recommended today.
- Two constraints that bound any adaptation work: the routing gate only fires
  for non-task, non-async intents (~35% of traffic; `dispatcher.go:893-928`),
  and a confident wrong route is worse than a fall-through by our own rule.
  Any adaptive mechanism must therefore abstain more, not claim more.
- If lane 2 is ever revisited, the safe shape is: **threshold sidecar +
  periodic batch refit**, with the drift monitored against the frozen prior.

## 5. What this means for classi-fly specifically

- Today the artifact is read-only, deterministic, and fail-closed. A plastic
  head would make it **stateful**: per-host weights that drift, needing
  persistence, versioning, an audit trail, and a way to roll back. That is a
  product-shape change, not a model change, and it would cost the two
  properties that make the artifact attractive to a host.
- Recommendation: keep the frozen artifact. If adaptation is ever wanted, ship
  a small mutable **calibration sidecar** (per-class thresholds) separate from
  the immutable `.fly`, so the model file stays reproducible and auditable.

## 6. Open questions to settle before spending more time

1. Is there a meept-adjacent task where the error rate is high enough for
   adaptation to have signal (a harder taxonomy, a multi-label surface, a
   per-user shift), or is every candidate as precision-gated as routing?
2. Does the CNS-male optic lobe plus VNC earn its keep in a *control* loop,
   where the transducer is hand-built? This is the only remaining place where
   the fly graph might beat a hand-written controller - and it is the same
   "does biology matter" question asked in the domain the brain evolved for.
3. For lane 3 (temporal), would our existing Go runtime plus a small reservoir
   beat a simple state machine on a stream anomaly task? Cheap to test.
4. Is the robustness win (missing input dimensions) actually reachable in
   production meept, or is it solving a failure mode that never occurs?

## 7. Lane status

| lane | status | evidence |
|---|---|---|
| 1. fixed-graph control loop | open, untested | none yet - ranked first |
| 2. online plasticity | **closed (negative)** | `tools/eval/plasticity_probe.py` |
| 2b. calibration-only adaptation | benched, low priority | same run: neutral |
| 2c. scalar-reward reinforcement | untested, lowest priority | rationale above |
| 3. temporal / rhythmic tasks | open, untested | none yet |
| 4. geometry & motion (CNS-male) | open, untested; MaleCNS cost assessed | `docs/MALECNS-ASSESSMENT.md` |
| 5. fixed benchmark substrate | partially done | Go runtime + benchmarks exist |
