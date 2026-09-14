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

## 7. Lane 1 results (2026-09-12) - the first positive result in the project

Harness: `tools/control/world.py` (7 sensors -> 2 motor commands, unit-disc
phototaxis arena, 200-episode standard evaluation, modes `open` and `blink`).
Runners and raw rows: `tools/control/reservoir_{imitation,search,selective}.py`
and their `_results.json`.

Bars (measured): heuristic 100% / 17.5 steps (open), 100% / 50.6 (blink3);
memory-heuristic 100% / 18.1 (blink3); random 6.5%. Success saturates for both
good baselines, so **mean steps-to-success is the discriminating metric**.

| controller | open | blink3 | blink8 |
|---|---|---|---|
| reservoir + imitation readout (1A) | 100% / 17.4 | 100% / **17.8** | 100% / 40.9 |
| direct linear map, same 7 sensors, same fit | 100% / 17.5 | 50.5% / 32.9 | 42.5% / 61.7 |
| reservoir + random decoder | ~1% | ~1% | ~1% |
| teacher (hand-written) | 100% / 17.5 | 100% / 18.1 | 100% / 31.4 |
| reservoir + reward-search decoder (1B) | 100% / 17.9 | 99.5% / 31.6 | - |
| dense trained readout on all 2952 states (1C) | 100% / 23.2 | 100% / 23.7 | - |
| **DOOMFLY 2-neuron calibrated readout (1C)** | 100% / **18.4** | 49% / 71.1 | - |
| degree-preserving shuffled wiring (1C) | 100% / 23.2 | 100% / 22.4 | - |
| random sparse wiring, same counts (1C) | 100% / 23.1 | 100% / 22.3 | - |

### What this establishes

1. **Recurrence pays - the first measurable win for the substrate anywhere in
   this project.** On the partially observable task the reservoir holds 100%
   success while a memoryless map of the same sensors collapses to 50.5%
   (blink3) and 42.5% (blink8). Independently reproduced by the orchestrator
   with its own code (reservoir 100% / 17.1 vs direct 66.7% / 76.5 at n=60).
   The mechanism is not mysterious: the light sensors are blanked two steps in
   three, so a memoryless map literally cannot know where the light is.
2. **The connectome is still not special.** 1C: real wiring is never better -
   indistinguishable in open mode, and in blink mode it is slightly *worse*
   than all five shuffled and all five random draws. That is the third
   independent confirmation of the "biology is not the active ingredient"
   result, now in the control domain rather than classification.
3. **Reward search works but loses to imitation.** 1B: evolution strategy over
   32-64 parameters reaches imitation parity in open mode (17.9 vs 17.4 steps)
   and 99.5% in blink3 - but at 31.6 steps versus imitation's 17.8. It learns;
   it needs a teacher to match one. Cost: 432,000 training episodes, 51.4M env
   steps, ~29 minutes of wall clock.
4. **The DOOMFLY pattern is genuine.** 1C: a two-neuron readout with one
   calibrated gain per action - exactly the published pattern - reaches 18.4
   steps in open mode, beating the dense 2952-input trained readout (23.2) and
   nearly matching the hand-written rule (17.5). Selection matters: uncalibrated
   random 8-neuron draws lose 10-30 steps. So the method people are using on the
   male CNS is sound; it is the *wiring's* contribution that is not.
5. **Weight scale dwarfs every other knob.** 1B: the raw int8 connectome has
   spectral radius 42.7 and saturates 79% of units, capping blink learning at
   29.5%; normalizing to spectral radius 0.9 lifts it to 98.5%. Bigger effect
   than the search dimension or the training signal. Control-domain echo of the
   classification sweep: scaling, not topology.

### Bounds and caveats

- Blind-gap length matters: at blink8 the reservoir is 100% but 9.5 steps
  slower than the teacher, so it degrades faster than a hand-written rule as
  observability drops.
- The transducer is 7 synthetic sensors; this says nothing about real visual
  input, and nothing about the male CNS at scale.
- One harness bug was found and fixed during this lane (declared 8 sensors,
  emitted 7). An (8, N) random draw and a (7, N) draw share identical rows 0-6,
  so no measured result changes - verified directly.

## 8. Mapping lane 1 back onto classification (meept and elsewhere)

Lane 1's win was memory under partial input. Only one meept classification
sub-problem has that shape, one lesson transfers without any model, and one is
conditional.

### 8.1 The real mapping: session-state intents as a SEQUENCE problem

The campaign recorded that some intents are not decidable from the message
alone - "implement tasks 7 and 8" is code text but may mean "execute the
approved plan", and the discriminator is whether a plan is active. Per-message
accuracy on that class plateaus near 84% (docs/plans/classifier-iteration/).
That is a partially observable problem, exactly the shape lane 1 showed a
recurrent substrate can handle.

Design: classify a short SEQUENCE of turns rather than one message. Per-turn
features are cheap - message embedding, plan-active flag, tracked-task count,
previous agent, previous-turn-was-a-correction, turn index. A fixed recurrent
core folds the sequence into one state; a small readout names the intent. Only
the readout trains.

**Cheap prerequisite test - do this first, and it needs no reservoir.**
Concatenate the last three turns' features and fit a plain logistic
regression. If that beats the per-message ceiling on plan-versus-code, session
context is usable and a recurrent core is worth trying. If it does not, the
ceiling is real and no architecture will fix it. Test the hypothesis before
building the machinery.

Gate: requires session-labelled transcripts. **Data audit (2026-09-12) says
they do not exist yet.** From `~/.meept/metrics.db` table `dispatch_log`:

| fact | measured |
|---|---|
| dispatch rows | 606 |
| distinct sessions | 489 |
| sessions with 1 row | 484 |
| sessions with 2 rows | 4 |
| sessions with 3 or more rows | 19 |
| rows with `outcome` beyond 'pending' | **1 of 606** |
| rows with a populated `corrected_agent` | 0 |

So a session-sequence model is not trainable today: there are 19 multi-turn
sessions, and the outcome loop is installed but has recorded essentially no
outcomes, which also means there is no correction signal to learn from - the
same emptiness the plasticity probe found (3 corrections in 361 cases at the
shipped precision). The mapping is real but **blocked on instrumentation**: the
cheapest unblock is to start populating `outcome` / `corrected_agent` and to
harvest multi-turn sessions, not to build a model.

### 8.2 The immediate mapping: scaling discipline, no new model

Lane 1's largest single effect was scaling, not structure: raw connectome
weights saturate 79% of units and cap learning at 29.5%; rescaling to spectral
radius 0.9 lifts the identical network to 98.5%. Classification hit the same
class of bug twice - a ridge penalty of 1e-3 where 30 was correct (+9 points),
and a projection scale that mattered more than the network.

Checklist to apply before any head work: sweep embedding normalisation, route
threshold calibration, and any linear head's penalty. Cheap, and it has already
paid twice.

**Tested 2026-09-13 (`docs/MEEPT-TESTS.md` test 2): confirmed, but small.** All
three knobs move E2E (penalty 0.0177 > normalisation 0.0118 > calibration
0.0052). Best config found - standardised embeddings, penalty 100, global
threshold - gives 93 routes at P 0.9570 and E2E 0.8909, about +3.6 cases over
the probe baseline. **Adopt the procedure, not that config:** OOD abstain rate
falls to 0.857 (below the 0.95 bar) and a 3.6-case margin is inside noise at
n=361.

### 8.3 The conditional mapping: degraded embeddings - RETRACTED 2026-09-13

Measured again against meept's actual head shapes
(`docs/MEEPT-TESTS.md` test 3): the reservoir's robustness edge is real against
a linear probe (+68 cases at 75% truncation, reproduced) but the **centroid head
is better under missing input** (+32 cases at truncation, +25 at dropout), and
the shipping gate erases the difference - at 75% truncation every head sits
within 0.002 of the 0.868 chain floor with the reservoir routing zero cases.
Under dense noise the reservoir is the *less safe* head (routes 77 cases at
precision 0.22 at sigma 0.2, while the cosine-margin head abstains).

**Retracted: do not add the reservoir for degraded embeddings.** The transferable
finding is a safety property - cosine and vote margins fail safe, probability
margins do not - which argues *for* the centroid head shape in meept issue #39.

### 8.4 Corroboration of an existing recommendation

1C found that two hand-picked neurons with calibrated gains beat a dense
trained readout over all 2,952 neurons. That is the control-domain form of "a
few prototypes plus calibration beats dense learned weights on a big fixed
transform" - independent support for the centroid head recommendation in meept
issue #39.

### 8.5 What not to do

- Not a per-message reservoir head: parity with a ~13 KB linear probe at ~12x
  the memory.
- Not online learning from corrections: 3 corrections in 361 cases at the
  shipped precision - no signal (`tools/eval/plasticity_probe.py`).
- Not the fly wiring: three independent confirmations across classification and
  control that it is not the active ingredient.

## 9. Lane status

Housekeeping closed 2026-09-14: the 8 stray tracked .pyc files are untracked
and pushed (c201d9b); the Qwen3 embedding server (pid 28308) used for the
real-embedding runs has been stopped. Restart command is recorded in
docs/EXPERIMENTS.md's companion notes: meept's scripts/embed_server.py on port
8090 with /Volumes/LLMs/Qwen3-Embedding-0.6B-4bit-DWQ.

| lane | status | evidence |
|---|---|---|
| 1. fixed-graph control loop | **done - recurrence pays, biology does not** | `tools/control/`, table above |
| 2. online plasticity | **closed (negative)** | `tools/eval/plasticity_probe.py` |
| 2b. calibration-only adaptation | benched, low priority | same run: neutral |
| 2c. scalar-reward reinforcement | **partially answered by 1B**: reward-only search learns but trails a teacher | `tools/control/reservoir_search_results.json` |
| 3. temporal / rhythmic tasks | open, untested - now better motivated (recurrence paid off) | none yet |
| 4. geometry & motion (CNS-male) | open, untested; MaleCNS cost assessed | `docs/MALECNS-ASSESSMENT.md` |
| 5. fixed benchmark substrate | partially done | Go runtime + benchmarks exist |

### Safety-first gate (owner decision 2026-09-13: OOD policy = safety)

Status: three-part wave dispatched. (a) real-OOD mining from meept's
adversarial corpus - **WAITING-429** (rate-limited at dispatch; to be
re-dispatched after the siblings land); (b) safety-first gate module (centroid +
cosine margin, supervised OOD probe as a second opinion, calibrated on the
probe-passing subset, fail-safe under noise); (c) lane-1 hardening (blink-period
sweep + 512-neuron size test). Results pending.

Also this wave: meept issue #40 filed (outcome-loop instrumentation), measured
evidence posted on issue #39.

### Next candidates, in priority order

1. **Lane 3 (temporal),** now the strongest open lead: recurrence demonstrably
   paid off on a partially observable task, so a stream/rhythm task is the
   natural next test, and unlike classification it is not saturated.
2. **Harden lane 1:** sweep blink period and sensor noise to find where the
   recurrence advantage disappears, and test whether it survives with only 512
   neurons (which would make the artifact cheap).
3. **Real visual input** (optic lobe or a real camera-frame transducer) if the
   synthetic result is to be extended - this is where MaleCNS's optic lobes
   would actually be the point, at the cost documented in the MaleCNS
   assessment.
