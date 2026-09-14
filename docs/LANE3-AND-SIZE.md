# Lane 3 + size co-sweep (2026-09-14) — results

Both experiments landed on the same day as their dispatch. Runners:
`tools/control/rhythm_task.py`, `anomaly_task.py`, `size_cosweep.py`. Raw rows:
`lane3_results.json`, `rhythm_task_results.json`, `anomaly_task_results.json`,
`size_cosweep_results.json`. Protocol: 200-episode evaluation from seed 1000 on
the frozen phototaxis harness, blink mode.

## Lane 3 — rhythm and anomaly: the reservoir wins, and the biology finally differentiates

### Rhythm (pulse on the step before each expected light onset)

| head | P=3 | P=8 |
|---|---|---|
| **larval reservoir** | **200/200 pass**, 100% hits, 0.0 false/100 | 0/200 (31.8% hits) |
| synthetic 2048 | 193/200, 98.6% hits | 0/200 (2.0%) |
| memoryless linear | 0/200 | 0/200 |
| memory heuristic | 0/200 (58 false/100) | 0/200 |
| constant thrust | 0/200 (67 false/100) | 0/200 |

### Anomaly (period P changes to 2P in the last third; flag it)

| head | P=3 | P=8 |
|---|---|---|
| **larval reservoir** | **152/200 flagged**, median latency 5 steps, 0.0 false/100 | 0/200 |
| synthetic 2048 | 0/200 | 0/200 |
| memoryless linear | 0/200 | 0/200 |

**Verdicts: RESERVOIR WINS on rhythm P3 and anomaly P3.** The memoryless map
cannot solve either task by construction, but neither can the memory heuristic
or a constant-output controller. Only the larval reservoir tracks the rhythm
and detects the change. At P8 every head fails - 200-step exploration episodes
do not expose an 8-step period well enough for the readout to fit, and wider
penalty sweeps did not help.

**The surprise: the real connectome beat the synthetic matrix on every
discriminating cell for the first time in this project** - 200/200 vs 193/200 on
rhythm, and 152/200 vs 0/200 on anomaly detection. The synthetic matrix could
not detect the period change at all. This contradicts the earlier "biology is
not the active ingredient" finding for classification - there, larval and
synthetic were interchangeable. For time-dependent tasks, the real wiring is
different in kind, not just degree.

## Size co-sweep — small synthetic MATCHES at the right spectral radius

Measured spectral radii: real larval W/127 = **0.3596**; the synthetic
generator's default = **0.9** (2.5x too high).

Best config per size (blink3, bar = real-2952's 100% / 17.8 steps):

| size | spectral radius | weight distribution | success | steps |
|---:|---:|---|---:|---:|
| 512 | 0.5 | lognormal | 100% | **18.2** |
| 1,024 | 0.5 | lognormal | 100% | 18.2 |
| 2,048 | 0.5 | lognormal | 100% | 18.0 |
| 512 (blink8 validation) | 0.5 | lognormal | 100% | 37.6 |

**512 neurons at spectral radius 0.5 with lognormal weights matches the real
2,952-neuron connectome exactly** - and holds at blink8 (100% / 37.6). Success
is monotone decreasing in spectral radius (mean 0.999 at 0.5 down to 0.652 at
2.0), so the lane-1 hardening failure at 512 was an artifact of the generator's
default rho=0.9, not of size.

The deployable artifact is now **13 of 54 configurations** that pass the
bar, cheapest at 512 neurons / ~0.1 MB projection (versus 12.1 MB for the
current 2,952-neuron build) - roughly 100x cheaper than the previous estimate
of what was needed.

## What this changes

- **The "512 neurons fails" conclusion from the lane-1 hardening is retracted.**
  It failed at the default spectral radius, not at small size.
- **The connectome-vs-synthetic question reopens for temporal tasks.** On
  classification they were interchangeable; on rhythm and anomaly detection the
  real connectome detected what the synthetic could not. One task, one corpus,
  one seed - so this is a lead, not a proof, but it is the first lead.
- **The cheapest deployable artifact is now 512 neurons / rho 0.5 / lognormal,
  license-free, at about 1/12 the current size** and it matches the real
  connectome on the phototaxis task at both blink3 and blink8.

## What is still untested

- Whether the rhythm advantage holds at longer periods with more training
  episodes (P8 failed for all heads; more data may fix it).
- Whether the anomaly-detection advantage is specific to the larval graph's
  recurrent loops, or specific to this one anomaly type.
- A real-signal version (audio, sensor stream) of the rhythm task.
