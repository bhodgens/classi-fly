"""Lane 3, task 1: rhythm/interval detection on the Arena's blink metronome.

The Arena (frozen harness, world.py) in mode='blink' shows the light only on
steps t with t % P == 0 - a metronome with period P. The controller's motor
output does not influence when the light blinks (position changes only the
distance/bearing encodings), so the blink schedule is a pure clock the agent
must learn to track: detect the rhythm and act on its phase.

TASK (docs/DIRECTIONS.md lane 3, "interval/rhythm sensing"): at the step
IMMEDIATELY BEFORE each expected light onset - i.e. at every step t where
(t + 1) % P == 0 - the controller must emit thrust > 0.5 (a "pulse" on the
action's second channel). This requires knowing the phase: a memoryless map
of the current 7 sensors cannot do it, because the sensor vector on pulse
steps (t % P == P-1, blind) is statistically identical to the other blind
steps. Only a state that integrates time can hit the phase.

Success per episode (task spec):
  hit rate >= 0.8  where hit rate = correct pulses / expected pulses
  AND false pulses <= 2 per 100 steps elsewhere in the episode.
Reported aggregates: mean hit rate, episodes meeting the full criterion, mean
false pulses per 100 steps, mean latency of the first pulse per onset cycle.

Heads (see lane3_common.py): larval connectome reservoir (2952), synthetic
reservoir (2048, seed 42, fan-in 8, inhib 0.2), memoryless linear map on the
raw 7 sensors (control that cannot encode time), MemoryHeuristic (lane-1
reference; expected to fail - it carries bearing memory, not a clock).

Readout training = lane-1A two-stage ridge: states over randomized-policy
episodes, tanh-squashed output, arctanh targets, lambda swept 1..1000 and
selected on held-out validation states, refit on train+val. All seeds fixed.

Run:
  python3 tools/control/rhythm_task.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lane3_common import (  # noqa: E402
    LAMBDAS, Reservoir, ReservoirController, collect_sensor_traces,
    collect_states, fit_ridge, load_connectome, load_synthetic,
    make_projection, r2,
)
from world import Arena, MemoryHeuristic, SENSOR_DIM  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MAX_STEPS = 200
EVAL_EPISODES = 200
EVAL_SEED0 = 1000
TRAIN_SEED0 = 20000
VAL_SEED0 = 21000
POLICY_SEED0 = 30000
TRAIN_EPISODES = 120
POLICY_KINDS = ("uniform", "biased", "rotate")

PERIODS = [3, 8]

HIT_THRESHOLD = 0.8
FALSE_PULSE_BUDGET_PER_100 = 2.0
THRUST_PULSE = 0.5


# --------------------------------------------------------------------------- #
# episode rollout + scoring
# --------------------------------------------------------------------------- #
def run_rhythm_episode(controller, seed, period, max_steps=MAX_STEPS):
    """One episode; returns per-step thrust plus the onset/pulse-step masks."""
    env = Arena(seed=seed, mode="blink", blink_period=period)
    obs = env.reset()
    thrusts = np.zeros(max_steps, dtype=np.float64)
    for t in range(max_steps):
        a = controller(obs)
        thrusts[t] = float(np.clip(a[1], -1.0, 1.0))
        obs, _r, done, _info = env.step(a)
        if done:
            break
    steps = env.steps
    # step t is an ONSET step if t % P == 0 (Arena.light_visible is checked
    # against env.steps BEFORE the increment, so onset steps are exactly those).
    onset_steps = np.array([t for t in range(steps) if t % period == 0])
    pulse_steps = np.array([t for t in range(steps) if (t + 1) % period == 0
                            and t + 1 < steps])
    return {"thrusts": thrusts[:steps], "steps": steps,
            "onset_steps": onset_steps, "pulse_steps": pulse_steps}


def score_rhythm(roll, period):
    """hits, misses, false pulses, first-pulse latency per onset cycle."""
    thr = roll["thrusts"]
    is_pulse = np.zeros(roll["steps"], dtype=bool)
    is_pulse[roll["pulse_steps"]] = True
    pulses = thr > THRUST_PULSE
    hits = int(np.sum(pulses & is_pulse))
    expected = int(len(roll["pulse_steps"]))
    misses = expected - hits
    # false pulses: anywhere that is not a pulse step
    false_pulses = int(np.sum(pulses & ~is_pulse))
    false_per_100 = 100.0 * false_pulses / max(roll["steps"], 1)
    hit_rate = hits / expected if expected > 0 else 0.0
    # latency: for each cycle, steps from the pulse step to the first thrust>0.5
    latencies = []
    for ps in roll["pulse_steps"]:
        window = thr[ps:ps + period]
        fired = np.nonzero(window > THRUST_PULSE)[0]
        latencies.append(int(fired[0]) if len(fired) else period)
    success = (hit_rate >= HIT_THRESHOLD
               and false_per_100 <= FALSE_PULSE_BUDGET_PER_100)
    return {"hit_rate": round(hit_rate, 4), "hits": hits, "expected": expected,
            "misses": misses, "false_pulses": false_pulses,
            "false_per_100": round(false_per_100, 3),
            "median_latency": float(np.median(latencies)),
            "success": bool(success)}


def evaluate_rhythm(factory, period, n_episodes=EVAL_EPISODES, seed0=EVAL_SEED0):
    rolls = [run_rhythm_episode(factory(), seed0 + i, period)
             for i in range(n_episodes)]
    scores = [score_rhythm(r, period) for r in rolls]
    hit_rates = np.array([s["hit_rate"] for s in scores])
    n_ok = int(sum(s["success"] for s in scores))
    return {
        "episodes": n_episodes,
        "period": period,
        "mean_hit_rate": round(float(hit_rates.mean()), 4),
        "std_hit_rate": round(float(hit_rates.std()), 4),
        "median_hit_rate": round(float(np.median(hit_rates)), 4),
        "episodes_meeting_criterion": n_ok,
        "criterion_rate": round(n_ok / n_episodes, 4),
        "total_hits": int(sum(s["hits"] for s in scores)),
        "total_expected": int(sum(s["expected"] for s in scores)),
        "total_false_pulses": int(sum(s["false_pulses"] for s in scores)),
        "mean_false_per_100": round(float(np.mean([s["false_per_100"]
                                                   for s in scores])), 3),
        "mean_median_latency": round(float(np.mean([s["median_latency"]
                                                    for s in scores])), 2),
        "per_episode": scores,
    }


# --------------------------------------------------------------------------- #
# training-data targets
# --------------------------------------------------------------------------- #
def rhythm_targets(n_steps, period):
    """Desired thrust at each step: 1.0 on pulse steps, -1.0 elsewhere
    (sharp margin so the tanh readout's > 0.5 threshold has room)."""
    y = np.full((n_steps, 2), -1.0)   # channel 0 (turn) unused, set to -1
    for t in range(n_steps):
        if (t + 1) % period == 0:
            y[t, 1] = 1.0
    return y


def collect_rhythm_data(build_res, period, seed0, n_episodes,
                        policy_seed0=POLICY_SEED0,
                        policy_kinds=POLICY_KINDS, verbose=True):
    """Roll randomized policies; record (representation state, thrust target)."""
    from lane3_common import biased_policy_factory, random_policy_factory, \
        rotate_policy_factory
    Xs, Ys = [], []
    t0 = time.time()
    for i in range(n_episodes):
        env = Arena(seed=seed0 + i, mode="blink", blink_period=period)
        kind = policy_kinds[i % len(policy_kinds)]
        ps = policy_seed0 + i
        if kind == "uniform":
            pol = random_policy_factory(ps)
        elif kind == "biased":
            from numpy.random import default_rng
            pol = biased_policy_factory(ps, float(default_rng(ps).uniform(-0.4, 0.8)))
        else:
            pol = rotate_policy_factory(ps, period)
        res = build_res()
        obs = env.reset()
        for t in range(MAX_STEPS):
            Xs.append(res.step(obs).astype(np.float32))
            y = rhythm_targets(MAX_STEPS, period)
            Ys.append(y[t])
            obs, _r, done, _info = env.step(pol(obs))
            if done:
                break
    X = np.asarray(Xs, dtype=np.float64)
    Y = np.asarray(Ys, dtype=np.float64)[:X.shape[0]]
    if verbose:
        print(f"  rhythm data: {X.shape[0]} pairs from {n_episodes} episodes "
              f"({time.time() - t0:.1f}s)")
    return X, Y


def collect_sensor_rhythm_data(period, seed0, n_episodes,
                               policy_seed0=POLICY_SEED0,
                               policy_kinds=POLICY_KINDS):
    """Raw 7-sensor observations aligned with the same rhythm targets."""
    from lane3_common import biased_policy_factory, random_policy_factory, \
        rotate_policy_factory
    Xs, Ys = [], []
    for i in range(n_episodes):
        env = Arena(seed=seed0 + i, mode="blink", blink_period=period)
        kind = policy_kinds[i % len(policy_kinds)]
        ps = policy_seed0 + i
        if kind == "uniform":
            pol = random_policy_factory(ps)
        elif kind == "biased":
            from numpy.random import default_rng
            pol = biased_policy_factory(ps, float(default_rng(ps).uniform(-0.4, 0.8)))
        else:
            pol = rotate_policy_factory(ps, period)
        obs = env.reset()
        for t in range(MAX_STEPS):
            Xs.append(np.asarray(obs, dtype=np.float32).copy())
            y = rhythm_targets(MAX_STEPS, period)
            Ys.append(y[t])
            obs, _r, done, _info = env.step(pol(obs))
            if done:
                break
    X = np.asarray(Xs, dtype=np.float64)
    Y = np.asarray(Ys, dtype=np.float64)[:X.shape[0]]
    return X, Y


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
def make_reservoir_heads(W, proj, period, label):
    """Fit a ridge readout on reservoir states; returns (factory, fit-info)."""
    Xtr, Ytr = collect_rhythm_data(lambda: Reservoir(W, proj), period,
                                   TRAIN_SEED0, TRAIN_EPISODES)
    Xva, Yva = collect_rhythm_data(lambda: Reservoir(W, proj), period,
                                   VAL_SEED0, 10)
    Wout, bout, info = fit_ridge(Xtr, Ytr, Xva, Yva, lambdas=LAMBDAS, label=label)
    # in-sample sanity: how well does the readout separate pulse from non-pulse?
    pred = np.tanh(Xtr @ Wout.T + bout)[:, 1]
    info["train_r2_thrust"] = r2(Ytr[:, 1], pred)

    def factory():
        return ReservoirController(Reservoir(W, proj), Wout, bout,
                                   squash_fn=lambda v: np.tanh(v))
    return factory, info


def make_direct_head(period, label):
    Xtr, Ytr = collect_sensor_rhythm_data(period, TRAIN_SEED0, TRAIN_EPISODES)
    Xva, Yva = collect_sensor_rhythm_data(period, VAL_SEED0, 10)
    Wout, bout, info = fit_ridge(Xtr, Ytr, Xva, Yva, lambdas=LAMBDAS, label=label)
    pred = np.tanh(Xtr @ Wout.T + bout)[:, 1]
    info["train_r2_thrust"] = r2(Ytr[:, 1], pred)

    def factory():
        return (lambda obs: np.tanh(Wout @ np.asarray(obs, dtype=np.float64) + bout))
    return factory, info


def make_memory_heuristic_head(period):
    """MemoryHeuristic emits thrust>0 only when it thinks it is aligned; it has
    no clock, so it is a reference, not a competitor, on this task."""
    def factory():
        return MemoryHeuristic(decay=1.0)
    return factory, {"note": "hand-written lane-1 memory rule; no clock, "
                             "thrust is bearing-driven, not phase-driven"}


def make_constant_thrust_head(level):
    def factory():
        return lambda obs: np.array([0.0, level])
    return factory, {"note": f"constant thrust={level}; sanity floor - hits every "
                             f"pulse step if level>{THRUST_PULSE} but fails the "
                             f"false-pulse budget, or vice versa"}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.time()
    adj, W_larv = load_connectome(REPO / "data" / "larval_adjacency.json")
    proj_larv = make_projection(adj["neurons"])
    adj_syn, W_syn = load_synthetic(seed=42, n=2048, fan_in=8, inhib_frac=0.2)
    proj_syn = make_projection(2048)
    print(f"larval: {adj['name']} n={adj['neurons']} edges={adj['edges']}")
    print(f"synthetic: {adj_syn['name']} n={adj_syn['neurons']} "
          f"edges={adj_syn['edges']}")

    results = {
        "lane": "3a - rhythm/interval detection on the Arena blink metronome",
        "repo": str(REPO),
        "task": {
            "definition": ("Arena mode='blink', period P. Pulse = thrust>0.5 "
                           "emitted at step t with (t+1)%P==0 (immediately "
                           "before each light onset)."),
            "success_criterion": (f"hit_rate >= {HIT_THRESHOLD} AND "
                                  f"false_pulses_per_100_steps <= "
                                  f"{FALSE_PULSE_BUDGET_PER_100}"),
            "pulse_threshold": THRUST_PULSE,
            "periods": PERIODS,
        },
        "protocol": {
            "eval_episodes": EVAL_EPISODES, "eval_seed0": EVAL_SEED0,
            "train_seed0": TRAIN_SEED0, "val_seed0": VAL_SEED0,
            "train_episodes": TRAIN_EPISODES, "policy_seed0": POLICY_SEED0,
            "lambdas": LAMBDAS, "reservoir": {
                "decay": 0.8, "inner_steps": 4, "proj_seed": 42,
                "proj_scale": 0.02, "proj_range": [-6, 6],
                "weight_scale": "1/127 (larval int8)"},
            "determinism": "all seeds fixed; fresh controller per episode",
        },
        "rows": [],
        "fits": {},
    }

    for period in PERIODS:
        print(f"\n=== rhythm period={period} ===")
        heads = {}
        for name, (fac, info) in {
            "larval_reservoir": make_reservoir_heads(W_larv, proj_larv, period,
                                                     f"larval/P{period}"),
            "synthetic_2048": make_reservoir_heads(W_syn, proj_syn, period,
                                                   f"synthetic/P{period}"),
            "memoryless_linear": make_direct_head(period, f"direct/P{period}"),
            "memory_heuristic": make_memory_heuristic_head(period),
            "constant_thrust_0.9": make_constant_thrust_head(0.9),
        }.items():
            heads[name] = fac
            results["fits"][f"{name}/P{period}"] = info

        for name, fac in heads.items():
            m = evaluate_rhythm(fac, period)
            results["rows"].append({"task": "rhythm", "period": period,
                                    "controller": name, **m})
            print(f"  {name:22s} hit={m['mean_hit_rate']:.3f} "
                  f"criterion={m['criterion_rate']:.3f} "
                  f"({m['episodes_meeting_criterion']}/{m['episodes']}) "
                  f"false/100={m['mean_false_per_100']:.2f} "
                  f"lat={m['mean_median_latency']:.1f}")

    # verdicts
    verdicts = {}
    for period in PERIODS:
        rows = {r["controller"]: r for r in results["rows"]
                if r["period"] == period}
        best = max(rows, key=lambda k: rows[k]["criterion_rate"])
        rates = {k: rows[k]["criterion_rate"] for k in rows}
        if rates["larval_reservoir"] > max(v for k, v in rates.items()
                                           if k != "larval_reservoir") + 0.05:
            verdicts[f"P{period}"] = "RESERVOIR WINS"
        elif (abs(rates["larval_reservoir"]
                  - rates["synthetic_2048"]) <= 0.05
              and rates["larval_reservoir"] > rates["memoryless_linear"] + 0.05):
            verdicts[f"P{period}"] = "PARITY (reservoirs beat memoryless)"
        elif rates["larval_reservoir"] <= 0.5:
            verdicts[f"P{period}"] = "ALL FAIL"
        else:
            verdicts[f"P{period}"] = "INCONCLUSIVE"
    results["verdicts"] = verdicts
    results["runtime_seconds"] = round(time.time() - t_start, 1)
    out = REPO / "tools" / "control" / "rhythm_task_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nverdicts: {json.dumps(verdicts)}")
    print(f"wrote {out} ({results['runtime_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
