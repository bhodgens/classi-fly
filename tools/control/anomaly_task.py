"""Lane 3, task 2: stream anomaly detection on the Arena's blink metronome.

The Arena (frozen harness, world.py) in mode='blink' shows the light only on
steps t with t % P == 0 - a regular metronome. In this task the metronome
itself is the signal: for the first portion of the episode the light blinks
with period P; at a hidden switch step in the LAST THIRD of the episode the
period doubles to 2P. The controller must detect the anomaly by emitting
thrust < -0.5 (a distinct "flag" signature on the action's second channel)
for a window after the switch.

Reported per head (200 episodes, seeds 1000-1199):
  detection_latency  - steps from the switch to the first flag, per episode
                       (max_steps cap when never flagged; reported in steps
                       and as episodes_flagged / median latency).
  false_alarm_rate   - flag rate per 100 steps BEFORE the switch (must be
                       low - flagging the regular metronome is the failure
                       mode this task is designed to expose).

The period change is mechanically undetectable from any single sensor frame
(the onsets look identical, just further apart), so a memoryless linear map
cannot beat chance here - same control as lane 1's blink8 collapse.

Run:
  python3 tools/control/anomaly_task.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lane3_common import (  # noqa: E402
    LAMBDAS, Reservoir, ReservoirController, fit_ridge, load_connectome,
    load_synthetic, make_projection, r2,
)
from world import Arena  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MAX_STEPS = 200
EVAL_EPISODES = 200
EVAL_SEED0 = 1000
TRAIN_SEED0 = 20000
VAL_SEED0 = 21000
POLICY_SEED0 = 30000
TRAIN_EPISODES = 120

PERIODS = [3, 8]
FLAG_THRUST = -0.5           # thrust < -0.5 counts as the anomaly flag
FLAG_WINDOW_AFTER_CHANGE = 2 * 8   # "a window after the change": 16 steps
MIN_PRE_CHANGE_STEPS = 60          # switch never earlier than this
# Training-target window for the flag. Longer than the eval scoring window:
# a phase estimate that re-locks one full new-cycle late still lands inside
# the scoring window, so the target teaches the (state -> "period changed")
# mapping rather than penalizing a fast-but-slightly-late detection.
TRAIN_WINDOW_AFTER_CHANGE = 3 * 8   # 24 steps


def switch_step_for_seed(seed, period):
    """Deterministic hidden switch step in the last third of the episode."""
    rng = np.random.default_rng(seed * 7919 + period)   # independent of everything
    lo = max(MIN_PRE_CHANGE_STEPS, 2 * MAX_STEPS // 3)
    return int(rng.integers(lo, MAX_STEPS))            # in [133, 199]


# --------------------------------------------------------------------------- #
# episode construction (a thin wrapper around the frozen Arena)
# --------------------------------------------------------------------------- #
class AnomalyArena:
    """Period-P blink that switches to 2P at a hidden step; everything else
    (physics, sensors, seeds) is the frozen Arena untouched - we just swap
    env.blink_period mid-episode."""

    def __init__(self, seed, period):
        self.env = Arena(seed=seed, mode="blink", blink_period=period)
        self.period = period
        self.switch_step = switch_step_for_seed(seed, period)

    def reset(self):
        return self.env.reset()

    @property
    def steps(self):
        return self.env.steps

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        if self.env.steps == self.switch_step:
            self.env.blink_period = 2 * self.period
        return obs, r, done, info


def run_anomaly_episode(controller, seed, period, max_steps=MAX_STEPS):
    env = AnomalyArena(seed=seed, period=period)
    obs = env.reset()
    flags = np.zeros(max_steps, dtype=bool)
    for t in range(max_steps):
        a = controller(obs)
        flags[t] = float(np.clip(a[1], -1.0, 1.0)) < FLAG_THRUST
        obs, _r, done, _info = env.step(a)
        if done:
            break
    steps = env.steps
    sw = env.switch_step
    pre = flags[:sw]
    post = flags[sw:steps]
    flagged_idx = np.nonzero(post)[0]
    latency = int(flagged_idx[0]) if len(flagged_idx) else (steps - sw)
    return {"flags": flags[:steps], "steps": steps, "switch_step": sw,
            "latency": latency, "flagged": bool(len(flagged_idx)),
            "pre_flags": int(pre.sum()),
            "pre_rate_per_100": round(100.0 * pre.sum() / max(sw, 1), 2),
            "post_flags": int(post.sum()),
            "window_flags": int(post[:FLAG_WINDOW_AFTER_CHANGE].sum())}


def evaluate_anomaly(factory, period, n_episodes=EVAL_EPISODES, seed0=EVAL_SEED0):
    rolls = [run_anomaly_episode(factory(), seed0 + i, period)
             for i in range(n_episodes)]
    lat = np.array([r["latency"] for r in rolls], dtype=float)
    return {
        "episodes": n_episodes,
        "period": period,
        "switch_steps": [r["switch_step"] for r in rolls],
        "episodes_flagged": int(sum(r["flagged"] for r in rolls)),
        "flag_rate": round(float(np.mean([r["flagged"] for r in rolls])), 4),
        "median_latency_when_flagged": (float(np.median([r["latency"] for r in rolls
                                                         if r["flagged"]]))
                                        if any(r["flagged"] for r in rolls) else None),
        "mean_latency": round(float(lat.mean()), 1),
        "total_pre_switch_flags": int(sum(r["pre_flags"] for r in rolls)),
        "mean_pre_rate_per_100": round(float(np.mean([r["pre_rate_per_100"]
                                                      for r in rolls])), 3),
        "mean_window_flags": round(float(np.mean([r["window_flags"]
                                                  for r in rolls])), 2),
        "per_episode": [{"switch_step": r["switch_step"], "flagged": r["flagged"],
                         "latency": r["latency"],
                         "pre_rate_per_100": r["pre_rate_per_100"]}
                        for r in rolls],
    }


# --------------------------------------------------------------------------- #
# training targets
# --------------------------------------------------------------------------- #
def anomaly_targets(n_steps, period, seed):
    """Desired thrust: -1.0 (flag) for FLAG_WINDOW_AFTER_CHANGE steps after
    the switch, +1.0 elsewhere (sharp margin around the -0.5 threshold)."""
    y = np.ones((n_steps, 2))
    sw = switch_step_for_seed(seed, period)
    y[sw:min(sw + TRAIN_WINDOW_AFTER_CHANGE, n_steps), 1] = -1.0
    y[:, 0] = 0.0   # turn channel unused
    return y


def collect_anomaly_data(build_res, period, seed0, n_episodes,
                         policy_seed0=POLICY_SEED0,
                         policy_kinds=("uniform", "biased", "rotate"),
                         use_reservoir=True, verbose=True):
    from lane3_common import (biased_policy_factory, random_policy_factory,
                              rotate_policy_factory)
    Xs, Ys = [], []
    t0 = time.time()
    for i in range(n_episodes):
        env = AnomalyArena(seed=seed0 + i, period=period)
        kind = policy_kinds[i % len(policy_kinds)]
        ps = policy_seed0 + i
        if kind == "uniform":
            pol = random_policy_factory(ps)
        elif kind == "biased":
            from numpy.random import default_rng
            pol = biased_policy_factory(ps, float(default_rng(ps).uniform(-0.4, 0.8)))
        else:
            pol = rotate_policy_factory(ps, period)
        res = build_res() if use_reservoir else None
        obs = env.reset()
        y = anomaly_targets(MAX_STEPS, period, seed0 + i)
        for t in range(MAX_STEPS):
            if use_reservoir and res is not None:
                Xs.append(res.step(obs).astype(np.float32))
            else:
                Xs.append(np.asarray(obs, dtype=np.float32).copy())
            Ys.append(y[t])
            obs, _r, done, _info = env.step(pol(obs))
            if done:
                break
    X = np.asarray(Xs, dtype=np.float64)
    Y = np.asarray(Ys, dtype=np.float64)[:X.shape[0]]
    if verbose:
        print(f"  anomaly data: {X.shape[0]} pairs from {n_episodes} episodes "
              f"({time.time() - t0:.1f}s) [reservoir={use_reservoir}]")
    return X, Y


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
def make_reservoir_heads(W, proj, period, label):
    Xtr, Ytr = collect_anomaly_data(lambda: Reservoir(W, proj), period,
                                    TRAIN_SEED0, TRAIN_EPISODES,
                                    use_reservoir=True)
    Xva, Yva = collect_anomaly_data(lambda: Reservoir(W, proj), period,
                                    VAL_SEED0, 10, use_reservoir=True)
    Wout, bout, info = fit_ridge(Xtr, Ytr, Xva, Yva, lambdas=LAMBDAS, label=label)
    pred = np.tanh(Xtr @ Wout.T + bout)[:, 1]
    info["train_r2_thrust"] = r2(Ytr[:, 1], pred)

    def factory():
        return ReservoirController(Reservoir(W, proj), Wout, bout,
                                   squash_fn=lambda v: np.tanh(v))
    return factory, info


def make_direct_head(period, label):
    Xtr, Ytr = collect_anomaly_data(None, period, TRAIN_SEED0, TRAIN_EPISODES,
                                    use_reservoir=False)
    Xva, Yva = collect_anomaly_data(None, period, VAL_SEED0, 10,
                                    use_reservoir=False)
    Wout, bout, info = fit_ridge(Xtr, Ytr, Xva, Yva, lambdas=LAMBDAS, label=label)
    pred = np.tanh(Xtr @ Wout.T + bout)[:, 1]
    info["train_r2_thrust"] = r2(Ytr[:, 1], pred)

    def factory():
        return lambda obs: np.tanh(Wout @ np.asarray(obs, dtype=np.float64) + bout)
    return factory, info


def make_constant_flag_head():
    def factory():
        return lambda obs: np.array([0.0, -1.0])
    return factory, {"note": "always-flag ceiling: latency 0 everywhere but "
                             "pre_rate_per_100 = 100 (worst false alarms)"}


def make_never_flag_head():
    def factory():
        return lambda obs: np.array([0.0, 1.0])
    return factory, {"note": "never-flag floor: zero false alarms, never detects"}


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
        "lane": "3b - stream anomaly detection (blink period change P -> 2P)",
        "repo": str(REPO),
        "task": {
            "definition": ("Arena mode='blink' with period P; at a hidden "
                           "switch step in the last third of the episode the "
                           "period doubles to 2P. Flag = thrust < "
                           f"{FLAG_THRUST} within "
                           f"{FLAG_WINDOW_AFTER_CHANGE} steps after the "
                           "switch."),
            "switch_rule": (f"seed-dependent, in [{max(MIN_PRE_CHANGE_STEPS, 2 * MAX_STEPS // 3)}, "
                            f"{MAX_STEPS}) via rng(seed*7919+period); deterministic"),
            "metrics": ("detection latency (steps from switch to first flag), "
                        "false-alarm rate per 100 steps before the switch"),
        },
        "protocol": {
            "eval_episodes": EVAL_EPISODES, "eval_seed0": EVAL_SEED0,
            "train_seed0": TRAIN_SEED0, "val_seed0": VAL_SEED0,
            "train_episodes": TRAIN_EPISODES, "policy_seed0": POLICY_SEED0,
            "lambdas": LAMBDAS,
            "reservoir": {"decay": 0.8, "inner_steps": 4, "proj_seed": 42,
                          "proj_scale": 0.02, "proj_range": [-6, 6],
                          "weight_scale": "1/127 (larval int8)"},
        },
        "rows": [],
        "fits": {},
    }

    for period in PERIODS:
        print(f"\n=== anomaly period={period} -> 2P ===")
        heads = {}
        for name, (fac, info) in {
            "larval_reservoir": make_reservoir_heads(W_larv, proj_larv, period,
                                                     f"larval/P{period}"),
            "synthetic_2048": make_reservoir_heads(W_syn, proj_syn, period,
                                                   f"synthetic/P{period}"),
            "memoryless_linear": make_direct_head(period, f"direct/P{period}"),
            "always_flag_ceiling": make_constant_flag_head(),
            "never_flag_floor": make_never_flag_head(),
        }.items():
            heads[name] = fac
            results["fits"][f"{name}/P{period}"] = info

        for name, fac in heads.items():
            m = evaluate_anomaly(fac, period)
            results["rows"].append({"task": "anomaly", "period": period,
                                    "controller": name, **m})
            print(f"  {name:22s} flagged={m['flag_rate']:.3f} "
                  f"med_lat={m['median_latency_when_flagged']} "
                  f"pre/100={m['mean_pre_rate_per_100']:.2f} "
                  f"window_flags={m['mean_window_flags']:.2f}")

    # verdicts
    verdicts = {}
    for period in PERIODS:
        rows = {r["controller"]: r for r in results["rows"]
                if r["period"] == period}
        # a useful detector: high flag rate AND low pre-switch false alarms
        def score(r):
            return r["flag_rate"] - min(r["mean_pre_rate_per_100"] / 100.0, 1.0)
        best = max(rows, key=lambda k: score(rows[k]))
        if rows[best]["flag_rate"] < 0.5:
            verdicts[f"P{period}"] = "ALL FAIL"
        elif best in ("larval_reservoir", "synthetic_2048"):
            other = ("synthetic_2048" if best == "larval_reservoir"
                     else "larval_reservoir")
            if abs(score(rows[best]) - score(rows[other])) < 0.05:
                verdicts[f"P{period}"] = "PARITY (reservoirs win over memoryless)"
            else:
                verdicts[f"P{period}"] = f"RESERVOIR WINS ({best})"
        else:
            verdicts[f"P{period}"] = f"MEMORYLESS WINS ({best})"
    results["verdicts"] = verdicts
    results["runtime_seconds"] = round(time.time() - t_start, 1)
    out = REPO / "tools" / "control" / "anomaly_task_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nverdicts: {json.dumps(verdicts)}")
    print(f"wrote {out} ({results['runtime_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
