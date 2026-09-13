"""Lane 1A: can the fixed larval-connectome reservoir CONTROL an agent?

docs/DIRECTIONS.md lane 1 ("fixed-graph control loop", ranked first) and the
third question in the frozen harness docstring:

  1. Can a fixed connectome plus a trained output layer control the agent?
  2. Does it beat a trivial hand-written rule (turn toward the light)?
  3. Does the connectome help beyond a DIRECT LINEAR MAP from the same 8
     sensors to the same 2 motors, fitted by the same procedure on the same
     data? If not, the graph adds nothing - consistent with every
     classification result in docs/EXPERIMENTS.md.

Answering (3) is the point of this file. The ablation is exact: identical
(targets, ridge sweep, selection protocol, data) - only the representation the
readout sees differs (reservoir states 2952-d vs raw sensors 8-d).

Method
------
Teacher = the hand-written reference policy (heuristic_action in open mode,
MemoryHeuristic in blink mode). We collect (state, teacher_action) pairs by
running the teacher in randomized episodes with seeds OUTSIDE the evaluation
range, then fit a ridge readout.

Two representations share one procedure:
  reservoir : s_t = tanh(decay*s_{t-1} + W@s_{t-1} + proj^T obs_t), 4 inner
              steps per world step, state carried across world steps and reset
              at episode start (a stateful controller -> run_many_factory).
  direct    : the raw 8 sensors (memoryless -> run_many).
Both are re-fit per mode (the teacher differs between open and blink).

Targets are squashed with tanh at the output, so the ridge is fitted in the
inverse-squash space: t = arctanh(clip(a, -0.995, 0.995)) and the controller
emits tanh(W s + b). Ridge lambda is swept over 1e-3..100 and selected by MSE
on HELD-OUT VALIDATION EPISODES (seeds disjoint from both training and
evaluation), never on the evaluation arenas. This matters: EXPERIMENTS.md's
most expensive error was an unregularized 2952-wide readout.

Third row: a RANDOM FIXED DECODER (reservoir states -> untrained random linear
map, gain set so the pre-squash output has unit-ish scale). Lower bound.

Run:
  python3 tools/control/reservoir_imitation.py
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "control"))

from world import (  # noqa: E402
    ACTION_DIM, MAX_STEPS, SENSOR_DIM, Arena, MemoryHeuristic, heuristic_action,
    run_episode, run_many, run_many_factory, summarize,
)

# --- reservoir construction (the project-standard one) ----------------------
DECAY = 0.8
INNER_STEPS = 4
PROJ_SEED = 42
PROJ_LO, PROJ_HI = -6, 6          # int8 projection
PROJ_SCALE = 0.02
WEIGHT_SCALE = 1.0 / 127.0        # signed int8 connectome weights -> float

LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SQUASH_EPS = 0.005

MODES = [("open", None), ("blink3", 3), ("blink8", 8)]


# --------------------------------------------------------------------------- #
# reservoir
# --------------------------------------------------------------------------- #
def load_connectome(path):
    adj = json.loads(Path(path).read_text())
    n = int(adj["neurons"])
    W = sparse.csr_matrix(
        (np.asarray(adj["weights"], dtype=np.float64) * WEIGHT_SCALE,
         np.asarray(adj["indices"], dtype=np.int64),
         np.asarray(adj["indptr"], dtype=np.int64)),
        shape=(n, n),
    )
    return adj, W


def make_projection(n, seed=PROJ_SEED):
    rng = np.random.default_rng(seed)
    return (rng.integers(PROJ_LO, PROJ_HI + 1, size=(SENSOR_DIM, n)).astype(np.float64)
            * PROJ_SCALE)


def obs_dim():
    """The REAL observation width.

    world.py declares SENSOR_DIM = 8 but obs_vector() returns 7 values
    (4 light sensors + hx, hy, wall). world.py is frozen, so we measure the
    actual width instead of trusting the constant, and slice the specified
    (SENSOR_DIM, neurons) projection down to it. Reported in the JSON.
    """
    from world import obs_vector as _ov
    return int(len(_ov(np.zeros(2), 0.0, np.array([1.0, 0.0]))))


class Reservoir:
    """Fixed-graph recurrent substrate. State persists across world steps."""

    def __init__(self, W, proj, decay=DECAY, steps=INNER_STEPS):
        self.W = W
        self.proj = proj
        self.decay = decay
        self.steps = steps
        self.n = W.shape[0]
        self.s = np.zeros(self.n, dtype=np.float64)

    def reset(self):
        self.s = np.zeros(self.n, dtype=np.float64)

    def step(self, obs):
        drive = self.proj[:obs.shape[0]].T @ np.asarray(obs, dtype=np.float64)
        s = self.s
        for _ in range(self.steps):
            s = np.tanh(self.decay * s + (self.W @ s) + drive)
        self.s = s
        return s


class ReservoirController:
    """Fresh per episode: reservoir state must not leak between arenas."""

    def __init__(self, W, proj, Wout, bout):
        self.res = Reservoir(W, proj)
        self.Wout = Wout
        self.bout = bout

    def __call__(self, obs):
        s = self.res.step(obs)
        return np.tanh(self.Wout @ s + self.bout)


# --------------------------------------------------------------------------- #
# teachers and data collection
# --------------------------------------------------------------------------- #
def make_teacher(mode):
    return heuristic_action if mode == "open" else MemoryHeuristic(decay=1.0)


def collect_pairs(W, proj, mode, blink_period, seed0, n_episodes, target_pairs,
                  verbose=True):
    """Run the teacher in randomized episodes; record (representation, action)."""
    X_res, X_dir, Y = [], [], []
    t0 = time.time()
    seen = 0
    i = -1
    for i in range(n_episodes):
        env = Arena(seed=seed0 + i, mode=mode, blink_period=blink_period or 3)
        teacher = make_teacher(mode)
        res = Reservoir(W, proj)
        obs = env.reset()
        for _ in range(MAX_STEPS):
            s = res.step(obs)
            a = teacher(obs)
            X_res.append(s.astype(np.float32))
            X_dir.append(np.asarray(obs, dtype=np.float32).copy())
            Y.append(np.asarray(a, dtype=np.float32).copy())
            seen += 1
            obs, _r, done, _info = env.step(a)
            if done:
                break
        if seen >= target_pairs:
            break
    X = (np.asarray(X_res, dtype=np.float64), np.asarray(X_dir, dtype=np.float64))
    Yv = np.asarray(Y, dtype=np.float64)
    if verbose:
        print(f"  collected {len(Yv)} pairs from {i + 1} episodes "
              f"({time.time() - t0:.1f}s)")
    return X, Yv


def to_targets(actions):
    """Inverse of the output squash: tanh(readout) should reproduce `actions`."""
    return np.arctanh(np.clip(actions, -1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS))


# --------------------------------------------------------------------------- #
# ridge readout (eigen-decomposed Gram so the lambda sweep is cheap)
# --------------------------------------------------------------------------- #
def gram(X, Y, xm, ym):
    """Centered moments of (X, Y) about means xm/ym, computed from raw sums."""
    n = X.shape[0]
    Xc = X - xm
    Yc = Y - ym
    return Xc.T @ Xc, Xc.T @ Yc, float(np.sum(Yc * Yc)), n


class RidgeFit:
    def __init__(self, G, C, syy, n):
        self.g, self.V = np.linalg.eigh(G)
        self.VtC = self.V.T @ C
        self.syy = syy
        self.n = n
        self.yc2 = syy

    def weights(self, lam):
        return (self.V @ (self.VtC / (self.g + lam)[:, None])).T   # (2, d)

    def sse(self, W, G, C, syy):
        # ||Y - X W^T||^2 from moments; no need to materialise predictions
        a = float(np.trace(W @ C))
        b = float(np.trace(W @ G @ W.T))
        return syy - 2.0 * a + b


def fit_representation(Xtr, Ytr, Xva, Yva, lambdas=LAMBDAS, verbose=True, label=""):
    """Sweep lambda on train->val MSE, then refit on train+val at the best one."""
    Yt = to_targets(Ytr)
    xm, ym = Xtr.mean(axis=0), Yt.mean(axis=0)
    G, C, syy, n = gram(Xtr, Yt, xm, ym)
    fit = RidgeFit(G, C, syy, n)
    sweep = []
    for lam in lambdas:
        W = fit.weights(lam)
        # validation, centered with the TRAIN means
        Gv, Cv, syyv, nv = gram(Xva, to_targets(Yva), xm, ym)
        sse = fit.sse(W, Gv, Cv, syyv)
        sweep.append({"lambda": lam, "val_mse": round(sse / (nv * ACTION_DIM), 6)})
    best = min(sweep, key=lambda r: r["val_mse"])
    lam = best["lambda"]

    # final fit on train + validation at the selected penalty (all data)
    Xa = np.vstack([Xtr, Xva])
    Ya = np.vstack([Ytr, Yva])
    Yta = to_targets(Ya)
    xma, yma = Xa.mean(axis=0), Yta.mean(axis=0)
    Ga, Ca, syya, na = gram(Xa, Yta, xma, yma)
    fa = RidgeFit(Ga, Ca, syya, na)
    W = fa.weights(lam)
    b = yma - W @ xma
    if verbose:
        print(f"  [{label}] best lambda={lam} val_mse={best['val_mse']} "
              f"| sweep " + " ".join(f"{r['lambda']:g}:{r['val_mse']:.4f}" for r in sweep))
    return W, b, {"best_lambda": lam, "val_mse": best["val_mse"], "sweep": sweep,
                  "pairs_train": int(Xtr.shape[0]), "pairs_val": int(Xva.shape[0])}


def random_readout(X, seed=7, target_std=0.7, out_dim=ACTION_DIM):
    """Untrained fixed decoder, gain matched to the trained readout's scale."""
    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(out_dim, X.shape[1]))
    pre = X @ W0.T
    std = float(np.std(pre))
    W = W0 * (target_std / max(std, 1e-12))
    b = rng.uniform(-0.2, 0.2, size=out_dim)
    return W, b, {"seed": seed, "target_pre_std": target_std,
                  "raw_pre_std": round(std, 6),
                  "gain": round(target_std / max(std, 1e-12), 6)}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(W, proj, mode, blink_period, n_episodes, seed0, Wout, bout,
             stateful) -> dict:
    if stateful:
        return run_many_factory(lambda: ReservoirController(W, proj, Wout, bout),
                                n_episodes=n_episodes, seed0=seed0,
                                mode=mode, blink_period=blink_period or 3)
    ctl = lambda obs: np.tanh(Wout @ obs + bout)  # noqa: E731 (memoryless map)
    return run_many(ctl, n_episodes=n_episodes, seed0=seed0, mode=mode,
                    blink_period=blink_period or 3)


def per_episode_steps(W, proj, mode, bp, Wout, bout, stateful, n, seed0):
    """Steps per episode, NaN where the controller failed (so means match summarize)."""
    out = []
    for i in range(n):
        if stateful:
            ctl = ReservoirController(W, proj, Wout, bout)
        else:
            ctl = lambda obs: np.tanh(Wout @ obs + bout)  # noqa: E731
        r = run_episode(ctl, seed=seed0 + i, mode=mode, blink_period=bp or 3)
        out.append(r["steps"] if r["success"] else np.nan)
    return np.asarray(out)


def paired_report(a, b, n_boot=4000, seed=0):
    """Paired comparison on episodes where BOTH controllers succeeded."""
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() == 0:
        return {"n_paired": 0}
    diff = a[m] - b[m]
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(rng.choice(diff, size=len(diff), replace=True))
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_paired": int(m.sum()),
            "mean_delta_steps": round(float(diff.mean()), 2),
            "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "faster": int(np.sum(diff < 0)), "tie": int(np.sum(diff == 0)),
            "slower": int(np.sum(diff > 0)),
            "significant": bool(lo > 0 or hi < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adjacency", default=str(REPO / "data" / "larval_adjacency.json"))
    ap.add_argument("--out", default=str(REPO / "tools" / "control"
                                        / "reservoir_imitation_results.json"))
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--eval-seed0", type=int, default=1000)
    ap.add_argument("--train-seed0", type=int, default=5000)
    ap.add_argument("--val-seed0", type=int, default=6000)
    ap.add_argument("--train-pairs", type=int, default=10000)
    ap.add_argument("--val-pairs", type=int, default=2000)
    ap.add_argument("--max-episodes", type=int, default=2000)
    args = ap.parse_args()

    t_start = time.time()
    adj, W = load_connectome(args.adjacency)
    proj = make_projection(adj["neurons"])
    print(f"connectome: {adj['name']} neurons={adj['neurons']} edges={adj['edges']} "
          f"({adj.get('license', '')}) ; decay={DECAY} inner_steps={INNER_STEPS} "
          f"proj=int{PROJ_LO}..{PROJ_HI}*{PROJ_SCALE} seed={PROJ_SEED}")

    results = {
        "lane": "1A - fixed-graph reservoir control via imitation of a hand-written teacher",
        "repo": str(REPO),
        "connectome": {"name": adj["name"], "neurons": int(adj["neurons"]),
                       "edges": int(adj["edges"]), "license": adj.get("license", ""),
                       "weight_scale": WEIGHT_SCALE},
        "reservoir": {"decay": DECAY, "inner_steps": INNER_STEPS,
                      "proj_range": [PROJ_LO, PROJ_HI], "proj_scale": PROJ_SCALE,
                      "proj_seed": PROJ_SEED, "state_carried_across_world_steps": True,
                      "reset_per_episode": True,
                      "output_squash": "tanh",
                      "target_transform": f"arctanh(clip(a, +-{1 - SQUASH_EPS}))"},
        "sensor_dim_declared": SENSOR_DIM,
        "sensor_dim_actual": obs_dim(),
        "sensor_dim_note": ("world.py declares SENSOR_DIM=8 but obs_vector() returns 7 "
                            "values (4 light + hx, hy, wall). The specified (8, neurons) "
                            "projection is drawn but only its first 7 rows are used."),
        "lambdas_swept": LAMBDAS,
        "eval": {"episodes": args.eval_episodes, "seed0": args.eval_seed0,
                 "train_seed0": args.train_seed0, "val_seed0": args.val_seed0},
        "mode_notes": {"open": "light always visible (fully observable)",
                       "blink3": "light sensors zeroed except every 3rd step",
                       "blink8": "light sensors zeroed except every 8th step"},
        "metric_note": ("mean_steps_to_success follows world.summarize: it averages "
                        "SUCCESSFUL episodes only, so read it together with "
                        "success_rate/timeouts. Success is saturated for the teacher, "
                        "which is why steps is the discriminating metric."),
        "rows": [],
        "fits": {},
    }

    for mode, bp in MODES:
        tag = mode
        print(f"\n=== mode={tag} (blink_period={bp}) ===")
        (Xtr_res, Xtr_dir), Ytr = collect_pairs(
            W, proj, mode, bp, args.train_seed0, args.max_episodes, args.train_pairs)
        (Xva_res, Xva_dir), Yva = collect_pairs(
            W, proj, mode, bp, args.val_seed0, 500, args.val_pairs)

        fits = {}
        for rep, Xtr, Xva in (("reservoir", Xtr_res, Xva_res),
                              ("direct", Xtr_dir, Xva_dir)):
            Wout, bout, info = fit_representation(Xtr, Ytr, Xva, Yva, label=f"{tag}/{rep}")
            fits[rep] = {"readout": {"Wout": Wout.tolist(), "bout": bout.tolist()}, **info}

        Wr, br, rinfo = random_readout(Xtr_res)
        fits["reservoir_random"] = rinfo

        # --- evaluation on the frozen runners ---
        ev = {}
        ev["reservoir_imitation"] = evaluate(
            W, proj, mode, bp, args.eval_episodes, args.eval_seed0,
            np.asarray(fits["reservoir"]["readout"]["Wout"]),
            np.asarray(fits["reservoir"]["readout"]["bout"]), stateful=True)
        ev["direct_linear_map"] = evaluate(
            W, proj, mode, bp, args.eval_episodes, args.eval_seed0,
            np.asarray(fits["direct"]["readout"]["Wout"]),
            np.asarray(fits["direct"]["readout"]["bout"]), stateful=False)
        ev["reservoir_random_decoder"] = evaluate(
            W, proj, mode, bp, args.eval_episodes, args.eval_seed0, Wr, br, stateful=True)
        ev["teacher_heuristic"] = (
            run_many(heuristic_action, args.eval_episodes, seed0=args.eval_seed0,
                     mode=mode, blink_period=bp or 3)
            if mode == "open" else
            run_many_factory(MemoryHeuristic, args.eval_episodes, seed0=args.eval_seed0,
                             mode=mode, blink_period=bp or 3))

        results["fits"][tag] = fits
        for name, m in ev.items():
            results["rows"].append({"mode": tag, "blink_period": bp, "controller": name,
                                    **m})
            print(f"  {name:26s} {json.dumps(m)}")

        # --- paired comparison (same arenas; only episodes where both succeeded) ---
        Wr2 = np.asarray(fits["reservoir"]["readout"]["Wout"])
        br2 = np.asarray(fits["reservoir"]["readout"]["bout"])
        Wd2 = np.asarray(fits["direct"]["readout"]["Wout"])
        bd2 = np.asarray(fits["direct"]["readout"]["bout"])
        a = per_episode_steps(W, proj, mode, bp, Wr2, br2, True,
                              args.eval_episodes, args.eval_seed0)
        d = per_episode_steps(W, proj, mode, bp, Wd2, bd2, False,
                             args.eval_episodes, args.eval_seed0)
        if mode == "open":
            t = np.array([run_episode(heuristic_action, seed=args.eval_seed0 + i,
                                      mode=mode)["steps"]
                          for i in range(args.eval_episodes)], dtype=float)
        else:
            t = []
            for i in range(args.eval_episodes):
                r = run_episode(MemoryHeuristic(), seed=args.eval_seed0 + i, mode=mode,
                                blink_period=bp or 3)
                t.append(r["steps"] if r["success"] else np.nan)
            t = np.asarray(t, dtype=float)
        results.setdefault("paired", {})[tag] = {
            "reservoir_vs_teacher": paired_report(a, t),
            "reservoir_vs_direct": paired_report(a, d),
            "note": ("delta = reservoir steps - other steps, on the episodes where both "
                     "succeeded; negative = reservoir faster; 4000-sample paired bootstrap"),
        }
        print(f"  paired: {json.dumps(results['paired'][tag])}")

    # --- verdict: matched-mode reservoir vs direct map ---
    def get(mode, ctl):
        return next(r for r in results["rows"] if r["mode"] == mode and r["controller"] == ctl)

    verdicts = {}
    for mode, _bp in MODES:
        r = get(mode, "reservoir_imitation")["mean_steps_to_success"]
        d = get(mode, "direct_linear_map")["mean_steps_to_success"]
        h = get(mode, "teacher_heuristic")["mean_steps_to_success"]
        rr = get(mode, "reservoir_imitation")["success_rate"]
        dr = get(mode, "direct_linear_map")["success_rate"]
        if r is None or d is None:
            # one side never succeeded: fall back to the success rates
            if rr > dr + 0.05:
                v = "RESERVOIR HELPS"
            elif dr > rr + 0.05:
                v = "WORSE"
            else:
                v = "INCONCLUSIVE"
        elif abs(r - d) <= 0.05 * d:
            v = "PARITY"
        elif r < d:
            v = "RESERVOIR HELPS"
        else:
            v = "WORSE"
        verdicts[mode] = {
            "verdict": v,
            "reservoir_vs_direct_steps": [r, d],
            "reservoir_delta_steps_vs_direct": None if (r is None or d is None) else round(r - d, 1),
            "reservoir_vs_direct_success": [rr, dr],
            "teacher_heuristic_steps": h,
            "gap_reservoir_vs_teacher_steps": None if (r is None or h is None) else round(r - h, 1),
        }
    results["verdicts"] = verdicts
    results["runtime_seconds"] = round(time.time() - t_start, 1)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nverdicts: {json.dumps(verdicts)}")
    print(f"wrote {args.out} in {results['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
