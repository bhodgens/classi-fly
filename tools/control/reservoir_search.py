#!/usr/bin/env python3
"""Lane 1B - reservoir output decoder trained by REWARD SEARCH (no teacher).

The question this answers, in the bluntest possible form: *can the fixed larval
connectome learn the lane-1 phototaxis task by trial and error alone?*

Everything except the readout is frozen:

    s = tanh(0.8*s + W@s + P^T obs),  s0 = 0, 4 steps        (fixed reservoir)
    a = tanh((R @ s) @ theta)                                (R fixed, theta trained)

`theta` is trained ONLY from the environment's scalar reward
(``reward = -dist + 10 on success``) - the teacher's actions are never read,
and neither `heuristic_action` nor `MemoryHeuristic` appears in any training
path.  A teacher-forced imitation readout is therefore NOT comparable from
inside this file; the reference bars below are the hand-written *baselines*
that the task is scored against (`heuristic`, `MemoryHeuristic`, random).

Black-box optimization over the full 2952x2 decoder (5904 params) is hopeless,
so the decoder is LOW-RANK: a fixed random projection R (k x 2952, seeded)
maps states to k dims and only the k x 2 = 16..64 entries of `theta` are
searched.  The optimizer is a self-contained diagonal CMA-ES (sep-CMA-ES),
seeded and deterministic, with a fixed evaluation budget; a plain random-search
control runs on the same budget so "the search learned something" is a claim
with a floor under it.

Usage:
    python3 tools/control/reservoir_search.py --smoke      # tiny budget, sanity
    python3 tools/control/reservoir_search.py --selftest   # optimizer on toy fns
    python3 tools/control/reservoir_search.py              # full run + results

Harness note (frozen file, not modified): ``world.obs_vector`` returns 7
values while ``SENSOR_DIM`` is 8.  The input projection is built at the
specified shape (SENSOR_DIM, N) and observations are zero-padded to
SENSOR_DIM; because ``rng.integers(..., size=(8, N))`` fills row-major, the
first 7 rows are bit-identical to a (7, N) draw, so the padding is inert.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from world import (  # noqa: E402  (frozen harness)
    ACTION_DIM,
    MAX_STEPS,
    SENSOR_DIM,
    Arena,
    heuristic_action,
    MemoryHeuristic,
    random_action_factory,
    run_many,
    run_many_factory,
)

ADJ_PATH = REPO / "data" / "larval_adjacency.json"
OUT_PATH = HERE / "reservoir_search_results.json"

# --- frozen reservoir build -------------------------------------------------
RESERVOIR_STEPS = 4
LEAK = 0.8
PROJ_SEED = 42
PROJ_LOW, PROJ_HIGH = -6, 6
PROJ_SCALE = 0.02
POWER_ITER_SWEEPS = 60

# --- search protocol --------------------------------------------------------
TRAIN_SEED0 = 5000          # training arenas; disjoint from the 1000-range eval
N_TRAIN = 40                # episodes per candidate
N_CANDIDATES = 600          # total candidate decoders evaluated per search
POPSIZE = 24                # lambda: candidates per generation
EVAL_N = 200
EVAL_SEED0 = 1000
BLINK_PERIOD = 3
K_VALUES = (8, 16, 32)
TRAIN_MODES = ("open", "blink")
# weight-scale x decoder-rank grid actually searched.  "raw" is the literal
# spec build (int8 weights as shipped); "spectral0.9" first normalises the
# connectome to the spectral-radius target the repo's own ingest code uses.
SEARCH_GRID = (
    ("raw", (8, 16, 32)),
    ("spectral0.9", (8, 16, 32)),
)
CONTROL_GRID = (("raw", 16),)
# untrained theta draws (architecture floor); evaluated on the standard protocol
UNTRAINED_GRID = (("raw", (8, 16, 32)),)
UNTRAINED_DRAWS = 2
# the sibling lane-1A build dequantises int8 weights by 1/127; searching at that
# exact scale makes "did the search reach the imitation readout?" apples-to-apples
EXTRA_GRID = (("inv127", (8, 32)),)
SIBLING_RESULTS = HERE / "reservoir_imitation_results.json"


# ---------------------------------------------------------------------------
# reservoir
# ---------------------------------------------------------------------------
def load_adjacency(path=ADJ_PATH):
    """Return (n_neurons, csr_matrix W) from the larval connectome artifact."""
    with open(path) as fh:
        d = json.load(fh)
    n = int(d["neurons"])
    indptr = np.asarray(d["indptr"], dtype=np.int64)
    indices = np.asarray(d["indices"], dtype=np.int64)
    weights = np.asarray(d["weights"], dtype=np.float64)
    W = sparse.csr_matrix((weights, indices, indptr), shape=(n, n))
    return n, W


def spectral_radius(W, sweeps=POWER_ITER_SWEEPS):
    """Deterministic power-iteration estimate of |lambda_max|."""
    n = W.shape[0]
    v = np.random.default_rng(0).standard_normal(n)
    v /= np.linalg.norm(v) or 1.0
    norm = 0.0
    for _ in range(sweeps):
        v = W.dot(v)
        norm = float(np.linalg.norm(v))
        if norm <= 1e-300:
            return 0.0
        v /= norm
    return norm


def build_W(scale_name, target_radius=0.9):
    """Frozen build: raw int8 weights as shipped, or the repo's documented
    spectral-radius normalisation (tools/ingest/synthetic.py: target < 0.9)."""
    n, W = load_adjacency()
    sr = spectral_radius(W)
    if scale_name == "raw":
        return n, W, sr, 1.0
    if scale_name == "inv127":
        alpha = 1.0 / 127.0
        return n, W * alpha, sr, alpha
    if scale_name.startswith("spectral"):
        target = float(scale_name.split("spectral")[1]) or target_radius
        alpha = target / sr
        return n, W * alpha, sr, alpha
    raise ValueError(f"unknown weight scale {scale_name!r}")


def build_input_projection(n, sensor_dim=SENSOR_DIM, seed=PROJ_SEED):
    """int8 uniform in [-6, 6] * 0.02, shape (sensor_dim, n), returned as (n, d)."""
    P = np.random.default_rng(seed).integers(
        PROJ_LOW, PROJ_HIGH + 1, size=(sensor_dim, n)
    ).astype(np.float64) * PROJ_SCALE
    return np.ascontiguousarray(P.T)          # (n, sensor_dim): drive = P_T @ obs


def build_readout_projection(n, k, seed):
    """Fixed random states -> k map, row-scaled so R@s is O(1) for |s|<=1."""
    R = np.random.default_rng(seed).standard_normal((k, n)) / math.sqrt(n)
    return np.ascontiguousarray(R)


class ReservoirController:
    """Fresh per episode (stateful).  obs -> action in [-1, 1]^2."""

    __slots__ = ("_s", "_W", "_P", "_R", "_theta", "_buf")

    def __init__(self, W, P_T, R, theta):
        self._s = np.zeros(W.shape[0])
        self._W = W
        self._P = P_T
        self._R = R
        self._theta = theta
        self._buf = np.zeros(SENSOR_DIM)

    def __call__(self, obs):
        buf = self._buf
        buf[:] = 0.0
        buf[: len(obs)] = obs
        drive = self._P @ buf
        s = self._s
        for _ in range(RESERVOIR_STEPS):
            s = np.tanh(LEAK * s + self._W.dot(s) + drive)
        self._s = s
        return np.tanh((self._R @ s) @ self._theta)


# ---------------------------------------------------------------------------
# reward-only episode loop (frozen Arena; run_episode does not expose reward)
# ---------------------------------------------------------------------------
def rollout(build_ctrl, seed, mode, blink_period):
    """One episode.  Returns (return, success, steps, final_dist)."""
    env = Arena(seed=seed, mode=mode, blink_period=blink_period)
    ctrl = build_ctrl()
    obs = env.reset()
    total = 0.0
    info = {"dist": 1.0, "success": False}
    for _ in range(MAX_STEPS):
        obs, reward, done, info = env.step(ctrl(obs))
        total += float(reward)
        if done:
            break
    return total, bool(info["success"]), env.steps, float(info["dist"])


def score_theta(ctx, theta, mode, seeds, blink_period):
    """Reward-only fitness over a fixed seed pool.  ctx = dict of frozen arrays."""
    W, P_T, R = ctx["W"], ctx["P_T"], ctx["R"]
    build = lambda: ReservoirController(W, P_T, R, theta)  # noqa: E731
    total = 0.0
    succ = 0
    steps_succ = []
    fdist = []
    env_steps = 0
    for seed in seeds:
        ret, ok, steps, fd = rollout(build, seed, mode, blink_period)
        total += ret
        env_steps += steps
        if ok:
            succ += 1
            steps_succ.append(steps)
        fdist.append(fd)
    n = len(seeds)
    return {
        "mean_reward": total / n,
        "success_rate": succ / n,
        "mean_steps_to_success": float(np.mean(steps_succ)) if steps_succ else None,
        "mean_final_dist": float(np.mean(fdist)),
        "env_steps": env_steps,
    }


# ---------------------------------------------------------------------------
# optimizer: deterministic diagonal CMA-ES (sep-CMA-ES)
# ---------------------------------------------------------------------------
class SepCMAES:
    """Textbook sep-CMA-ES (Ros & Hansen 2008), maximising a 1-D fitness.

    Diagonal covariance makes it exact for separable problems and cheap for
    d <= 64; the full-covariance version needs far more samples than the
    600-candidate budget allows.
    """

    def __init__(self, dim, x0, sigma0, popsize, seed):
        assert popsize >= 4 and popsize % 2 == 0
        self.dim = dim
        self.lam = popsize
        self.mu = popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / float(np.sum(self.w ** 2))
        self.cc = 4.0 / (dim + 4.0)
        self.cs = (self.mueff + 2.0) / (dim + self.mueff + 5.0)
        self.c1 = 2.0 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((dim + 2.0) ** 2 + self.mueff),
        )
        self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (dim + 1.0)) - 1.0) + self.cs
        self.chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))
        self.rng = np.random.default_rng(seed)
        self.m = np.asarray(x0, dtype=np.float64).copy()
        self.sigma = float(sigma0)
        self.C = np.ones(dim)                 # diagonal covariance
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.gen = 0
        self._x = None
        self._y = None

    def ask(self):
        d = self.dim
        self._z = self.rng.standard_normal((self.lam, d))
        self._y = self._z * np.sqrt(self.C)[None, :]
        self._x = self.m[None, :] + self.sigma * self._y
        return self._x

    def tell(self, fitness):
        f = np.asarray(fitness, dtype=np.float64)
        order = np.argsort(-f, kind="stable")
        xs = self._x[order[: self.mu]]
        ys = self._y[order[: self.mu]]
        self.m = self.w @ xs
        y_w = self.w @ ys
        # C^{-1/2}(m - m_old)/sigma == y_w for a diagonal covariance
        self.ps = (1.0 - self.cs) * self.ps + math.sqrt(
            self.cs * (2.0 - self.cs) * self.mueff
        ) * y_w
        self.sigma *= math.exp(
            (self.cs / self.damps) * (float(np.linalg.norm(self.ps)) / self.chiN - 1.0)
        )
        self.sigma = min(max(self.sigma, 1e-6), 1e6)
        self.gen += 1
        hsig = float(np.linalg.norm(self.ps)) / math.sqrt(
            1.0 - (1.0 - self.cs) ** (2.0 * self.gen)
        ) / self.chiN < (1.4 + 2.0 / (self.dim + 1.0))
        self.pc = (1.0 - self.cc) * self.pc + (
            (math.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y_w) if hsig else 0.0
        )
        self.C = (
            (1.0 - self.c1 - self.cmu) * self.C
            + self.c1 * (self.pc ** 2 + (0.0 if hsig else self.cc * (2.0 - self.cc) * self.C))
            + self.cmu * (self.w @ (ys ** 2))
        )
        np.clip(self.C, 1e-12, None, out=self.C)


# ---------------------------------------------------------------------------
# parallel fitness evaluation
# ---------------------------------------------------------------------------
_CTX = {}
_CTX_KEY = {}


def _make_ctx(key):
    k, scale_name, readout_seed = key
    n, W, sr, alpha = build_W(scale_name)
    P_T = build_input_projection(n)
    R = build_readout_projection(n, k, readout_seed)
    return {"n": n, "W": W, "P_T": P_T, "R": R, "sr": sr, "alpha": alpha}


def _worker(task):
    key, theta_flat, mode, seeds, blink_period = task
    if _CTX_KEY.get("key") != key:
        _CTX.clear()
        _CTX.update(_make_ctx(key))
        _CTX_KEY["key"] = key
    ctx = _CTX
    theta = np.asarray(theta_flat, dtype=np.float64).reshape(ctx["R"].shape[0], ACTION_DIM)
    return score_theta(ctx, theta, mode, seeds, blink_period)


def train_seeds():
    return [TRAIN_SEED0 + i for i in range(N_TRAIN)]


def run_search(k, scale_name, train_mode, budget, pool, readout_seed, report=None):
    """One reward search.  Returns the results row (best theta + curves)."""
    n_cand = budget["candidates"]
    dim = k * ACTION_DIM
    gen_size = min(POPSIZE, n_cand)
    n_gens = max(1, n_cand // gen_size)
    seeds = train_seeds()
    es = SepCMAES(dim, np.zeros(dim), sigma0=0.8 / math.sqrt(k),
                  popsize=gen_size, seed=1000 + k * 10 + (0 if train_mode == "open" else 1)
                  + (0 if scale_name == "raw" else 7))
    episodes = 0
    env_steps = 0
    curve = []
    best = None  # (fitness, theta, stats)
    t0 = time.time()
    for gen in range(n_gens):
        X = es.ask()
        tasks = [( (k, scale_name, readout_seed), X[i], train_mode, seeds, BLINK_PERIOD)
                 for i in range(X.shape[0])]
        stats = pool.map(_worker, tasks)
        fits = np.array([s["mean_reward"] for s in stats])
        episodes += X.shape[0] * len(seeds)
        env_steps += sum(s["env_steps"] for s in stats)
        i = int(np.argmax(fits))
        if best is None or fits[i] > best[0]:
            best = (float(fits[i]), X[i].copy(), stats[i])
        curve.append({
            "gen": gen,
            "best_fitness": round(float(fits[i]), 4),
            "mean_fitness": round(float(fits.mean()), 4),
            "best_success_rate": round(float(stats[i]["success_rate"]), 4),
            "mean_success_rate": round(float(np.mean([s["success_rate"] for s in stats])), 4),
            "cum_best_fitness": round(best[0], 4),
            "cum_best_success_rate": round(float(best[2]["success_rate"]), 4),
            "sigma": round(float(es.sigma), 5),
        })
        es.tell(fits)
        if report:
            report(gen, n_gens, fits[i], stats[i]["success_rate"], es.sigma)
    return {
        "k": k,
        "weight_scale": scale_name,
        "train_mode": train_mode,
        "readout_seed": readout_seed,
        "optimizer": "sep-CMA-ES",
        "budget": {
            "candidates": n_cand,
            "episodes_per_candidate": len(seeds),
            "train_episodes": episodes,
            "generations": n_gens,
            "population": gen_size,
        },
        "train_env_steps": env_steps,
        "train_best_fitness": round(best[0], 4),
        "train_best_success_rate": round(float(best[2]["success_rate"]), 4),
        "best_theta": best[1].reshape(k, ACTION_DIM).round(8).tolist(),
        "learning_curve": curve,
        "search_wall_s": round(time.time() - t0, 2),
        "_episodes": episodes,
        "_theta": best[1],
    }


# ---------------------------------------------------------------------------
# final evaluation on the standard 200-episode protocol
# ---------------------------------------------------------------------------
def eval_standard(k, scale_name, theta, readout_seed):
    n, W, sr, alpha = build_W(scale_name)
    P_T = build_input_projection(n)
    R = build_readout_projection(n, k, readout_seed)
    th = np.asarray(theta, dtype=np.float64).reshape(k, ACTION_DIM)
    factory = lambda: ReservoirController(W, P_T, R, th)  # noqa: E731
    out = {}
    for mode in ("open", "blink"):
        out[f"{mode}3" if mode == "blink" else "open"] = run_many_factory(
            factory, EVAL_N, seed0=EVAL_SEED0, mode=mode, blink_period=BLINK_PERIOD
        )
    return out


def reference_bars():
    bars = {}
    bars["heuristic_open"] = run_many(heuristic_action, EVAL_N, seed0=EVAL_SEED0, mode="open")
    bars["memory_heuristic_open"] = run_many_factory(
        MemoryHeuristic, EVAL_N, seed0=EVAL_SEED0, mode="open")
    bars["random_open"] = run_many(
        random_action_factory(0), EVAL_N, seed0=EVAL_SEED0, mode="open")
    bars["heuristic_blink3"] = run_many(
        heuristic_action, EVAL_N, seed0=EVAL_SEED0, mode="blink", blink_period=BLINK_PERIOD)
    bars["memory_heuristic_blink3"] = run_many_factory(
        MemoryHeuristic, EVAL_N, seed0=EVAL_SEED0, mode="blink", blink_period=BLINK_PERIOD)
    bars["random_blink3"] = run_many(
        random_action_factory(0), EVAL_N, seed0=EVAL_SEED0, mode="blink",
        blink_period=BLINK_PERIOD)
    return bars


def untrained_baseline(k, scale_name, readout_seed, draws=5, seed=12345):
    """Floor for this architecture: theta ~ N(0, 1/sqrt(k)), same scale the ES
    starts from, evaluated on the standard protocol.  No search at all."""
    n, W, sr, alpha = build_W(scale_name)
    P_T = build_input_projection(n)
    R = build_readout_projection(n, k, readout_seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(draws):
        th = rng.standard_normal((k, ACTION_DIM)) / math.sqrt(k)
        factory = lambda: ReservoirController(W, P_T, R, th)  # noqa: E731
        r = {"open": run_many_factory(factory, EVAL_N, seed0=EVAL_SEED0, mode="open")}
        r["blink3"] = run_many_factory(factory, EVAL_N, seed0=EVAL_SEED0, mode="blink",
                                       blink_period=BLINK_PERIOD)
        rows.append(r)
    agg = {
        f"{m}_{f}": round(float(np.mean([r[m][f] for r in rows if r[m][f] is not None])), 4)
        if rows[0][m][f] is not None else None
        for m in ("open", "blink3")
        for f in ("success_rate", "mean_steps_to_success", "mean_final_dist")
    }
    agg["draws"] = draws
    return agg


def run_random_search_control(k, scale_name, train_mode, budget, pool, readout_seed):
    """Same budget, no adaptation: uniform theta draws.  The floor the ES must beat."""
    dim = k * ACTION_DIM
    seeds = train_seeds()
    rng = np.random.default_rng(999 + k)
    best = None
    episodes = 0
    env_steps = 0
    t0 = time.time()
    for chunk_start in range(0, budget["candidates"], POPSIZE):
        m = min(POPSIZE, budget["candidates"] - chunk_start)
        X = rng.standard_normal((m, dim)) * (0.8 / math.sqrt(k))
        tasks = [((k, scale_name, readout_seed), X[i], train_mode, seeds, BLINK_PERIOD)
                 for i in range(m)]
        stats = pool.map(_worker, tasks)
        fits = [s["mean_reward"] for s in stats]
        episodes += m * len(seeds)
        env_steps += sum(s["env_steps"] for s in stats)
        i = int(np.argmax(fits))
        if best is None or fits[i] > best[0]:
            best = (float(fits[i]), X[i].copy(), stats[i])
    return {
        "k": k, "weight_scale": scale_name, "train_mode": train_mode,
        "optimizer": "random-search-control",
        "budget": {"candidates": budget["candidates"],
                   "episodes_per_candidate": len(seeds), "train_episodes": episodes},
        "train_best_fitness": round(best[0], 4),
        "train_best_success_rate": round(float(best[2]["success_rate"]), 4),
        "best_theta": best[1].reshape(k, ACTION_DIM).round(8).tolist(),
        "search_wall_s": round(time.time() - t0, 2),
        "_episodes": episodes, "_theta": best[1],
    }


# ---------------------------------------------------------------------------
# optimizer self-test (is the ES the bottleneck?  no.)
# ---------------------------------------------------------------------------
def selftest():
    problems = {
        "sphere-10": (10, lambda x: -float(np.sum(x ** 2)), np.zeros(10), 1.0, 0.0),
        "quadratic-64": (64, lambda x: -float(np.sum((np.arange(64) + 1) * x ** 2)),
                         np.ones(64) * 2.0, 1.0, 0.0),
        "rosenbrock-8": (8, lambda x: -float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2
                                                     + (1 - x[:-1]) ** 2)),
                         np.zeros(8), 0.5, 1.0),
    }
    for name, (d, f, x0, s0, _) in problems.items():
        es = SepCMAES(d, x0, s0, 24, seed=7)
        be = -np.inf
        for _ in range(25):
            X = es.ask()
            fit = np.array([f(x) for x in X])
            be = max(be, float(fit.max()))
            es.tell(fit)
        print(f"  selftest {name:14s} d={d:2d} evals={25*24:4d} best={be:.6f}")


# ---------------------------------------------------------------------------
def sibling_imitation_reference():
    """Read lane 1A's imitation numbers if present (for the cross-lane verdict)."""
    if not SIBLING_RESULTS.exists():
        return None
    with open(SIBLING_RESULTS) as fh:
        d = json.load(fh)
    out = {"source": SIBLING_RESULTS.name,
           "note": "teacher-forced ridge readout, full-rank (2952x2), weight_scale=1/127"}
    for r in d.get("rows", []):
        out[f"{r['controller']}__{r['mode']}"] = {
            "success_rate": r["success_rate"],
            "mean_steps_to_success": r["mean_steps_to_success"],
            "mean_final_dist": r["mean_final_dist"],
        }
    return out


def run_extra(args):
    """Second pass: search the sibling's exact weight scale and append the rows."""
    with open(args.out) as fh:
        payload = json.load(fh)
    budget = {"candidates": N_CANDIDATES, "episodes_per_candidate": N_TRAIN,
              "total_eval_episodes": EVAL_N}
    procs = args.procs or max(2, min(20, (os.cpu_count() or 4) - 4))
    t0 = time.time()
    episodes = 0
    env_steps = 0
    rows = []
    with Pool(procs) as pool:
        for sc, ks in EXTRA_GRID:
            for mode in TRAIN_MODES:
                for k in ks:
                    tag = f"{sc:12s} train={mode:5s} k={k:2d}"
                    print(f"--- search {tag} ---", flush=True)
                    row = run_search(k, sc, mode, budget, pool, 7000 + k, report=None)
                    episodes += row["_episodes"]
                    env_steps += row["train_env_steps"]
                    theta = row.pop("_theta")
                    row["eval"] = eval_standard(k, sc, theta, 7000 + k)
                    rows.append(row)
                    e = row["eval"]
                    print(f"    {tag} -> open {e['open']['success_rate']:.3f} "
                          f"steps={e['open']['mean_steps_to_success']} | "
                          f"blink3 {e['blink3']['success_rate']:.3f} "
                          f"steps={e['blink3']['mean_steps_to_success']}", flush=True)
    payload["rows"].extend(rows)
    payload["sibling_lane_1A_imitation"] = sibling_imitation_reference()
    payload["totals"]["search_episodes"] += episodes
    payload["totals"]["search_env_steps"] += env_steps
    payload["totals"]["wall_seconds"] = round(
        payload["totals"]["wall_seconds"] + (time.time() - t0), 1)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"appended {len(rows)} rows -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny budget sanity run")
    ap.add_argument("--selftest", action="store_true", help="optimizer toy problems")
    ap.add_argument("--extra", action="store_true",
                    help="append EXTRA_GRID runs to an existing results file")
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.extra:
        run_extra(args)
        return

    if args.smoke:
        budget = {"candidates": 24, "episodes_per_candidate": 8, "total_eval_episodes": 200}
        global N_TRAIN, N_CANDIDATES, EVAL_N
        N_TRAIN, N_CANDIDATES, EVAL_N = 8, 24, 40
        modes, grid = ("open",), (("raw", (8,)),)
        control_grid = ()
    else:
        budget = {"candidates": N_CANDIDATES, "episodes_per_candidate": N_TRAIN,
                  "total_eval_episodes": EVAL_N}
        modes, grid, control_grid = TRAIN_MODES, SEARCH_GRID, CONTROL_GRID

    procs = args.procs or max(2, min(20, (os.cpu_count() or 4) - 4))
    print(f"procs={procs} candidates={budget['candidates']} "
          f"episodes/candidate={budget['episodes_per_candidate']}")

    n, W, sr, alpha = build_W("raw")
    print(f"reservoir n={n} nnz={W.nnz} spectral_radius(raw)={sr:.3f} "
          f"spectral0.9 alpha={0.9/sr:.6f}")

    t_start = time.time()
    total_episodes = 0
    total_env_steps = 0
    rows = []
    controls = []

    with Pool(procs) as pool:
        bars = reference_bars()
        print("reference bars:", json.dumps(bars, indent=None), flush=True)
        untrained = []
        for sc, ks in UNTRAINED_GRID if not args.smoke else ():
            for k in ks:
                u = untrained_baseline(k, sc, readout_seed=7000 + k, draws=UNTRAINED_DRAWS)
                u.update({"k": k, "weight_scale": sc})
                untrained.append(u)
                print("untrained", k, sc, json.dumps(u), flush=True)

        for sc, ks in grid:
            for mode in modes:
                for k in ks:
                    tag = f"{sc:12s} train={mode:5s} k={k:2d}"
                    print(f"--- search {tag} ---", flush=True)
                    row = run_search(k, sc, mode, budget, pool, 7000 + k, report=None)
                    total_episodes += row["_episodes"]
                    total_env_steps += row["train_env_steps"]
                    theta = row.pop("_theta")
                    row["eval"] = eval_standard(k, sc, theta, 7000 + k)
                    rows.append(row)
                    e = row["eval"]
                    print(f"    {tag} -> open {e['open']['success_rate']:.3f} "
                          f"steps={e['open']['mean_steps_to_success']} | "
                          f"blink3 {e['blink3']['success_rate']:.3f} "
                          f"steps={e['blink3']['mean_steps_to_success']} | "
                          f"train_best_succ={row['train_best_success_rate']:.3f}",
                          flush=True)

        for sc, k in control_grid:
            for mode in modes:
                c = run_random_search_control(k, sc, mode, budget, pool, 7000 + k)
                total_episodes += c["_episodes"]
                theta = c.pop("_theta")
                c["eval"] = eval_standard(k, sc, theta, 7000 + k)
                controls.append(c)
                print(f"random-control k={k} {mode}: open "
                      f"{c['eval']['open']['success_rate']} blink3 "
                      f"{c['eval']['blink3']['success_rate']}", flush=True)

    wall = time.time() - t_start
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": "1B: reservoir output decoder trained by reward search (no teacher)",
        "setup": {
            "reservoir": {"source": str(ADJ_PATH.relative_to(REPO)), "neurons": n,
                          "edges": int(W.nnz), "steps": RESERVOIR_STEPS, "leak": LEAK,
                          "spectral_radius_raw": round(sr, 4),
                          "spectral0.9_alpha": round(0.9 / sr, 8)},
            "input_projection": {"seed": PROJ_SEED, "shape": [SENSOR_DIM, n],
                                 "distribution": f"int({PROJ_LOW},{PROJ_HIGH})*{PROJ_SCALE}",
                                 "sensor_pad_note": "world.obs_vector returns 7 values; "
                                                    "padded to SENSOR_DIM=8 with a zero"},
            "decoder": "a = tanh((R @ s) @ theta); R (k x n) fixed random, theta (k x 2) SEARCHED",
            "readout_seed": "7000 + k",
            "reward": "env reward = -dist + 10 on success (teacher actions never used)",
            "train_protocol": {"seed0": TRAIN_SEED0, "episodes": N_TRAIN,
                               "note": "disjoint from the eval range"},
            "eval_protocol": {"episodes": budget["total_eval_episodes"], "seed0": EVAL_SEED0,
                              "modes": ["open", "blink_period=3"]},
            "epsilon": {"sigma0": "0.8/sqrt(k)", "theta_init": "zeros", "pop": POPSIZE,
                        "optimizer": "sep-CMA-ES (deterministic, seeded)"},
        },
        "reference_bars": bars,
        "untrained_random_theta": untrained,
        "random_search_control": controls,
        "rows": rows,
        "totals": {"search_episodes": total_episodes,
                   "search_env_steps": total_env_steps,
                   "wall_seconds": round(wall, 1)},
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")
    print(json.dumps(payload["totals"], indent=2))


if __name__ == "__main__":
    main()
