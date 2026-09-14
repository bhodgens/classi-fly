"""Shared substrate for lane 3: rhythm + anomaly detection on the Arena.

Lane 3 (docs/DIRECTIONS.md section 2.3): temporal/rhythmic tasks where the
answer depends on TIME, not just the current input - the shape where lane 1
measured that recurrence pays (100% blink3/blink8 vs 42.5% for the memoryless
map at blink8). The blink mode gives each head a metronome; both tasks here
ask the controller to act on its phase.

Substrate (FROZEN, imported from world.py): the Arena in mode='blink' with a
blink period P. The light sensors are zeroed except on steps t with
t % P == 0; those steps are the ONSETS (phase 0), and the 1..P-1 blind steps
between consecutive onsets are the light "off" intervals. Sensor layout
(obs_vector): [0]=cos bearing, [1]=sin bearing, [2]=1/(1+5*dist),
[3]=dist/(2R), [4:6]=(cos,sin heading), [6]=wall clearance.
NOTE: obs[2]/obs[3] encode DISTANCE only and are blind-phase-safe phase
carriers in the sense that obs[0]/obs[1] (bearing) alone vanish when blind;
any head must recover the blink phase from temporal structure.

Heads (shared with lane 1's protocol, see reservoir_imitation.py):
  larval connectome  : real wiring, rng-42 int8 (7, N) projection *0.02,
                       s = tanh(0.8*s + W@s + drive), 4 inner steps,
                       W = weights/127 (scipy.sparse csr). N=2952.
  synthetic 2048     : tools/ingest/synthetic.py generate(seed=42, n=2048,
                       fan_in=8, inhib_frac=0.2). Same dynamics/projection.
  memoryless map     : ridge on the raw 7 sensors, same fit protocol. Cannot
                       represent time by construction - the lane-1 control.
  MemoryHeuristic    : imported for reference; not used by these tasks.

Both tasks share this module's data collection (randomized policies over
training seeds, disjoint from the eval seed range) and the two-stage ridge
readout: sweep lambda 1..1000 (log grid, val-selected), refit on train+val,
tanh output squash with arctanh targets exactly like lane 1A.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]

# --- lane-1-standard reservoir construction -------------------------------- #
DECAY = 0.8
INNER_STEPS = 4
PROJ_SEED = 42
PROJ_LO, PROJ_HI = -6, 6
PROJ_SCALE = 0.02
WEIGHT_SCALE = 1.0 / 127.0

LAMBDAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
SQUASH_EPS = 0.005

SENSOR_DIM = 7


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


def load_synthetic(seed=42, n=2048, fan_in=8, inhib_frac=0.2):
    """Import tools/ingest/synthetic.py by path (it is not a package sibling)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_lane3_synthetic", str(REPO / "tools" / "ingest" / "synthetic.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    adj = mod.generate(seed=seed, n=n, fan_in=fan_in, inhib_frac=inhib_frac)
    W = sparse.csr_matrix(
        (np.asarray(adj["weights"], dtype=np.float64),
         np.asarray(adj["indices"], dtype=np.int64),
         np.asarray(adj["indptr"], dtype=np.int64)),
        shape=(n, n),
    )
    return adj, W


def make_projection(n, seed=PROJ_SEED):
    rng = np.random.default_rng(seed)
    return (rng.integers(PROJ_LO, PROJ_HI + 1, size=(SENSOR_DIM, n)).astype(np.float64)
            * PROJ_SCALE)


class Reservoir:
    """Lane-1-standard recurrent substrate; state persists across world steps."""

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
        drive = self.proj.T @ np.asarray(obs, dtype=np.float64)
        s = self.s
        for _ in range(self.steps):
            s = np.tanh(self.decay * s + (self.W @ s) + drive)
        self.s = s
        return s


class ReservoirController:
    """Fresh per episode (stateful): wraps a reservoir + trained readout."""

    def __init__(self, res, Wout, bout, squash_fn):
        self.res = res
        self.Wout = Wout
        self.bout = bout
        self.squash_fn = squash_fn

    def __call__(self, obs):
        s = self.res.step(obs)
        return self.squash_fn(self.Wout @ s + self.bout)


# --- two-stage ridge readout (lane 1A protocol) ----------------------------- #

def to_targets(a):
    return np.arctanh(np.clip(a, -1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS))


class RidgeFit:
    """Eigen-decomposed Gram so the lambda sweep is one matvec per lambda."""

    def __init__(self, X, Yt):
        xm, ym = X.mean(axis=0), Yt.mean(axis=0)
        Xc, Yc = X - xm, Yt - ym
        G = Xc.T @ Xc
        C = Xc.T @ Yc
        syy = float(np.sum(Yc * Yc))
        n = X.shape[0]
        self.xm, self.ym = xm, ym
        self.g, self.V = np.linalg.eigh(G)
        self.VtC = self.V.T @ C
        self.syy = syy
        self.n = n

    def weights(self, lam):
        return (self.V @ (self.VtC / (self.g + lam)[:, None])).T   # (out, d)

    def val_mse(self, W, Xva, Ytva):
        Xc = Xva - self.xm
        Yc = Ytva - self.ym
        syy = float(np.sum(Yc * Yc))
        a = float(np.trace(W @ (Xc.T @ Yc)))
        b = float(np.trace(W @ (Xc.T @ Xc) @ W.T))
        sse = syy - 2.0 * a + b
        return sse / (Xva.shape[0] * Yva_dim(Ytva))


def Yva_dim(Yt):
    return Yt.shape[1] if Yt.ndim > 1 else 1


def fit_ridge(Xtr, Ytr, Xva, Yva, lambdas=LAMBDAS, label=""):
    """Sweep lambda on val MSE, refit on train+val at the best one."""
    Ytr_t, Yva_t = to_targets(Ytr), to_targets(Yva)
    fit = RidgeFit(Xtr, Ytr_t)
    sweep = [{"lambda": lam, "val_mse": round(fit.val_mse(fit.weights(lam),
                                                          Xva, Yva_t), 8)}
             for lam in lambdas]
    best = min(sweep, key=lambda r: r["val_mse"])
    lam = best["lambda"]
    Xa = np.vstack([Xtr, Xva])
    Ya = np.vstack([Ytr_t, Yva_t])
    fa = RidgeFit(Xa, Ya)
    W = fa.weights(lam)
    b = fa.ym - W @ fa.xm
    if label:
        print(f"  [{label}] lambda={lam} val_mse={best['val_mse']:.6f} "
              f"pairs={Xa.shape[0]}")
    return W, b, {"best_lambda": lam, "val_mse": best["val_mse"], "sweep": sweep,
                  "pairs_train": int(Xtr.shape[0]), "pairs_val": int(Xva.shape[0])}


# --- policy pool for training-data collection ------------------------------- #

def random_policy_factory(seed):
    """Action = fixed per-seed random draw (stateful but decorrelated)."""
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(-1.0, 1.0, size=2)


def biased_policy_factory(seed, thrust_bias):
    """Random policy with a thrust offset - pushes the reservoir through more
    of the (bearing, wall, motion) state space than uniform noise alone."""
    rng = np.random.default_rng(seed)
    def act(obs):
        a = rng.uniform(-1.0, 1.0, size=2)
        a[1] = float(np.clip(a[1] + thrust_bias, -1.0, 1.0))
        return a
    return act


def rotate_policy_factory(seed, period):
    """Turn at a fixed per-seed rate - excites bearing dimension broadly."""
    rng = np.random.default_rng(seed)
    rate = rng.uniform(-1.0, 1.0)
    def act(obs):
        return np.array([rate, rng.uniform(-1.0, 1.0)])
    return act


def collect_states(build_res, period, seed0, n_episodes, max_steps,
                   policy_seed0=9000, policy_kinds=("uniform", "biased", "rotate"),
                   verbose=True):
    """Randomized-policy episode rollouts; returns concatenated reservoir states.

    Policies are drawn from a seed range disjoint from both training and eval
    arenas, so the state distribution covers the arena but no eval episode
    leaks into the readout.
    """
    Xs = []
    t0 = time.time()
    for i in range(n_episodes):
        from world import Arena  # noqa: PLC0415 (frozen harness)
        env = Arena(seed=seed0 + i, mode="blink", blink_period=period)
        kind = policy_kinds[i % len(policy_kinds)]
        ps = policy_seed0 + i
        if kind == "uniform":
            pol = random_policy_factory(ps)
        elif kind == "biased":
            pol = biased_policy_factory(ps, float(np.random.default_rng(ps).uniform(-0.4, 0.8)))
        else:
            pol = rotate_policy_factory(ps, period)
        res = build_res()
        obs = env.reset()
        for _ in range(max_steps):
            Xs.append(res.step(obs).astype(np.float32))
            obs, _r, done, _info = env.step(pol(obs))
            if done:
                break
    X = np.asarray(Xs, dtype=np.float64)
    if verbose:
        print(f"  collected {X.shape[0]} states over {n_episodes} episodes "
              f"({time.time() - t0:.1f}s)")
    return X


def collect_sensor_traces(period, seed0, n_episodes, max_steps,
                          policy_seed0=9000,
                          policy_kinds=("uniform", "biased", "rotate")):
    """Same rollouts, raw 7-sensor observations (the memoryless head's input)."""
    Xs = []
    for i in range(n_episodes):
        from world import Arena  # noqa: PLC0415
        env = Arena(seed=seed0 + i, mode="blink", blink_period=period)
        kind = policy_kinds[i % len(policy_kinds)]
        ps = policy_seed0 + i
        if kind == "uniform":
            pol = random_policy_factory(ps)
        elif kind == "biased":
            pol = biased_policy_factory(ps, float(np.random.default_rng(ps).uniform(-0.4, 0.8)))
        else:
            pol = rotate_policy_factory(ps, period)
        obs = env.reset()
        for _ in range(max_steps):
            Xs.append(np.asarray(obs, dtype=np.float32).copy())
            obs, _r, done, _info = env.step(pol(obs))
            if done:
                break
    return np.asarray(Xs, dtype=np.float64)


# --- metrics helpers ---------------------------------------------------------#

def r2(y, yhat):
    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return round(1.0 - ss_res / max(ss_tot, 1e-12), 4)
