"""Size co-sweep: can a SMALL synthetic reservoir match the real connectome?

docs/DIRECTIONS.md section 7 (lane-1 hardening) found: 512-neuron synthetic
reservoirs FAIL on phototaxis (22% vs real-2952 100%) while validation error
is FLAT in N - so it is not capacity. The suspected differentiator is graph
structure, specifically SPECTRAL RADIUS. Lane 1 never co-varied size with
spectral radius; this sweep does.

Protocol (identical to lane 1A so rows are comparable):
  - imitation training: teacher = heuristic_action in open, MemoryHeuristic
    in blink; targets arctanh(clip(action, +-0.995)); ridge lambda swept
    1e-3..100 and picked on held-out validation pairs (train seeds 5000+,
    val seeds 6000+).
  - evaluation: run_many_factory(factory, 200, seed0=1000, mode='blink3').
  - reservoir stateful, fresh per episode; scipy.sparse for W@s.

Axes:
  size          : 512, 1024, 2048
  spectral rho  : 0.5, 0.8, 0.95, 1.1, 1.5, 2.0   (power-iteration rescale)
  weight dist   : uniform   - mag U(0.05,1), 20% inhib (generator default shape)
                  signbal   - 50/50 signs, constant magnitude 0.5
                  lognormal - heavy-tailed magnitudes, 20% inhib
All variants share fan_in=6, generator seed 7, and are globally rescaled to
the exact target rho (so only distribution SHAPE + sign balance survive).

The bar: real-2952 scores 100% / 17.8 steps on blink3. A 512-2048 synthetic
config at >=95% and <=25 steps is a CHEAP deployable artifact -> validated in
mode='blink8'.

Run:  python3 tools/control/size_cosweep.py [--quick]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "control"))

from world import run_many_factory  # noqa: E402
import reservoir_imitation as ri    # noqa: E402  (frozen lane-1A machinery)

SIZES = [512, 1024, 2048]
RHOS = [0.5, 0.8, 0.95, 1.1, 1.5, 2.0]
DISTS = ["uniform", "signbal", "lognormal"]
GEN_SEED = 7
FAN_IN = 6
INHIB_FRAC = 0.2
TRAIN_PAIRS = 4000
VAL_PAIRS = 1000
MAX_EPISODES_COLLECT = 2000
EVAL_EPISODES = 200
EVAL_SEED0 = 1000
TRAIN_SEED0 = 5000
VAL_SEED0 = 6000

BAR_SUCCESS = 0.95
BAR_STEPS = 25.0

RESULTS_PATH = REPO / "tools" / "control" / "size_cosweep_results.json"


# --------------------------------------------------------------------------- #
# spectral radius
# --------------------------------------------------------------------------- #
def spectral_radius(W, iters=200, seed=0):
    """Power-iteration estimate of the largest-magnitude eigenvalue."""
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    rho = 0.0
    for _ in range(iters):
        w = W @ v
        norm = np.linalg.norm(w)
        if norm < 1e-14:
            return 0.0
        v = w / norm
        rho = float(norm)
    return rho


def rescale_to_rho(W, target):
    """Uniform rescale so the power-iteration spectral radius hits target."""
    r = spectral_radius(W)
    if r <= 0:
        raise ValueError("zero spectral radius")
    return (W @ sparse.diags(np.full(W.shape[0], 1.0))).multiply(target / r).tocsr()


# --------------------------------------------------------------------------- #
# synthetic graph builders (fan_in distinct targets, no self-loops, seed 7)
# --------------------------------------------------------------------------- #
def build_synthetic(n, dist, seed=GEN_SEED):
    rng = np.random.default_rng(seed)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.empty(n * FAN_IN, dtype=np.int64)
    weights = np.empty(n * FAN_IN, dtype=np.float64)
    k = 0
    for row in range(n):
        targets = np.sort(rng.choice(n - 1, size=FAN_IN, replace=False))
        targets = np.where(targets < row, targets, targets + 1)
        indices[k:k + FAN_IN] = targets
        if dist == "uniform":
            mag = rng.uniform(0.05, 1.0, size=FAN_IN)
            sign = np.where(rng.random(FAN_IN) < INHIB_FRAC, -1.0, 1.0)
        elif dist == "signbal":
            mag = np.full(FAN_IN, 0.5)
            sign = np.where(rng.random(FAN_IN) < 0.5, -1.0, 1.0)
        elif dist == "lognormal":
            mag = rng.lognormal(mean=0.0, sigma=1.0, size=FAN_IN)
            sign = np.where(rng.random(FAN_IN) < INHIB_FRAC, -1.0, 1.0)
        else:
            raise ValueError(dist)
        weights[k:k + FAN_IN] = sign * mag
        k += FAN_IN
        indptr[row + 1] = k
    return sparse.csr_matrix((weights, indices, indptr), shape=(n, n))


def build_W(n, rho_target, dist):
    W = build_synthetic(n, dist)
    r = spectral_radius(W)
    return (W * (rho_target / r)).tocsr()


# --------------------------------------------------------------------------- #
# one config -> one row
# --------------------------------------------------------------------------- #
def run_config(cfg, quick=False):
    n, rho, dist = cfg
    t0 = time.time()
    W = build_W(n, rho, dist)
    proj = ri.make_projection(n)
    mode, bp = "blink3", 3

    (Xtr, _), Ytr = ri.collect_pairs(W, proj, mode, bp, TRAIN_SEED0,
                                     MAX_EPISODES_COLLECT, TRAIN_PAIRS,
                                     verbose=False)
    (Xva, _), Yva = ri.collect_pairs(W, proj, mode, bp, VAL_SEED0,
                                     500, VAL_PAIRS, verbose=False)
    Wout, bout, info = ri.fit_representation(Xtr, Ytr, Xva, Yva,
                                             verbose=False, label=f"{n}/{rho}/{dist}")

    ctl = lambda: ri.ReservoirController(W, proj, Wout, bout)  # noqa: E731
    m = run_many_factory(ctl, n_episodes=30 if quick else EVAL_EPISODES,
                         seed0=EVAL_SEED0, mode=mode, blink_period=bp)
    row = {
        "size": n, "spectral_radius": rho, "weight_dist": dist,
        "lam": info["best_lambda"], "val_mse": info["val_mse"],
        "success_rate": m["success_rate"],
        "mean_steps": m["mean_steps_to_success"],
        "mean_final_dist": m["mean_final_dist"], "timeouts": m["timeouts"],
        "episodes": m["episodes"], "seconds": round(time.time() - t0, 1),
    }
    print("  " + json.dumps(row), flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="30-episode eval smoke test")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    results = {
        "sweep": "size x spectral-radius x weight-distribution co-sweep (blink3)",
        "protocol": "lane-1A imitation; eval run_many_factory 200 eps seed0=1000",
        "bar": {"ref": "real-2952: 100% / 17.8 steps (blink3)",
                "pass": f">={BAR_SUCCESS:.0%} success and <={BAR_STEPS} steps"},
        "real_matrix": {}, "rows": [], "blink8_rows": [], "verdict": None,
    }

    # --- axis 0: measured spectral radii ---------------------------------- #
    adj = json.loads((REPO / "data" / "larval_adjacency.json").read_text())
    n_real = int(adj["neurons"])
    W_real = sparse.csr_matrix(
        (np.asarray(adj["weights"], np.float64) / 127.0,
         np.asarray(adj["indices"], np.int64),
         np.asarray(adj["indptr"], np.int64)), shape=(n_real, n_real))
    rho_real = spectral_radius(W_real)
    W_syn_def = build_synthetic(2048, "uniform")  # pre-scale shape
    rho_syn_shape = spectral_radius(W_syn_def)
    results["real_matrix"] = {
        "name": adj["name"], "neurons": n_real, "edges": int(adj["edges"]),
        "spectral_radius_W_over_127": round(rho_real, 4),
        "synthetic_default_target": 0.9,
        "synthetic_presolve_shape_radius": round(rho_syn_shape, 4),
        "note": ("synthetic.generate() rescales to target=0.9 by construction; "
                 f"the raw drawn shape's radius is {rho_syn_shape:.2f}; the real "
                 f"connectome's W/127 radius is {rho_real:.4f}"),
    }
    print(f"real rho(W/127) = {rho_real:.4f}; "
          f"synthetic prescale shape rho = {rho_syn_shape:.4f}", flush=True)

    # --- resume support ---------------------------------------------------- #
    if RESULTS_PATH.exists() and not args.quick:
        try:
            prev = json.loads(RESULTS_PATH.read_text())
            done = {(r["size"], r["spectral_radius"], r["weight_dist"])
                    for r in prev.get("rows", [])}
            if prev.get("real_matrix"):
                results["real_matrix"] = prev["real_matrix"]
            results["rows"] = prev.get("rows", [])
            print(f"resuming: {len(done)} configs already done", flush=True)
        except Exception:
            done = set()
    else:
        done = set()

    configs = [(n, r, d) for n in SIZES for r in RHOS for d in DISTS
               if (n, r, d) not in done]

    if args.quick:
        row = run_config((512, 0.95, "uniform"), quick=True)
        results["rows"].append(row)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        print("quick smoke OK"); return 0

    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_config, configs):
            results["rows"].append(row)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    # --- verdicts ---------------------------------------------------------- #
    rows = results["rows"]
    passing = [r for r in rows
               if r["success_rate"] >= BAR_SUCCESS
               and r["mean_steps"] is not None and r["mean_steps"] <= BAR_STEPS]
    best_per_size = {}
    for n in SIZES:
        sub = [r for r in rows if r["size"] == n and r["success_rate"] > 0]
        if sub:
            best_per_size[n] = max(sub, key=lambda r: (r["success_rate"],
                                                       -(r["mean_steps"] or 999)))
    results["best_per_size"] = {str(k): v for k, v in best_per_size.items()}

    if passing:
        # validate every passing config in the harder blink8 mode
        for r in passing:
            n, rho, dist = r["size"], r["spectral_radius"], r["weight_dist"]
            W = build_W(n, rho, dist)
            proj = ri.make_projection(n)
            (Xtr, _), Ytr = ri.collect_pairs(W, proj, "blink8", 8, TRAIN_SEED0,
                                             MAX_EPISODES_COLLECT, TRAIN_PAIRS,
                                             verbose=False)
            (Xva, _), Yva = ri.collect_pairs(W, proj, "blink8", 8, VAL_SEED0,
                                             500, VAL_PAIRS, verbose=False)
            Wout, bout, info = ri.fit_representation(Xtr, Ytr, Xva, Yva,
                                                     verbose=False,
                                                     label=f"blink8/{n}/{rho}/{dist}")
            m = run_many_factory(lambda: ri.ReservoirController(W, proj, Wout, bout),
                                 n_episodes=EVAL_EPISODES, seed0=EVAL_SEED0,
                                 mode="blink8", blink_period=8)
            results["blink8_rows"].append({
                "size": n, "spectral_radius": rho, "weight_dist": dist,
                "lam": info["best_lambda"], "success_rate": m["success_rate"],
                "mean_steps": m["mean_steps_to_success"],
                "timeouts": m["timeouts"], "episodes": m["episodes"],
            })
            print("blink8 " + json.dumps(results["blink8_rows"][-1]), flush=True)

        cheapest = min(passing, key=lambda r: (r["size"], r["mean_steps"] or 999))
        b8 = next((b for b in results["blink8_rows"]
                   if (b["size"], b["spectral_radius"], b["weight_dist"])
                   == (cheapest["size"], cheapest["spectral_radius"],
                       cheapest["weight_dist"])), None)
        results["verdict"] = {
            "result": "SMALL SYNTHETIC MATCHES",
            "cheapest_deployable": cheapest,
            "blink8_check": b8,
            "n_passing_configs": len(passing),
        }
    else:
        results["verdict"] = {
            "result": "DOES NOT MATCH",
            "best_overall": max(rows, key=lambda r: r["success_rate"]),
        }

    results["runtime_seconds"] = round(time.time() - t0, 1)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nVERDICT: {json.dumps(results['verdict'])}")
    print(f"wrote {RESULTS_PATH} in {results['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
