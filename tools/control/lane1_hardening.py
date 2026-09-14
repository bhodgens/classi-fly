"""Lane 1 hardening: blink-period sweep + reservoir size sweep.

Questions (docs/DIRECTIONS.md section 7 follow-ups):
  1. WHERE does the reservoir-over-linear-map advantage disappear? Sweep
     blink_period in {2,3,5,8,12,20} x {reservoir, direct linear map,
     MemoryHeuristic}.
  2. DOES the advantage survive at 512 neurons? Size sweep at blink3,
     N in {512, 1024, 2952 (real connectome)} with matched density
     (fan_in ~ 21.5 edges/neuron, weights/127 dequant, random signed).

Procedure is IDENTICAL to lane 1A (reservoir_imitation.py): same teachers,
same seeds (train 5000+, val 6000+, eval 1000..1199), same ridge sweep and
selection protocol, same reservoir dynamics. Only the sweep axes differ.
Synthetic reservoirs use a random signed sparse graph with the same edge
count as the real connectome at matching N (fan_in = edges/neurons = 21.53).

Run: python3 tools/control/lane1_hardening.py
"""

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
    SENSOR_DIM, run_many, run_many_factory,
)
import reservoir_imitation as ri  # noqa: E402

TRAIN_SEED0, VAL_SEED0 = 5000, 6000
TRAIN_PAIRS, VAL_PAIRS = 10000, 2000
EVAL_EPISODES, EVAL_SEED0 = 200, 1000
PERIODS = [2, 3, 5, 8, 12, 20]
SIZE_SWEEP_AT = 3
SIZES = [512, 1024]           # 2952 = real connectome, always included
FAN_IN = 63545 / 2952         # edges per neuron of the real connectome
SYN_SEED = 123                # graph draw seed for synthetic reservoirs
MAX_EPISODES = 2000


def synthetic_reservoir(n, fan_in=FAN_IN, seed=SYN_SEED):
    """Random signed sparse graph, matched density, weights/127 dequant."""
    rng = np.random.default_rng(seed)
    n_edges = int(round(fan_in * n))
    rows = rng.integers(0, n, size=n_edges)
    cols = rng.integers(0, n, size=n_edges)
    vals = rng.integers(-127, 128, size=n_edges).astype(np.float64) / 127.0
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    proj = (np.random.default_rng(ri.PROJ_SEED)
            .integers(ri.PROJ_LO, ri.PROJ_HI + 1, size=(SENSOR_DIM, n))
            .astype(np.float64) * ri.PROJ_SCALE)
    return W, proj


def collect_res_pairs(W, proj, blink_period, seed0, n_episodes, target_pairs):
    """Reservoir states + teacher actions (MemoryHeuristic teacher), blink mode."""
    X, Y = [], []
    seen, i = 0, -1
    for i in range(n_episodes):
        env = ri.Arena(seed=seed0 + i, mode="blink", blink_period=blink_period)
        teacher = ri.MemoryHeuristic(decay=1.0)
        res = ri.Reservoir(W, proj)
        obs = env.reset()
        for _ in range(ri.MAX_STEPS):
            s = res.step(obs)
            X.append(s.astype(np.float32))
            Y.append(np.asarray(teacher(obs), dtype=np.float32))
            seen += 1
            obs, _r, done, _info = env.step(Y[-1])
            if done:
                break
        if seen >= target_pairs:
            break
    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64), i + 1


def train_and_eval(W, proj, blink_period, label, X_dir_tr=None, Y_dir_tr=None,
                   X_dir_va=None, Y_dir_va=None):
    """Full lane-1A procedure for one (reservoir, blink_period); returns rows."""
    t0 = time.time()
    Xtr, Ytr, n_tr = collect_res_pairs(W, proj, blink_period, TRAIN_SEED0,
                                       MAX_EPISODES, TRAIN_PAIRS)
    Xva, Yva, n_va = collect_res_pairs(W, proj, blink_period, VAL_SEED0,
                                       500, VAL_PAIRS)
    Wout, bout, info = ri.fit_representation(Xtr, Ytr, Xva, Yva, label=label)
    ev = ri.run_many_factory(lambda: ri.ReservoirController(W, proj, Wout, bout),
                             n_episodes=EVAL_EPISODES, seed0=EVAL_SEED0,
                             mode="blink", blink_period=blink_period)
    row = {"controller": f"reservoir[{label}]", "blink_period": blink_period,
           "n_train_episodes": n_tr, "n_val_episodes": n_va,
           "best_lambda": info["best_lambda"], "val_mse": info["val_mse"], **ev}
    print(f"  [{label}] {json.dumps(row)}  ({time.time() - t0:.0f}s)")
    del Xtr, Xva
    gc.collect()

    rows = [row]

    # direct linear map: representation is W-independent -> fit once per period
    if X_dir_tr is not None:
        Wd, bd, dinfo = ri.fit_representation(X_dir_tr, Y_dir_tr, X_dir_va, Y_dir_va,
                                              label=f"direct/bp{blink_period}")
        evd = ri.run_many(lambda obs: np.tanh(Wd @ obs + bd),
                          n_episodes=EVAL_EPISODES, seed0=EVAL_SEED0,
                          mode="blink", blink_period=blink_period)
        rows.append({"controller": "direct_linear_map", "blink_period": blink_period,
                     "best_lambda": dinfo["best_lambda"], "val_mse": dinfo["val_mse"],
                     **evd})
        print(f"  [direct bp{blink_period}] {json.dumps(rows[-1])}")
    return rows


def main() -> int:
    t_start = time.time()
    adj, W_real = ri.load_connectome(str(REPO / "data" / "larval_adjacency.json"))
    proj_real = ri.make_projection(adj["neurons"])
    print(f"real connectome: neurons={adj['neurons']} edges={adj['edges']} "
          f"fan_in={adj['edges'] / adj['neurons']:.2f}")

    results = {
        "lane": "1 hardening - blink-period sweep and reservoir size sweep",
        "harness": "tools/control/world.py (frozen)",
        "procedure": "identical to lane 1A (tools/control/reservoir_imitation.py): "
                     "MemoryHeuristic teacher in blink mode, ridge sweep in "
                     "arctanh space, val-selected lambda, refit on train+val",
        "connectome": {"name": adj["name"], "neurons": int(adj["neurons"]),
                       "edges": int(adj["edges"])},
        "eval": {"episodes": EVAL_EPISODES, "seed0": EVAL_SEED0,
                 "train_seed0": TRAIN_SEED0, "val_seed0": VAL_SEED0,
                 "train_pairs": TRAIN_PAIRS, "val_pairs": VAL_PAIRS},
        "sweeps": {"blink_periods": PERIODS,
                   "size_sweep_blink_period": SIZE_SWEEP_AT,
                   "sizes": SIZES + [int(adj["neurons"])],
                   "synthetic_fan_in": round(FAN_IN, 3),
                   "synthetic_graph_seed": SYN_SEED,
                   "synthetic_note": "random signed sparse, integers/-127 dequant, "
                                     "density matched to the real connectome; "
                                     "projection = rng(42) int -6..6 * 0.02"},
        "metric_note": "mean_steps_to_success averages SUCCESSFUL episodes only; "
                       "timeouts is the count of failed (200-step) episodes.",
        "reference_lane1A": {"blink3_reservoir": "100% / 17.8",
                             "blink3_direct": "50.5% / 32.9",
                             "blink8_reservoir": "100% / 40.9",
                             "blink8_direct": "42.5% / 61.7"},
        "blink_sweep": [],
        "size_sweep": [],
    }

    # ---- 1. blink-period sweep (real connectome, N=2952) -------------------- #
    print("\n=== blink sweep (real connectome) ===")
    for bp in PERIODS:
        print(f"--- blink_period={bp} ---")
        # direct representation is W-independent: collect it alongside the
        # real-reservoir states (same episodes -> exact lane-1A pairing)
        (Xtr_res, Xtr_dir), Ytr = ri.collect_pairs(
            W_real, proj_real, "blink", bp, TRAIN_SEED0, MAX_EPISODES, TRAIN_PAIRS)
        (Xva_res, Xva_dir), Yva = ri.collect_pairs(
            W_real, proj_real, "blink", bp, VAL_SEED0, 500, VAL_PAIRS)
        del Xtr_res, Xva_res
        gc.collect()

        rows = train_and_eval(W_real, proj_real, bp, f"real2952_bp{bp}",
                              X_dir_tr=Xtr_dir, Y_dir_tr=Ytr,
                              X_dir_va=Xva_dir, Y_dir_va=Yva)
        mh = run_many_factory(lambda: ri.MemoryHeuristic(decay=1.0),
                              n_episodes=EVAL_EPISODES, seed0=EVAL_SEED0,
                              mode="blink", blink_period=bp)
        rows.append({"controller": "MemoryHeuristic", "blink_period": bp, **mh})
        print(f"  [memory] {json.dumps(rows[-1])}")
        results["blink_sweep"].extend(rows)
        gc.collect()

    # sanity: reference points must reproduce
    def get(sweep, ctl, bp):
        return next(r for r in sweep if r["controller"].startswith(ctl)
                    and r["blink_period"] == bp)
    r3, r8 = get(results["blink_sweep"], "reservoir[real", 3), \
        get(results["blink_sweep"], "reservoir[real", 8)
    assert r3["success_rate"] >= 0.99 and r3["mean_steps_to_success"] < 25, \
        f"blink3 reference NOT reproduced: {r3} - STOP"
    assert r8["success_rate"] >= 0.99 and r8["mean_steps_to_success"] < 50, \
        f"blink8 reference NOT reproduced: {r8} - STOP"
    print("\nsanity vs lane-1A reference: blink3 and blink8 reproduced OK")

    # ---- 2. size sweep at blink3 -------------------------------------------- #
    print(f"\n=== size sweep at blink{SIZE_SWEEP_AT} ===")
    for n in SIZES:
        W_syn, proj_syn = synthetic_reservoir(n)
        print(f"--- N={n} (synthetic, fan_in={FAN_IN:.2f}, "
              f"edges={int(round(FAN_IN * n))}) ---")
        results["size_sweep"].extend(
            train_and_eval(W_syn, proj_syn, SIZE_SWEEP_AT, f"syn{n}_bp{SIZE_SWEEP_AT}"))
        del W_syn, proj_syn
        gc.collect()

    # real connectome at blink3: reuse the sweep row
    results["size_sweep"].append(dict(r3, controller="reservoir[real2952]"))
    d3 = get(results["blink_sweep"], "direct", 3)
    results["size_sweep"].append(dict(d3, controller="direct_linear_map"))

    out = REPO / "tools" / "control" / "lane1_hardening_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}  ({time.time() - t_start:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
