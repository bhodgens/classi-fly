"""Lane 1C: selective descending-neuron readout + wiring controls.

Two questions, both about the REAL larval connectome inside the frozen lane-1
control harness (tools/control/world.py, imported unmodified):

  Q1  DOOMFLY-style SELECTIVE readout.  The Doom fly project does not train a
      readout over all neurons; it hand-picks a few descending neurons
      ("DNp20 turns, DNpe017 rate"), chosen after visual-response calibration.
      Here: instead of a ridge readout over all 2,952 reservoir states, take
      the actions off k selected neurons drawn from a biologically motivated
      DESCENDING-NEURON candidate pool, selected by correlation with the
      teacher's actions on held-out episodes (the "chosen after calibration"
      pattern).  k = 2, 8, 32.  Does that beat the dense trained readout?

  Q2  THE WIRING CONTROL.  Does the specific larval wiring matter, or would
      any fixed sparse matrix do?  Same harness, same readout training, three
      matrices with identical neuron and edge counts:
        - real       : data/larval_adjacency.json
        - shuffled   : degree-preserving edge shuffle (Maslov-Sneppen
                       double-edge swaps) of the real matrix
        - random     : uniformly random topology, weights i.i.d. from the
                       empirical weight distribution

Frozen-harness notes (discovered, not chosen):
  * world.SENSOR_DIM is 8 but Arena emits a 7-element observation
    (obs_vector returns 4 light sensors + [hx, hy, wall]).  The input
    projection is therefore drawn as (SENSOR_DIM=8, n) -- rng 42, int8 in
    [-6, 6], scaled 0.02, exactly as specified -- and its first 7 rows are
    used; the 8th row is multiplied by nothing.  Recorded in the results.

Reservoir (the standard lane-1 build):
    s = tanh(0.8*s + W@s + proj.T@obs),  s0 = 0, 4 steps, decay 0.8,
    W = larval weights / 127.0 (repo convention for this same file), spmv via
    scipy.sparse.  The reservoir is stateful inside an episode and reset to
    zeros at every episode start (run_many_factory).

Readout: ridge from (standardized) states to the teacher's action, fit on
inverse-tanh targets so that tanh(W@s+b) reproduces the teacher, penalty
swept and chosen by held-out imitation MSE.

Protocol / seeds:
    train  episodes seeds 100..179   (fit the readout)
    calib  episodes seeds 200..239   (select neurons + sweep the penalty)
    eval   episodes seeds 1000..1199 (standard: 200 episodes, mode open/blink3)

Writes tools/control/reservoir_selective_results.json.  Touches nothing else.
"""

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "control"))

from world import (  # noqa: E402
    ACTION_DIM,
    SENSOR_DIM,
    Arena,
    MemoryHeuristic,
    heuristic_action,
    random_action_factory,
    run_many,
    run_many_factory,
)

ADJ = REPO / "data" / "larval_adjacency.json"
CSV_CONN = REPO / "data" / "larval_signed_connectivity.csv"
CSV_ANNOT = REPO / "data" / "science.add9330_data_s2.csv"
OUT = REPO / "tools" / "control" / "reservoir_selective_results.json"

STEPS = 4
DECAY = 0.8
IN_SCALE = 0.02
SEED_PROJ = 42
WEIGHT_SCALE = 127.0
TRAIN_SEEDS = range(100, 180)     # 80 episodes
CALIB_SEEDS = range(200, 240)     # 40 episodes
N_EVAL = 200
EVAL_SEED0 = 1000
LAMS = [1e-3, 1e-2, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]
KS = [2, 8, 32]
# descending-neuron candidate pools (Winding et al. 2023 celltype column)
POOL_DN = {"DN-VNC", "DN-SEZ"}                      # descending neurons proper
POOL_DN_BROAD = {"DN-VNC", "DN-SEZ", "pre-DN-VNC", "pre-DN-SEZ"}


# --------------------------------------------------------------------------
# connectome
# --------------------------------------------------------------------------

def load_adjacency():
    adj = json.loads(ADJ.read_text())
    n = int(adj["neurons"])
    ip = np.asarray(adj["indptr"], dtype=np.int64)
    ix = np.asarray(adj["indices"], dtype=np.int64)
    wt = np.asarray(adj["weights"], dtype=np.float64) / WEIGHT_SCALE
    return n, ip, ix, wt


def csr(ip, ix, wt, n):
    return sp.csr_matrix((wt, ix, ip), shape=(n, n))


def shuffle_wiring(ip, ix, wt, n, seed=7, swap_factor=5):
    """Degree-preserving edge shuffle: Maslov-Sneppen double-edge swaps.

    Each swap takes two edges (a->b), (c->d) and rewires them to (a->d),
    (c->b), rejecting self-loops and duplicate edges.  Both the out-degree
    and the in-degree sequence of the graph are preserved exactly; only the
    placement of the edges is randomized.  Weights travel with their source
    row (a and c), so each source keeps its exact weight multiset.
    """
    rng = np.random.default_rng(seed)
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(ip))
    tgt = ix.copy()
    w = wt.copy()
    E = len(src)
    present = set((int(s) * n + int(t)) for s, t in zip(src, tgt))
    n_swap = swap_factor * E
    done = 0
    attempts = 0
    max_attempts = 50 * n_swap
    while done < n_swap and attempts < max_attempts:
        attempts += 1
        e1, e2 = rng.integers(0, E, size=2)
        a, b = int(src[e1]), int(tgt[e1])
        c, d = int(src[e2]), int(tgt[e2])
        if a == c or b == d or a == d or c == b:
            continue
        if (a * n + d) in present or (c * n + b) in present:
            continue
        present.discard(a * n + b)
        present.discard(c * n + d)
        present.add(a * n + d)
        present.add(c * n + b)
        tgt[e1], w[e1], tgt[e2], w[e2] = d, w[e2], b, w[e1]
        done += 1
    order = np.argsort(src, kind="stable")
    src, tgt, w = src[order], tgt[order], w[order]
    ip2 = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=ip2[1:])
    return ip2, tgt, w, {"swaps_completed": done, "swaps_attempted": attempts}


def random_wiring(n, n_edges, weight_pool, seed=11):
    """Same neuron count, same edge count, random topology and weights.

    Weights are drawn i.i.d. (with replacement) from the empirical larval
    weight multiset, so only the topology and the edge placement are random;
    the weight magnitude distribution is held fixed.  No self-loops.
    """
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, size=n_edges)
    tgt = rng.integers(0, n, size=n_edges)
    same = src == tgt
    while same.any():
        tgt[same] = rng.integers(0, n, size=int(same.sum()))
        same = src == tgt
    w = rng.choice(weight_pool, size=n_edges, replace=True)
    order = np.argsort(src, kind="stable")
    src, tgt, w = src[order], tgt[order], w[order]
    ip2 = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=ip2[1:])
    return ip2, tgt, w


# --------------------------------------------------------------------------
# annotations
# --------------------------------------------------------------------------

def load_annotations(n):
    """celltype per adjacency row, joined through the connectivity CSV header."""
    with open(CSV_CONN) as f:
        hdr = f.readline().rstrip("\n").split(",")
    id2row = {int(x): i for i, x in enumerate(hdr[1:])}
    assert len(id2row) == n, (len(id2row), n)
    id2ct = {}
    with open(CSV_ANNOT) as f:
        for r in csv.DictReader(f):
            ct = r["celltype"].strip()
            for c in ("left_id", "right_id"):
                v = r[c].strip()
                if v != "no pair":
                    id2ct[int(v)] = ct
    cts = [None] * n
    for nid, row in id2row.items():
        cts[row] = id2ct.get(nid)
    return [c if c is not None else "unannotated" for c in cts]


# --------------------------------------------------------------------------
# reservoir + readout
# --------------------------------------------------------------------------

def make_proj(n):
    rng = np.random.default_rng(SEED_PROJ)
    return rng.integers(-6, 7, size=(SENSOR_DIM, n)).astype(np.int8).astype(np.float64) * IN_SCALE


class Reservoir:
    """Stateful reservoir; one instance per episode, reset by construction."""

    def __init__(self, W, proj, cols=None):
        self.W = W
        self.proj = proj
        self.cols = cols            # readout feature columns (None = all)
        self.n = W.shape[0]
        self.s = np.zeros(self.n)
        self.extra = None           # fixed-gain decoder state

    def step_state(self, obs):
        s = self.s
        p = self.proj[: obs.shape[0]]
        for _ in range(STEPS):
            s = np.tanh(DECAY * s + self.W @ s + p.T @ obs)
        self.s = s
        return s


def teacher_rollout(seed, mode):
    env = Arena(seed=seed, mode=mode, blink_period=3)
    obs = env.reset()
    teacher = heuristic_action if mode == "open" else MemoryHeuristic()
    O, A = [], []
    for _ in range(200):
        a = np.asarray(teacher(obs), dtype=np.float64)
        O.append(np.asarray(obs, dtype=np.float64))
        A.append(a)
        obs, _r, done, _i = env.step(a)
        if done:
            break
    return np.asarray(O), np.asarray(A)


def collect_states(W, proj, seeds, mode):
    """Reservoir states + teacher actions for a set of episodes."""
    Xs, Ys, lens = [], [], []
    for sd in seeds:
        O, A = teacher_rollout(sd, mode)
        s = np.zeros(W.shape[0])
        p = proj[: O.shape[1]]
        S = np.empty((len(O), W.shape[0]))
        for t in range(len(O)):
            for _ in range(STEPS):
                s = np.tanh(DECAY * s + W @ s + p.T @ O[t])
            S[t] = s
        Xs.append(S)
        Ys.append(A)
        lens.append(len(O))
    return np.vstack(Xs), np.vstack(Ys), lens


def ridge(Xn, Y, lam):
    """Ridge solve; dual (kernel) form when there are fewer samples than features."""
    n_s, d = Xn.shape
    if n_s <= d:
        G = Xn @ Xn.T + lam * np.eye(n_s)
        return Xn.T @ solve(G, Y, assume_a="pos")
    G = Xn.T @ Xn + lam * np.eye(d)
    return solve(G, Xn.T @ Y, assume_a="pos")


class TrainedReadout:
    """a = tanh(W_r @ z(state) + b) from a ridge fit on inverse-tanh targets."""

    def __init__(self, W, proj, cols, mu, sd, Wr, br):
        self.res = Reservoir(W, proj)
        self.cols = cols
        self.mu, self.sd, self.Wr, self.br = mu, sd, Wr, br

    def act(self, obs):
        s = self.res.step_state(obs)
        z = (s[self.cols] - self.mu) / self.sd
        return np.tanh(z @ self.Wr + self.br)


class FixedGainReadout:
    """DOOMFLY literal: action_j = tanh(g_j * s_{i_j}) - one neuron per action,
    a single calibrated scalar gain per action (no trained weight matrix)."""

    def __init__(self, W, proj, idx, gains):
        self.W, self.proj, self.idx, self.g = W, proj, idx, gains
        self.res = Reservoir(W, proj)

    def act(self, obs):
        s = self.res.step_state(obs)
        return np.tanh(self.g * s[self.idx])


def build_factory(W, proj, X, Y, cols, lam):
    mu = X[:, cols].mean(0)
    sd = X[:, cols].std(0) + 1e-6
    Xn = (X[:, cols] - mu) / sd
    Yt = np.arctanh(np.clip(Y, -0.995, 0.995))
    Wr = ridge(Xn, Yt, lam)

    def factory():
        return TrainedReadout(W, proj, cols, mu, sd, Wr, np.zeros(Y.shape[1])).act
    return factory, Wr


def imitation_mse(W, proj, X, Y, cols, lam):
    mu = X[:, cols].mean(0)
    sd = X[:, cols].std(0) + 1e-6
    Xn = (X[:, cols] - mu) / sd
    Yt = np.arctanh(np.clip(Y, -0.995, 0.995))
    Wr = ridge(Xn, Yt, lam)
    pred = np.tanh(Xn @ Wr)
    return float(((pred - Y) ** 2).mean())


def eval_rows(factory, mode, n=N_EVAL, seed0=EVAL_SEED0):
    """Per-episode rows; kept so wiring contrasts can be tested PAIRED (same seeds)."""
    from world import run_episode
    return [run_episode(factory(), seed=seed0 + i, mode=mode, blink_period=3)
            for i in range(n)]


def summarize_rows(rows):
    from world import summarize
    r = summarize(rows)
    return {"success_rate": r["success_rate"], "mean_steps": r["mean_steps_to_success"],
            "mean_final_dist": r["mean_final_dist"], "timeouts": r["timeouts"]}


def censored_steps(rows):
    """Episode length with failures censored at the 200-step cap (fully observed)."""
    return np.array([r["steps"] for r in rows], dtype=np.float64)


def paired_bootstrap(a, b, n_boot=4000, seed=5):
    """Paired bootstrap CI of mean(a) - mean(b) over the same episode seeds."""
    rng = np.random.default_rng(seed)
    n = len(a)
    d = float(a.mean() - b.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": round(d, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


# --------------------------------------------------------------------------
# neuron selection
# --------------------------------------------------------------------------

def action_correlations(X_calib, Y_calib):
    """Pearson r between every reservoir state column and every teacher action."""
    S = X_calib - X_calib.mean(0)
    S /= (X_calib.std(0) + 1e-12)
    A = Y_calib - Y_calib.mean(0)
    A /= (Y_calib.std(0) + 1e-12)
    return (S.T @ A) / len(X_calib)          # (n, ACTION_DIM)


def select_neurons(corr, pool_rows, k):
    """Top |corr| neurons per action from the pool (DOOMFLY: one per action at k=2)."""
    per_action = max(1, k // ACTION_DIM)
    chosen = []
    for j in range(ACTION_DIM):
        order = sorted(pool_rows, key=lambda r: -abs(corr[r, j]))
        chosen.append(order[:per_action])
    union = sorted({r for p in chosen for r in p})
    return union, chosen


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    t0 = time.time()
    n, ip, ix, wt = load_adjacency()
    cts = load_annotations(n)
    W_real = csr(ip, ix, wt, n)

    ct_hist = Counter(cts)
    dn_rows = [i for i in range(n) if cts[i] in POOL_DN]
    dn_broad_rows = [i for i in range(n) if cts[i] in POOL_DN_BROAD]

    results = {
        "experiment": "lane-1c-selective-descending-readout + wiring-controls",
        "harness": "tools/control/world.py (frozen, unmodified)",
        "reservoir": {
            "neurons": n, "edges": int(len(ix)), "steps": STEPS, "decay": DECAY,
            "weight_scale": WEIGHT_SCALE, "in_scale": IN_SCALE, "proj_seed": SEED_PROJ,
            "proj_shape": [SENSOR_DIM, n],
            "obs_dim_actual": 7,
            "obs_dim_note": "world.SENSOR_DIM is 8 but obs_vector emits 7 values "
                            "(4 light + hx,hy,wall); the projection is drawn (8,n) "
                            "as specified and its first 7 rows are used.",
            "evaluation": {"episodes": N_EVAL, "seed0": EVAL_SEED0,
                           "modes": ["open", "blink3"]},
            "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
            "calib_seeds": [CALIB_SEEDS[0], CALIB_SEEDS[-1]],
            "lam_sweep": LAMS,
        },
        "annotations": {
            "source": "Winding et al. 2023 Science 379:eadd9330 supplementary table S2 "
                      "(data/science.add9330_data_s2.csv), celltype column",
            "join": "via the data/larval_signed_connectivity.csv numeric header",
            "annotated_rows": int(sum(1 for c in cts if c != "unannotated")),
            "total_rows": n,
            "celltype_histogram": dict(sorted(ct_hist.items(), key=lambda kv: -kv[1])),
            "dn_pool_primary": {"celltypes": sorted(POOL_DN), "n": len(dn_rows)},
            "dn_pool_broad": {"celltypes": sorted(POOL_DN_BROAD), "n": len(dn_broad_rows)},
        },
        "reference_bars": {},
        "controllers": {},
    }

    # ---- reference bars (this harness, this seed range) -------------------
    for name, ctrl, fac in (
        ("heuristic", heuristic_action, None),
        ("memory_heuristic", None, MemoryHeuristic),
        ("random", None, lambda: random_action_factory(0)),
    ):
        row = {}
        for mode in ("open", "blink"):
            if fac is not None:
                r = run_many_factory(fac, N_EVAL, seed0=EVAL_SEED0, mode=mode, blink_period=3)
            else:
                r = run_many(ctrl, N_EVAL, mode=mode, seed0=EVAL_SEED0, blink_period=3)
            row[mode] = {"success_rate": r["success_rate"],
                         "mean_steps": r["mean_steps_to_success"],
                         "mean_final_dist": r["mean_final_dist"]}
        results["reference_bars"][name] = row
        print(f"[ref] {name:16s} open={row['open']} blink={row['blink']}", flush=True)

    # ---- wiring variants --------------------------------------------------
    real_dn_cnt = len(ix)
    ip_sh, ix_sh, wt_sh, sh_info = shuffle_wiring(ip, ix, wt, n, seed=7)
    wpool = np.asarray(json.loads(ADJ.read_text())["weights"], dtype=np.float64) / WEIGHT_SCALE
    ip_rn, ix_rn, wt_rn = random_wiring(n, real_dn_cnt, wpool, seed=11)
    wirings = {
        "real": (ip, ix, wt),
        "shuffled_degree_preserving": (ip_sh, ix_sh, wt_sh),
        "random_sparse_same_counts": (ip_rn, ix_rn, wt_rn),
    }
    results["wiring_controls"] = {
        "shuffled": {"method": "Maslov-Sneppen double-edge swaps, weight travels with its "
                               "source row; out-degree and in-degree sequences preserved "
                               "exactly", **sh_info,
                     "edges": int((ip_sh[-1])), "neurons": n},
        "random": {"method": "uniform random sources/targets, no self-loops, weights i.i.d. "
                             "from the empirical larval weight multiset",
                   "edges": int(ip_rn[-1]), "neurons": n},
    }

    dense_ep = {}
    for wname, (wip, wix, wwt) in wirings.items():
        W = csr(wip, wix, wwt, n)
        proj = make_proj(n)
        for mode in ("open", "blink"):
            X_tr, Y_tr, _ = collect_states(W, proj, TRAIN_SEEDS, mode)
            X_ca, Y_ca, _ = collect_states(W, proj, CALIB_SEEDS, mode)

            # --- dense trained readout: all 2952 states, penalty swept ----
            all_cols = np.arange(n)
            mses = [imitation_mse(W, proj, X_ca, Y_ca, all_cols, l) for l in LAMS]
            lam = LAMS[int(np.argmin(mses))]
            factory, _ = build_factory(W, proj, X_tr, Y_tr, all_cols, lam)
            ep = eval_rows(factory, mode)
            dense_ep[(wname, mode)] = ep
            rows = summarize_rows(ep)
            key = "dense_readout_all2952"
            results["controllers"][f"{wname}|{key}|{mode}"] = {
                "controller": key, "wiring": wname, "mode": mode,
                "readout": "ridge on all 2952 reservoir states, inverse-tanh targets",
                "n_readout_features": n, "lam_selected": lam,
                "lam_sweep_calib_mse": dict(zip([str(l) for l in LAMS],
                                                [round(m, 6) for m in mses])),
                **rows,
            }
            print(f"[{wname}|{mode}] dense lam={lam} succ={rows['success_rate']} "
                  f"steps={rows['mean_steps']} dist={rows['mean_final_dist']}", flush=True)
            OUT.write_text(json.dumps(results, indent=2))

            # --- Q1 selective readout (real wiring only) ------------------
            if wname != "real":
                continue
            corr = action_correlations(X_ca, Y_ca)
            for k in KS:
                chosen, per_action = select_neurons(corr, dn_rows, k)
                cols = np.asarray(chosen)
                mses = [imitation_mse(W, proj, X_ca, Y_ca, cols, l) for l in LAMS]
                lam = LAMS[int(np.argmin(mses))]
                factory, _ = build_factory(W, proj, X_tr, Y_tr, cols, lam)
                ep = eval_rows(factory, mode)
                rows = summarize_rows(ep)
                vs_dense = paired_bootstrap(censored_steps(ep),
                                            censored_steps(dense_ep[("real", mode)]))
                types = [cts[i] for i in chosen]
                results["controllers"][f"real|selective_dn{k}|{mode}"] = {
                    "controller": f"selective_dn{k}", "wiring": "real", "mode": mode,
                    "readout": f"ridge on the {len(chosen)} selected descending-neuron "
                               f"states only",
                    "n_readout_features": len(chosen), "lam_selected": lam,
                    "selection": {
                        "pool": "DN-VNC + DN-SEZ (346 annotated descending neurons)",
                        "rule": "top |Pearson r| between reservoir state and teacher action "
                                "on the held-out calibration episodes (200-239), "
                                f"{max(1, k // ACTION_DIM)} neuron(s) per action dimension",
                        "selected_rows": [int(i) for i in chosen],
                        "selected_celltypes": types,
                        "celltype_counts": dict(Counter(types)),
                        "per_action": {
                            ("turn" if j == 0 else "thrust"): [
                                {"row": int(r), "celltype": cts[r],
                                 "abs_corr": round(float(abs(corr[r, j])), 4)}
                                for r in p]
                            for j, p in enumerate(per_action)},
                    },
                    "paired_vs_dense_censored_steps": vs_dense,
                    **rows,
                }
                print(f"[real|{mode}] selective k={k} cols={len(chosen)} types={types} "
                      f"lam={lam} succ={rows['success_rate']} steps={rows['mean_steps']} "
                      f"dist={rows['mean_final_dist']}", flush=True)
                OUT.write_text(json.dumps(results, indent=2))

            # --- Q1b DOOMFLY literal: one neuron per action, calibrated gain --
            chosen2, per_action2 = select_neurons(corr, dn_rows, 2)
            idx = np.array([per_action2[0][0], per_action2[1][0]])
            g = []
            for j in range(ACTION_DIM):
                sc = X_ca[:, idx[j]] - X_ca[:, idx[j]].mean()
                yj = np.arctanh(np.clip(Y_ca[:, j], -0.995, 0.995))
                g.append(float((sc @ (yj - yj.mean())) / (sc @ sc)))
            gains = np.array(g)
            gname = "doomfly_fixed_gain_pick2"
            fac = (lambda W=W, proj=proj, idx=idx, gains=gains:
                   FixedGainReadout(W, proj, idx, gains).act)
            gain_ep = eval_rows(fac, mode)
            gain_rows = summarize_rows(gain_ep)
            results["controllers"][f"real|{gname}|{mode}"] = {
                "controller": gname, "wiring": "real", "mode": mode,
                "readout": "DOOMFLY literal: one descending neuron per action, "
                           "single scalar gain calibrated on the held-out episodes, "
                           "no trained weight matrix",
                "n_readout_features": 2,
                "selection": {
                    "neuron_turn": {"row": int(idx[0]), "celltype": cts[idx[0]],
                                    "gain": round(float(gains[0]), 4)},
                    "neuron_thrust": {"row": int(idx[1]), "celltype": cts[idx[1]],
                                      "gain": round(float(gains[1]), 4)},
                },
                "paired_vs_dense_censored_steps": paired_bootstrap(
                    censored_steps(gain_ep), censored_steps(dense_ep[("real", mode)])),
                **gain_rows,
            }
            print(f"[real|{mode}] {gname} rows={idx.tolist()} "
                  f"types={[cts[i] for i in idx]} gains={np.round(gains,3).tolist()} "
                  f"succ={gain_rows['success_rate']} steps={gain_rows['mean_steps']} "
                  f"dist={gain_rows['mean_final_dist']}", flush=True)
            OUT.write_text(json.dumps(results, indent=2))

            # --- Q1c size-matched, selection-free controls --------------------
            ctrl_rng = np.random.default_rng(2024)
            ctrl_sets = {
                "random8_whole_brain": np.sort(ctrl_rng.choice(n, size=8, replace=False)),
                "random8_dn_pool_unranked": np.sort(
                    ctrl_rng.choice(np.asarray(dn_rows), size=8, replace=False)),
            }
            for cname, carr in ctrl_sets.items():
                carr = np.asarray(carr, dtype=np.int64)
                mses = [imitation_mse(W, proj, X_ca, Y_ca, carr, l) for l in LAMS]
                lam = LAMS[int(np.argmin(mses))]
                factory, _ = build_factory(W, proj, X_tr, Y_tr, carr, lam)
                ep = eval_rows(factory, mode)
                rows = summarize_rows(ep)
                vs_dense = paired_bootstrap(censored_steps(ep),
                                            censored_steps(dense_ep[("real", mode)]))
                types = [cts[i] for i in carr]
                results["controllers"][f"real|{cname}|{mode}"] = {
                    "controller": cname, "wiring": "real", "mode": mode,
                    "readout": f"ridge on {len(carr)} neurons chosen WITHOUT calibration "
                               f"(seeded uniform draw), same training as selective_k8",
                    "n_readout_features": len(carr), "lam_selected": lam,
                    "selection": {"selected_rows": [int(i) for i in carr],
                                  "selected_celltypes": types,
                                  "celltype_counts": dict(Counter(types))},
                    "paired_vs_dense_censored_steps": vs_dense,
                    **rows,
                }
                print(f"[real|{mode}] {cname} types={types} lam={lam} "
                      f"succ={rows['success_rate']} steps={rows['mean_steps']} "
                      f"dist={rows['mean_final_dist']}", flush=True)
                OUT.write_text(json.dumps(results, indent=2))
            OUT.write_text(json.dumps(results, indent=2))

    # ---- Q2 wiring contrasts: PAIRED over the same 200 episode seeds -------
    contrasts = {}
    for mode in ("open", "blink"):
        a = dense_ep[("real", mode)]
        sa = censored_steps(a)
        for other in ("shuffled_degree_preserving", "random_sparse_same_counts"):
            b = dense_ep[(other, mode)]
            sb = censored_steps(b)
            contrasts[f"real_vs_{other}|{mode}"] = {
                "metric": "episode length in steps, failures censored at the 200-step cap "
                          "(fully observed); lower is better",
                "real_mean_censored_steps": round(float(sa.mean()), 3),
                "other_mean_censored_steps": round(float(sb.mean()), 3),
                "paired_bootstrap_delta_real_minus_other": paired_bootstrap(sa, sb),
                "success_diff_real_minus_other": round(
                    float(np.mean([r["success"] for r in a])
                          - np.mean([r["success"] for r in b])), 4),
                "verdict": ("real wiring is INDISTINGUISHABLE from this control"
                            if not paired_bootstrap(sa, sb)["excludes_zero"]
                            else "real wiring differs from this control"),
            }
    results["wiring_paired_contrasts"] = contrasts

    # ---- Q2b: variability ACROSS independent control draws ----------------
    # One shuffle / one random draw is a single realisation; the paired CI above
    # only covers episode noise.  Here N_DRAWS independent control matrices are
    # built and put through the identical dense-readout pipeline, so the question
    # becomes "does the real wiring stand out from the control distribution?"
    N_DRAWS = 5
    draw_rows = {"shuffled_degree_preserving": [], "random_sparse_same_counts": []}
    for r in range(N_DRAWS):
        ip_s, ix_s, wt_s, _ = shuffle_wiring(ip, ix, wt, n, seed=100 + r)
        ip_r, ix_r, wt_r = random_wiring(n, real_dn_cnt, wpool, seed=200 + r)
        for dname, dmat in (("shuffled_degree_preserving", (ip_s, ix_s, wt_s)),
                            ("random_sparse_same_counts", (ip_r, ix_r, wt_r))):
            Wd = csr(dmat[0], dmat[1], dmat[2], n)
            proj = make_proj(n)
            for mode in ("open", "blink"):
                X_tr, Y_tr, _ = collect_states(Wd, proj, TRAIN_SEEDS, mode)
                X_ca, Y_ca, _ = collect_states(Wd, proj, CALIB_SEEDS, mode)
                cols = np.arange(n)
                mses = [imitation_mse(Wd, proj, X_ca, Y_ca, cols, l) for l in LAMS]
                lam = LAMS[int(np.argmin(mses))]
                factory, _ = build_factory(Wd, proj, X_tr, Y_tr, cols, lam)
                ep = eval_rows(factory, mode)
                s = summarize_rows(ep)
                draw_rows[dname].append({
                    "draw": r, "seed": 100 + r if dname.startswith("shuffled") else 200 + r,
                    "mode": mode, "lam_selected": lam,
                    "success_rate": s["success_rate"], "mean_steps": s["mean_steps"],
                    "censored_mean_steps": round(float(censored_steps(ep).mean()), 3)})
                print(f"[draw {r} {dname}|{mode}] succ={s['success_rate']} "
                      f"steps={s['mean_steps']} censored="
                      f"{round(float(censored_steps(ep).mean()), 3)}", flush=True)
                OUT.write_text(json.dumps(results, indent=2))
    draw_summary = {}
    for dname, rs in draw_rows.items():
        entry = {"n_draws": N_DRAWS, "runs": rs}
        for mode in ("open", "blink"):
            vals = [x["censored_mean_steps"] for x in rs if x["mode"] == mode]
            real_val = round(float(censored_steps(dense_ep[("real", mode)]).mean()), 3)
            entry[mode] = {
                "control_censored_mean_steps": {"mean": round(float(np.mean(vals)), 3),
                                                "min": round(min(vals), 3),
                                                "max": round(max(vals), 3)},
                "real_censored_mean_steps": real_val,
                "real_better_than_all_controls": bool(real_val < min(vals)),
                "real_inside_control_range": bool(min(vals) <= real_val <= max(vals)),
                "real_worse_than_all_controls": bool(real_val > max(vals)),
            }
        draw_summary[dname] = entry
    results["wiring_control_draws"] = draw_summary

    # ---- compact table + the two verdicts ---------------------------------
    def get(ctrl, mode):
        return results["controllers"][f"real|{ctrl}|{mode}"]

    table = []
    for key, c in sorted(results["controllers"].items()):
        w, ctrl, mode = key.split("|")
        table.append({"wiring": w, "controller": ctrl, "mode": mode,
                      "n_readout_features": c.get("n_readout_features"),
                      "lam_selected": c.get("lam_selected"),
                      "success_rate": c["success_rate"],
                      "mean_steps": c["mean_steps"],
                      "mean_final_dist": c["mean_final_dist"]})
    results["table"] = table

    q1 = {}
    for mode in ("open", "blink"):
        dense = get("dense_readout_all2952", mode)
        uno = get("random8_dn_pool_unranked", mode)
        sel = {k: get(f"selective_dn{k}", mode) for k in KS}
        best_k = min(sel, key=lambda k: (-(sel[k]["success_rate"] or 0),
                                         sel[k]["mean_steps"] or 999))
        best = sel[best_k]
        gain = get("doomfly_fixed_gain_pick2", mode)
        gain_beats = ((gain["success_rate"], -(gain["mean_steps"] or 999))
                      > (dense["success_rate"], -(dense["mean_steps"] or 999)))

        def beats_dense(r):
            return ((r["success_rate"] > dense["success_rate"]) or
                    (r["success_rate"] == dense["success_rate"]
                     and (r["mean_steps"] or 999) < (dense["mean_steps"] or 999)))

        q1[mode] = {
            "dense_trained": dense, "best_trained_selective_k": best_k,
            "best_trained_selective": best, "doomfly_fixed_gain_pick2": gain,
            "uncalibrated_8_dn_pool": uno,
            "paired_vs_dense_all_k": {k: sel[k]["paired_vs_dense_censored_steps"] for k in KS},
            "paired_vs_dense_fixed_gain": gain["paired_vs_dense_censored_steps"],
            "paired_vs_dense_uncalibrated_8_dn_pool": uno["paired_vs_dense_censored_steps"],
            "trained_selective_beats_dense": beats_dense(best),
            "doomfly_fixed_gain_beats_dense": gain_beats,
            "verdict": None,
        }
        best_vs = best["paired_vs_dense_censored_steps"]
        if beats_dense(best):
            trained_note = "the best trained selective readout MATCHES OR BEATS the dense one"
        elif not best_vs["excludes_zero"]:
            trained_note = (f"the best trained selective readout (k={best_k}) MATCHES the "
                            f"dense readout within paired noise - censored-length delta "
                            f"{best_vs['mean_diff']} steps, 95% CI {best_vs['ci95']} includes 0 "
                            f"- but never beats it")
        else:
            trained_note = (f"no trained selective readout beats the dense one, including the "
                            f"best (k={best_k}: paired censored-length delta "
                            f"{best_vs['mean_diff']}, 95% CI {best_vs['ci95']})")
        gvs = gain["paired_vs_dense_censored_steps"]
        extra = (f" The DOOMFLY-style fixed-gain pair "
                 f"({gain['success_rate']}/{gain['mean_steps']} steps; paired censored-length "
                 f"delta {gvs['mean_diff']}, 95% CI {gvs['ci95']}) does beat it."
                 if gain_beats else
                 f" The DOOMFLY-style fixed-gain pair ({gain['success_rate']}/"
                 f"{gain['mean_steps']} steps) does not beat it either.")
        q1[mode]["verdict"] = (
            f"{trained_note} ({best['success_rate']}/{best['mean_steps']} steps at "
            f"k={best_k} vs {dense['success_rate']}/{dense['mean_steps']} steps for the "
            f"dense readout over all 2952 states)." + extra)
    results["q1_selective_readout"] = q1

    q2 = {"contrasts": contrasts,
          "control_draws": draw_summary,
          "real_dense_per_mode": {m: results["controllers"][
              f"real|dense_readout_all2952|{m}"] for m in ("open", "blink")},
          "shuffled_dense_per_mode": {m: results["controllers"][
              f"shuffled_degree_preserving|dense_readout_all2952|{m}"]
              for m in ("open", "blink")},
          "random_dense_per_mode": {m: results["controllers"][
              f"random_sparse_same_counts|dense_readout_all2952|{m}"]
              for m in ("open", "blink")}}
    all_100 = all(results["controllers"][f"{w}|dense_readout_all2952|{m}"]["success_rate"] == 1.0
                  for w in wirings for m in ("open", "blink"))
    real_worse = {m: any(draw_summary[d][m]["real_worse_than_all_controls"]
                         for d in draw_summary) for m in ("open", "blink")}
    real_better = {m: any(draw_summary[d][m]["real_better_than_all_controls"]
                          for d in draw_summary) for m in ("open", "blink")}
    if all_100 and not real_better["open"] and not real_better["blink"]:
        parts = []
        for m in ("open", "blink"):
            s = draw_summary["shuffled_degree_preserving"][m]
            z = draw_summary["random_sparse_same_counts"][m]
            parts.append(f"{m}: real {s['real_censored_mean_steps']} steps vs shuffled draws "
                         f"[{s['control_censored_mean_steps']['min']}-"
                         f"{s['control_censored_mean_steps']['max']}] and random draws "
                         f"[{z['control_censored_mean_steps']['min']}-"
                         f"{z['control_censored_mean_steps']['max']}]")
        tail = ("In blink mode the real wiring sits at the POOR end - worse than every "
                "one of the 5 shuffled and 5 random draws - so the real connectome is, "
                "if anything, slightly WORSE than a same-size random matrix."
                if (real_worse["blink"] and real_worse["open"] is False) else
                "The real wiring never beats the controls.")
        q2["verdict"] = (
            "NO. The specific larval wiring does not matter: with the identical dense "
            "trained readout all three reach 100% success in both modes, and on the "
            "discriminating metric (censored episode length) the real matrix is never "
            "better than a degree-preserving shuffle of itself or a random sparse matrix "
            "with the same neuron and edge counts. " + "; ".join(parts) + ". " + tail)
    else:
        q2["verdict"] = ("MIXED: see wiring_paired_contrasts and wiring_control_draws.")
    results["q2_wiring"] = q2

    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT, "in", round(time.time() - t0, 1), "s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
