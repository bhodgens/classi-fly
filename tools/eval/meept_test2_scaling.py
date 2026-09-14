"""meept test 2 - do the three SCALE knobs buy anything on meept's corpus?

Mapping 8.2 in docs/DIRECTIONS.md: Lane 1's largest single effect was scaling,
not structure; classification hit the same class of bug twice (a ridge penalty
of 1e-3 where 30 was correct, +9 points; and a projection scale that mattered
more than the network). Checklist: sweep embedding normalisation, route
threshold calibration, and any linear head's penalty.

This settles it with numbers, one knob at a time against the frozen baselines,
then the best combination.

FROZEN PROTOCOL (identical to tools/eval/axis_readout_heads.py so rows are
comparable):
  5-fold stratified CV, seed 42 (shuffle per class with
  np.random.default_rng(42), fold = position % 5).
  Route rule: top-1 probability >= its per-class calibrated threshold AND
  (top1-top2) margin > 0.05; else abstain.
  Thresholds calibrated on TRAIN-side scores only (tools/train/calibrate).
  P = route_correct/routes (routed-but-wrong STAYS in the denominator),
  C = routes/total, E2E = (route_correct + 0.868*abstained)/total.

KNOBS
  1. embedding normalisation: raw / l2 / mean-centred-then-l2 / standardised
  2. threshold calibration: target precision {0.90,0.95,0.97} x
     {per-class, single global}
  3. ridge penalty {1e-3,1e-2,0.1,1,10,30,100,1000} for BOTH the linear probe
     on embeddings and the ridge head on larval reservoir states

Reservoir states use the standard recipe (seeded int8 projection rng 42,
shape (1024,2952), scaled 0.02; s = tanh(0.8*s + W@s + drive), 4 steps;
scipy.sparse for W@s) - the same construction that reproduced
e1_corrected_results.json (routes=71, P=0.9577, E2E=0.8857).

Run:
  python3 tools/eval/meept_test2_scaling.py
  python3 tools/eval/meept_test2_scaling.py --out tools/eval/meept_test2_scaling_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "train"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

from calibrate import choose_thresholds  # noqa: E402
from train_readout import _one_hot, resolve_ridge_lambda, ridge_fit  # noqa: E402

CHAIN = 0.868
FOLDS = 5
SEED = 42
MARGIN_FLOOR = 0.05
STEPS = 4
DECAY = 0.8
IN_SCALE = 0.02

PENALTIES = (1e-3, 1e-2, 0.1, 1.0, 10.0, 30.0, 100.0, 1000.0)
TARGET_PRECISIONS = (0.90, 0.95, 0.97)
PROBE_BASELINE_LAM = 0.1  # the committed linear-probe row
STATES_BASELINE_LAM = 30.0  # the committed ridge-on-states row

BASELINES = {
    "centroid_head": {"routes": 70, "P": 0.914, "C": 0.194, "E2E": 0.8770},
    "ridge_states_lam30": {"routes": 71, "P": 0.9577, "C": 0.1967, "E2E": 0.8857},
    "linear_probe_emb_lam0.1": {"routes": 88, "P": 0.9205, "C": 0.2438, "E2E": 0.8808},
}


# ------------------------------------------------------------------- plumbing

def softmax(row):
    m = np.max(row)
    e = np.exp(row - m)
    return e / e.sum()


def strat_folds(y, classes, folds=FOLDS, seed=SEED):
    """shuffle per class with rng(seed), fold = position % folds."""
    n = len(y)
    fold = np.zeros(n, dtype=int)
    rng = np.random.default_rng(seed)
    arr = np.asarray(y)
    for cls in classes:
        idx = np.where(arr == cls)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % folds
    return fold


def normalise(X, mode):
    X = np.asarray(X, dtype=np.float64)
    if mode == "raw":
        return X
    if mode == "l2":
        n = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X / n
    if mode == "l2_centred":
        Z = X - X.mean(axis=0, keepdims=True)
        n = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        return Z / n
    if mode == "standardised":
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-6
        return (X - mu) / sd
    raise ValueError(mode)


def build_states(adj, vecs):
    """Frozen larval reservoir states (standard recipe), scipy.sparse W@s.

    Numerically identical to tools/train/states.states: s = tanh(decay*s +
    W@s + win*x), steps=4, decay=0.8, weights/127, win = int8 proj (rng 42)
    * 0.02. Vectorised over all rows at once.
    """
    n = adj["neurons"]
    D = vecs.shape[1]
    W = sp.csr_matrix(
        (np.asarray(adj["weights"], dtype=np.float64) / 127.0,
         np.asarray(adj["indices"]), np.asarray(adj["indptr"])),
        shape=(n, n))
    rng = np.random.default_rng(SEED)
    proj = rng.integers(-6, 7, size=(D, n)).astype(np.float64) * IN_SCALE
    drives = np.asarray(vecs, dtype=np.float64) @ proj  # (rows, N)
    S = np.zeros((vecs.shape[0], n), dtype=np.float64)
    for _ in range(STEPS):
        S = np.tanh(DECAY * S + (W @ S.T).T + drives)
    return S, W, proj


# ----------------------------------------------------------------------- head

class RidgeHead:
    """Closed-form ridge readout on one-hot targets (production ridge_fit)."""

    def __init__(self, lam=None):
        self.lam = lam

    def fit(self, S, y, classes):
        Y = _one_hot(y, classes)
        self.selected_lam = resolve_ridge_lambda(S, Y, self.lam)
        W, bias = ridge_fit(S, Y, lam=self.lam)
        self.W = np.asarray(W, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.classes = list(classes)
        return self

    def logits(self, S):
        return np.asarray(S, dtype=np.float64) @ self.W + self.bias

    def score(self, S):
        logits = self.logits(S)
        probs, margins = [], []
        for row in logits:
            p = softmax(row)
            order = np.sort(p)
            probs.append({self.classes[j]: float(p[j]) for j in range(len(p))})
            margins.append(float(order[-1] - order[-2]))
        return probs, margins

    def predict(self, S):
        return [self.classes[int(i)] for i in np.argmax(self.logits(S), axis=1)]


# ------------------------------------------------------------- calibration

def choose_global_threshold(train_probs, train_labels, target_precision,
                            margin_floor=MARGIN_FLOOR):
    """Single threshold for every class (the per-class chooser's rival).

    Lowest pooled candidate (41 quantiles of all train top-1 probabilities)
    whose routed-row precision meets target_precision; routed rows must also
    clear the margin floor so the threshold is picked in the same space the
    gate uses. Falls back to the max observed probability.
    """
    tops = []
    for row, lab in zip(train_probs, train_labels):
        top = max(row, key=lambda c: row[c])
        tops.append((row[top], top, lab))
    vals = sorted(v for v, _, _ in tops)
    cands = [vals[min(int(round(q * (len(vals) - 1))), len(vals) - 1)]
             for q in (i / 40.0 for i in range(41))]
    for t in cands:
        routed = [(v, top, lab) for v, top, lab in tops if v >= t]
        if not routed:
            continue
        # margin floor: recompute on the caller side is impossible here, so the
        # caller passes only rows that already cleared it in its own scoring;
        # the pooled scan below re-applies nothing extra (documented).
        correct = sum(1 for _, top, lab in routed if top == lab)
        if correct / len(routed) >= target_precision:
            return float(t)
    return float(vals[-1])


def gate_metrics(S, y, classes, fold, make_head, target_precision=0.97,
                 calibrator="perclass"):
    """Fit per fold, gate with TRAIN-calibrated thresholds, accumulate."""
    routes = route_correct = abstained = ungated_ok = 0
    per_fold = []
    for f in range(FOLDS):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        head = make_head().fit(S[tr], [y[i] for i in tr], classes)
        tr_probs, tr_margins = head.score(S[tr])
        tr_labels = [y[i] for i in tr]
        # Calibration rows are the FULL train-side scores (the frozen protocol:
        # choose_thresholds(train_rows, train_labels, target_precision)). The
        # margin floor is applied by the gate, not by the calibration.
        q_rows, q_labs = list(tr_probs), list(tr_labels)
        present = set(q_labs)
        for p, l in zip(tr_probs, tr_labels):
            if l not in present:
                q_rows.append(p)
                q_labs.append(l)
                present.add(l)
        if calibrator == "perclass":
            thr = choose_thresholds(q_rows, q_labs, target_precision=target_precision)
            thr_global = None
        else:
            thr = None
            thr_global = choose_global_threshold(q_rows, q_labs, target_precision)
        te_probs, te_margins = head.score(S[te])
        preds = head.predict(S[te])
        for k, i in enumerate(te):
            p = te_probs[k]
            top = max(classes, key=lambda c: (p[c], -classes.index(c)))
            t = thr[top] if thr is not None else thr_global
            if p[top] >= t and te_margins[k] > MARGIN_FLOOR:
                routes += 1
                if top == y[i]:
                    route_correct += 1
            else:
                abstained += 1
            if preds[k] == y[i]:
                ungated_ok += 1
        per_fold.append({"fold": int(f), "train": int(len(tr)), "test": int(len(te))})
    total = routes + abstained
    return {
        "routes": routes,
        "route_correct": route_correct,
        "abstained": abstained,
        "total": total,
        "P": round(route_correct / routes, 4) if routes else 1.0,
        "C": round(routes / total, 4),
        "A": round(route_correct / total, 4),
        "A_all": round(ungated_ok / total, 4),
        "E2E": round((route_correct + CHAIN * abstained) / total, 4),
    }


def ood_abstain(S_gold, y, classes, S_ood, make_head, target_precision=0.97):
    head = make_head().fit(S_gold, y, classes)
    probs, margins = head.score(S_gold)
    thr = choose_thresholds(probs, y, target_precision=target_precision)
    ood_probs, ood_margins = head.score(S_ood)
    ok = 0
    for k in range(len(S_ood)):
        p = ood_probs[k]
        top = max(classes, key=lambda c: (p[c], -classes.index(c)))
        if not (p[top] >= thr[top] and ood_margins[k] > MARGIN_FLOOR):
            ok += 1
    return round(ok / len(S_ood), 4) if len(S_ood) else None


# ---------------------------------------------------------------- knob sweeps

def sweep_normalisation(vecs, y, classes, fold):
    """Knob 1: embedding normalisation, probe ridge at both the baseline lam
    (0.1, comparability) and the auto inner-CV lam (fairness)."""
    rows = {}
    for mode in ("raw", "l2", "l2_centred", "standardised"):
        X = normalise(vecs, mode)
        for tag, lam in (("lam0.1", PROBE_BASELINE_LAM), ("auto", None)):
            m = gate_metrics(X, y, classes, fold, lambda l=lam: RidgeHead(l))
            m["lambda"] = PROBE_BASELINE_LAM if lam else "auto"
            rows[f"{mode}__{tag}"] = m
    return rows


def sweep_penalty_embeddings(vecs, y, classes, fold, mode):
    rows = {}
    X = normalise(vecs, mode)
    for lam in PENALTIES:
        m = gate_metrics(X, y, classes, fold, lambda l=lam: RidgeHead(l))
        m["penalty"] = lam
        rows[f"lam{lam:g}"] = m
    return rows, X


def sweep_penalty_states(S, y, classes, fold):
    rows = {}
    for lam in PENALTIES:
        m = gate_metrics(S, y, classes, fold, lambda l=lam: RidgeHead(l))
        m["penalty"] = lam
        rows[f"lam{lam:g}"] = m
    return rows


def calibration_frontier(S, y, classes, fold, make_head):
    """Knob 2: precision/coverage frontier, per-class vs single global."""
    frontier = {}
    for cal in ("perclass", "global"):
        for tp in TARGET_PRECISIONS:
            m = gate_metrics(S, y, classes, fold, make_head,
                             target_precision=tp, calibrator=cal)
            m["target_precision"] = tp
            frontier[f"{cal}__tp{tp:.2f}"] = m
    return frontier


# -------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(REPO / "data" / "e1_inputs.json"))
    ap.add_argument("--adjacency", default=str(REPO / "data" / "larval_adjacency.json"))
    ap.add_argument("--out", default=str(REPO / "tools" / "eval" /
                                         "meept_test2_scaling_results.json"))
    ap.add_argument("--states-cache", default="/tmp/meept_test2_states.npy")
    args = ap.parse_args(argv)

    t0 = time.time()
    data = json.loads(Path(args.inputs).read_text())
    adj = json.loads(Path(args.adjacency).read_text())
    gold, ood = data["gold"], data["ood"]
    classes = sorted(data["classes"])
    vecs_all = np.asarray(data["vecs"], dtype=np.float64)
    n_gold = len(gold)
    y = [g["intent"] for g in gold]
    vecs = vecs_all[:n_gold]
    vecs_ood = vecs_all[n_gold:]
    fold = strat_folds(y, classes)
    one_case_pt = round(100.0 / n_gold, 2)

    # ---- reservoir states -------------------------------------------------
    cache = Path(args.states_cache)
    if cache.exists():
        S = np.load(cache)
    else:
        S, _, _ = build_states(adj, vecs_all)
        np.save(cache, S)
    assert S.shape == (len(vecs_all), adj["neurons"])
    Sg, So = S[:n_gold], S[n_gold:]
    sat = float(np.mean(np.abs(Sg) > 0.95))
    print(f"states {S.shape} saturation={sat:.3f} ({time.time()-t0:.1f}s)")

    out = {
        "experiment": "meept test 2 - do the three scaling knobs buy anything?",
        "mapping": "docs/DIRECTIONS.md section 8.2 (scaling discipline, no new model)",
        "source": {
            "inputs": str(args.inputs), "adjacency": str(args.adjacency),
            "gold": n_gold, "ood": len(ood), "classes": classes,
            "embed_dim": int(vecs.shape[1]), "neurons": adj["neurons"],
            "edges": adj["edges"], "states_saturation": round(sat, 4),
            "embeddings_unit_norm": bool(np.allclose(
                np.linalg.norm(vecs_all, axis=1), 1.0, atol=1e-4)),
            "states_recipe": f"s = tanh({DECAY}*s + W@s + win*x), steps={STEPS}, "
                             f"win = seeded int8 projection (rng {SEED}) shape "
                             f"({vecs.shape[1]},{adj['neurons']}) * {IN_SCALE}",
        },
        "protocol": {
            "folds": FOLDS, "seed": SEED,
            "fold_rule": "shuffle per class with np.random.default_rng(42), fold = position % 5",
            "route_rule": "top-1 prob >= calibrated threshold AND (top1-top2) margin > 0.05",
            "calibration": "choose_thresholds(train rows clearing the margin floor, "
                           "target_precision) - TRAIN side only",
            "metrics": {"P": "route_correct/routes (wrong routes stay in the denominator)",
                        "C": "routes/total", "A": "route_correct/total",
                        "A_all": "ungated out-of-fold argmax accuracy over all gold",
                        "E2E": "(route_correct + 0.868*abstained)/total"},
            "one_case_pt": one_case_pt,
            "chain_floor": CHAIN,
        },
        "baselines_to_beat": BASELINES,
    }

    # ---- self-check: reproduce the committed ridges -----------------------
    chk_probe = gate_metrics(vecs, y, classes, fold,
                             lambda: RidgeHead(PROBE_BASELINE_LAM))
    chk_states = gate_metrics(Sg, y, classes, fold,
                              lambda: RidgeHead(STATES_BASELINE_LAM))
    print(f"[check] probe lam0.1   {chk_probe}")
    print(f"[check] states lam30   {chk_states}")
    out["self_check"] = {
        "linear_probe_emb_lam0.1": chk_probe,
        "expected_probe": BASELINES["linear_probe_emb_lam0.1"],
        "ridge_states_lam30": chk_states,
        "expected_states": BASELINES["ridge_states_lam30"],
        "probe_reproduced": (chk_probe["routes"] == 88 and chk_probe["P"] == 0.9205
                             and chk_probe["E2E"] == 0.8808),
        "states_reproduced": (chk_states["routes"] == 71 and chk_states["P"] == 0.9577
                              and chk_states["E2E"] == 0.8857),
    }
    print(f"self-check: probe={out['self_check']['probe_reproduced']} "
          f"states={out['self_check']['states_reproduced']}")

    # ---- knob 1: normalisation -------------------------------------------
    knob1 = sweep_normalisation(vecs, y, classes, fold)
    out["knob1_normalisation"] = {
        "note": "linear probe (ridge) on embeddings; lam0.1 is the committed "
                "baseline penalty, auto is the inner-CV choice. 'raw' and 'l2' "
                "coincide because the stored embeddings are already unit-norm.",
        "rows": knob1,
    }
    print("\nKNOB 1 normalisation (probe lam0.1 | auto):")
    for k, v in knob1.items():
        print(f"  {k:22s} routes={v['routes']:3d} P={v['P']:.4f} C={v['C']:.4f} "
              f"A_all={v['A_all']:.4f} E2E={v['E2E']:.4f}")

    # ---- knob 3a/1 joint grid: normalisation x penalty for the probe ------
    grid = {}
    for mode in ("raw", "l2", "l2_centred", "standardised"):
        rows, _ = sweep_penalty_embeddings(vecs, y, classes, fold, mode)
        for lam_key, v in rows.items():
            grid[f"{mode}__{lam_key}"] = v
    out["knob1_x_knob3_probe_grid"] = grid

    # ---- knob 3b: penalty on reservoir states -----------------------------
    knob3_states = sweep_penalty_states(Sg, y, classes, fold)
    out["knob3_penalty_states"] = knob3_states
    print("\nKNOB 3 penalty, ridge on larval reservoir states:")
    for k, v in knob3_states.items():
        print(f"  {k:10s} routes={v['routes']:3d} P={v['P']:.4f} C={v['C']:.4f} "
              f"A_all={v['A_all']:.4f} E2E={v['E2E']:.4f}")

    # ---- best configuration ----------------------------------------------
    # Carriers: best states penalty; best probe (normalisation, penalty).
    best_states_key = max(knob3_states, key=lambda k: (knob3_states[k]["E2E"],
                                                       knob3_states[k]["A_all"]))
    best_grid_key = max(grid, key=lambda k: (grid[k]["E2E"], grid[k]["A_all"]))
    best_states_lam = float(best_states_key[3:])
    mode_best, lam_best = best_grid_key.split("__")
    lam_best = float(lam_best[3:])
    print(f"\nbest states penalty: {best_states_key} -> {knob3_states[best_states_key]}")
    print(f"best probe config: {best_grid_key} -> {grid[best_grid_key]}")

    # ---- knob 2: calibration frontier on both carriers --------------------
    front_states = calibration_frontier(
        Sg, y, classes, fold, lambda: RidgeHead(best_states_lam))
    Xbest = normalise(vecs, mode_best)
    front_probe = calibration_frontier(
        Xbest, y, classes, fold, lambda: RidgeHead(lam_best))
    out["knob2_calibration"] = {
        "note": "frontier over target precision x per-class/global threshold, "
                "run on the best states carrier and the best probe carrier",
        "carrier_states": {"penalty": best_states_lam, "rows": front_states},
        "carrier_probe": {"normalisation": mode_best, "penalty": lam_best,
                          "rows": front_probe},
    }
    print("\nKNOB 2 calibration frontier (states carrier):")
    for k, v in front_states.items():
        print(f"  {k:22s} routes={v['routes']:3d} P={v['P']:.4f} C={v['C']:.4f} "
              f"E2E={v['E2E']:.4f}")
    print("KNOB 2 calibration frontier (probe carrier):")
    for k, v in front_probe.items():
        print(f"  {k:22s} routes={v['routes']:3d} P={v['P']:.4f} C={v['C']:.4f} "
              f"E2E={v['E2E']:.4f}")

    # pick overall best row across all gated grids
    all_rows = []
    for section, rows in (("knob1", knob1), ("grid", grid),
                          ("states", knob3_states), ("front_states", front_states),
                          ("front_probe", front_probe)):
        for k, v in rows.items():
            all_rows.append((section, k, v))
    best_section, best_key, best_row = max(
        all_rows, key=lambda t: (t[2]["E2E"], t[2]["A_all"]))
    # Resolve the winning row's calibration settings (defaults: per-class, tp0.97).
    tp, cal = 0.97, "perclass"
    if "__tp" in best_key:
        tp = float(best_key.split("__tp")[1])
        cal = best_key.split("__")[0]
    if best_section in ("states", "front_states"):
        final_make = lambda: RidgeHead(best_states_lam)
        final_S = Sg
        final_Sood = So
        final_desc = (f"ridge on larval reservoir states, penalty {best_states_lam}, "
                      f"{cal} thresholds @ tp{tp:.2f}")
    else:
        lam_final = None if best_key.endswith("__auto") else lam_best
        final_make = lambda: RidgeHead(lam_final)
        final_S = normalise(vecs, mode_best)
        final_Sood = normalise(vecs_ood, mode_best)
        lam_txt = "auto inner-CV" if lam_final is None else f"penalty {lam_final}"
        final_desc = (f"linear probe on {mode_best} embeddings, {lam_txt}, "
                      f"{cal} thresholds @ tp{tp:.2f}")
    if best_section == "knob1":
        pass

    best_ood = ood_abstain(final_S, y, classes, final_Sood, final_make)
    best_row = dict(best_row)
    best_row["OOD_R"] = best_ood
    best_row["config"] = final_desc
    out["best_combination"] = {
        "section": best_section, "key": best_key, "row": best_row,
        "description": final_desc,
    }

    # ---- verdicts ---------------------------------------------------------
    def delta_vs(base, row):
        return {"dE2E": round(row["E2E"] - base["E2E"], 4),
                "droutes": row["routes"] - base["routes"],
                "dP": round(row["P"] - base["P"], 4)}

    probe_base = BASELINES["linear_probe_emb_lam0.1"]
    states_base = BASELINES["ridge_states_lam30"]

    norm_rows = {m: knob1[f"{m}__lam0.1"] for m in
                 ("raw", "l2", "l2_centred", "standardised")}
    norm_spread = max(v["E2E"] for v in norm_rows.values()) - \
        min(v["E2E"] for v in norm_rows.values())
    norm_best = max(norm_rows, key=lambda m: norm_rows[m]["E2E"])

    states_pen_spread = max(v["E2E"] for v in knob3_states.values()) - \
        min(v["E2E"] for v in knob3_states.values())
    probe_pen = {k: v for k, v in grid.items() if k.startswith("l2__")}
    probe_pen_spread = max(v["E2E"] for v in probe_pen.values()) - \
        min(v["E2E"] for v in probe_pen.values())

    cal_rows = list(front_states.values()) + list(front_probe.values())
    cal_spread = max(v["E2E"] for v in cal_rows) - min(v["E2E"] for v in cal_rows)

    def verdict_of(spread, threshold=0.005):
        return "KNOB MATTERS" if spread > threshold else "NEUTRAL"

    verdict = {
        "knob1_normalisation": verdict_of(norm_spread),
        "knob2_calibration": verdict_of(cal_spread),
        "knob3_penalty": verdict_of(max(states_pen_spread, probe_pen_spread)),
        "spreads": {
            "knob1_normalisation_E2E": round(norm_spread, 4),
            "knob2_calibration_E2E": round(cal_spread, 4),
            "knob3_penalty_states_E2E": round(states_pen_spread, 4),
            "knob3_penalty_probe_E2E": round(probe_pen_spread, 4),
        },
        "deltas_vs_baselines": {
            "best_vs_probe_lam0.1": delta_vs(probe_base, best_row),
            "best_vs_states_lam30": delta_vs(states_base, best_row),
            "states_best_vs_states_lam30": delta_vs(states_base,
                                                    knob3_states[best_states_key]),
        },
    }
    verdict["plain"] = (
        "Normalisation: {v1} (E2E spread {s1:.4f} over raw/l2/l2-centred/"
        "standardised at the baseline penalty). Calibration: {v2} (E2E spread "
        "{s2:.4f} across tp 0.90-0.97 x per-class/global on both carriers). "
        "Penalty: {v3} (E2E spread {s3:.4f} on states, {s4:.4f} on the probe). "
        "Best configuration: {desc} -> routes {r}, P {p}, C {c}, A_all {aa}, "
        "E2E {e2e}; vs the committed probe row ({br} routes, E2E {be}), vs the "
        "committed states row ({sr} routes, E2E {se})."
    ).format(
        v1=verdict["knob1_normalisation"], s1=norm_spread,
        v2=verdict["knob2_calibration"], s2=cal_spread,
        v3=verdict["knob3_penalty"], s3=states_pen_spread,
        s4=probe_pen_spread,
        desc=final_desc, r=best_row["routes"], p=best_row["P"],
        c=best_row["C"], aa=best_row["A_all"], e2e=best_row["E2E"],
        br=probe_base["routes"], be=probe_base["E2E"],
        sr=states_base["routes"], se=states_base["E2E"],
    )
    out["verdict"] = verdict
    out["runtime_s"] = round(time.time() - t0, 1)
    out["one_case_pt"] = one_case_pt

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out} ({time.time()-t0:.1f}s)")
    print("verdict:", json.dumps(out["verdict"]["spreads"], indent=1))
    print("knob verdicts:", {k: out["verdict"][k] for k in
                             ("knob1_normalisation", "knob2_calibration",
                              "knob3_penalty")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
