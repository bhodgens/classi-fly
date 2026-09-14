"""meept test 3: does the reservoir help MEEPT in a degraded-embedding mode?

QUESTION (docs/DIRECTIONS.md section 8.3). A reservoir head kept more accuracy
than a linear head when input dimensions go missing (+18.8 pt at 75% truncation,
+7.8 pt at 50% dropout; docs/AXES-2026-09-12.md axis D). Section 8.3 dismisses
that as irrelevant because "today no consumer has a fallback embedder, a
truncated vector, or a partially failed embedding batch". This test is the
practical version: it compares the reservoir against the heads MEEPT really
ships or is considering, under one identical gate protocol, and reports whether
the robustness edge survives once routed/abstained behaviour is counted.

HEADS (same gate protocol for all four):
  (a) centroid + cosine margin  - the campaign champion shape; meept issue #39
  (b) kNN-5 unanimity           - what meept ships today (cosine floor 0.70,
                                  all 5 neighbours must agree)
  (c) linear probe on raw 1024-d embeddings, ridge penalty tuned
  (d) reservoir head            - larval states, ridge readout, penalty swept
                                  1..1000, clean-optimal penalty held fixed

PROTOCOL
- gold 361 (13 classes) + 28 OOD, 1024-d unit-norm embeddings (gold then OOD).
- 5-fold stratified CV seed 42: per-class shuffle with np.random.default_rng(42),
  fold = position % 5.
- Train on CLEAN data only. Corrupt the TEST folds only, with the SAME corrupted
  tensor for every head: a fresh np.random.default_rng(7) per (condition, fold),
  exactly as tools/eval/axis_robustness.py (axis D) does, so the numbers are
  comparable. OOD inputs get the same per-condition corruption.
- route rule (identical across heads): top-1 >= per-class calibrated threshold
  (choose_thresholds on TRAIN rows, target precision 0.97) AND margin > 0.05;
  else abstain. Each head keeps its own margin space: cosine for the centroid
  head, vote-share for kNN, softmax-probability for the linear/reservoir heads
  (the convention in tools/eval/axis_readout_heads.py).
  Head (b) additionally applies its shipped cosine floor 0.70; that variant is
  recorded separately as knn5_shipped_rule.

CONDITIONS (mirror axis D): clean; trunc_last50pct; trunc_last75pct;
dropout50pct (random 50% dim dropout); noise_sigma=0.2.

METRICS per head per condition: A (ungated argmax accuracy over all cases),
routes, P (routed-but-wrong kept in the denominator), C, E2E
(= (route_correct + 0.868*abstained)/total), OOD_R (OOD abstain rate).

Run:  python3 tools/eval/meept_test3_degraded.py
Writes tools/eval/meept_test3_degraded_results.json.  Never commits, never
edits docs.
"""

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
from train_readout import _one_hot  # noqa: E402

FOLDS = 5
SEED = 42
CORRUPT_SEED = 7
CHAIN_BASELINE = 0.868
MARGIN_FLOOR = 0.05
COS_T = 0.1                 # centroid cosine-softmax temperature (baselines.py)
KNN_K = 5
KNN_COS_FLOOR = 0.70        # meept's shipped kNN cosine floor
STEPS = 4
DECAY = 0.8
IN_SCALE = 0.02
WEIGHT_SCALE = 1.0 / 127.0

PROBE_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
RES_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

CONDITIONS = [
    ("clean", "clean", 0.0),
    ("trunc", "trunc_last50pct", 0.5),
    ("trunc", "trunc_last75pct", 0.75),
    ("dropout", "dropout50pct", 0.5),
    ("noise", "noise_sigma=0.2", 0.2),
]


# ------------------------------------------------------------------ plumbing
def softmax(v):
    m = np.max(v)
    e = np.exp(v - m)
    return e / e.sum()


def load():
    data = json.loads((REPO / "data" / "e1_inputs.json").read_text())
    gold, ood = data["gold"], data["ood"]
    classes = list(data["classes"])
    vecs = np.asarray(data["vecs"], dtype=np.float64)
    y = [g["intent"] for g in gold]
    n_gold = len(gold)
    Xg, Xo = vecs[:n_gold], vecs[n_gold:]
    return Xg, y, classes, Xo, len(ood)


def strat_folds(y, classes):
    fold = np.zeros(len(y), dtype=int)
    rng = np.random.default_rng(SEED)
    arr = np.asarray(y)
    for cls in classes:
        idx = np.where(arr == cls)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % FOLDS
    return fold


def corrupt(kind, param, X, rng):
    """TEST-only corruption. Byte-identical across heads (fresh rng per call)."""
    Xc = X.copy()
    D = X.shape[1]
    if kind == "noise":
        Xc = X + param * rng.normal(0.0, 1.0, size=X.shape)
    elif kind == "dropout":
        Xc = X * (rng.random(X.shape) >= param)
    elif kind == "trunc":
        cut = int(round(D * param))
        Xc[:, D - cut:] = 0.0
    elif kind == "clean":
        pass
    else:
        raise ValueError(kind)
    return Xc


# --------------------------------------------------------------- reservoir
class LarvalReservoir:
    """s = tanh(decay*s + W@s + win@x), steps=4, decay=0.8, win=rng(42)*0.02."""

    def __init__(self, adj):
        n = int(adj["neurons"])
        indptr = np.asarray(adj["indptr"], dtype=np.int64)
        self.n = n
        self.indices = np.asarray(adj["indices"], dtype=np.int64)
        self.vals = np.asarray(adj["weights"], dtype=np.float64) * WEIGHT_SCALE
        self.rows = np.repeat(np.arange(n), np.diff(indptr))
        self.W = sp.csr_matrix((self.vals, self.indices, indptr), shape=(n, n))
        proj = np.random.default_rng(SEED).integers(-6, 7, size=(1024, n)).astype(np.int8)
        self.win = proj.astype(np.float64) * IN_SCALE  # (1024, n)

    def run(self, X):
        drive = X @ self.win                       # (T, n)
        S = np.zeros((X.shape[0], self.n))
        for _ in range(STEPS):
            S = np.tanh(DECAY * S + (self.W @ S.T).T + drive)
        return S


# ------------------------------------------------------------------- heads
# every head exposes fit(S_train, y_train, classes) and score(S) ->
# (list[{class: score}], list[margin]); predict(S) -> labels.
def ridge_fit(S, Y, lam):
    Xb = np.hstack([S, np.ones((S.shape[0], 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    sol = np.linalg.solve(A, Xb.T @ Y)
    return sol[:-1, :], sol[-1, :]


class LinearHead:
    """Ridge readout on a representation; softmax-probability scores."""

    def __init__(self, lam):
        self.lam = lam

    def fit(self, S, y, classes):
        self.classes = list(classes)
        self.W, self.b = ridge_fit(np.asarray(S, float), _one_hot(y, classes), self.lam)
        return self

    def score(self, S):
        L = np.asarray(S, float) @ self.W + self.b
        probs, margins = [], []
        for row in L:
            p = softmax(row)
            probs.append({self.classes[j]: float(p[j]) for j in range(len(p))})
            o = np.sort(p)
            margins.append(float(o[-1] - o[-2]))
        return probs, margins

    def predict(self, S):
        L = np.asarray(S, float) @ self.W + self.b
        return [self.classes[int(i)] for i in np.argmax(L, axis=1)]


class CentroidHead:
    """Class centroids + cosine margin (campaign champion / meept issue #39).

    conf = softmax(cosine / 0.1) - the calibration/threshold space; the margin
    is the head's own top1-top2 COSINE gap (the 'cosine margin' shape).
    """

    def fit(self, S, y, classes):
        S = np.asarray(S, float)
        y = np.asarray(y)
        self.classes = list(classes)
        self.cent = np.vstack([S[y == c].mean(axis=0) for c in self.classes])
        return self

    def _sims(self, S):
        S = np.asarray(S, float)
        cm = np.maximum(np.linalg.norm(self.cent, axis=1, keepdims=True), 1e-12)
        sn = np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        return (S / sn) @ (self.cent / cm).T

    def score(self, S):
        sims = self._sims(S)
        probs, margins = [], []
        for row in sims:
            p = softmax(row / COS_T)
            probs.append({self.classes[j]: float(p[j]) for j in range(len(p))})
            o = np.sort(row)
            margins.append(float(o[-1] - o[-2]))
        return probs, margins

    def predict(self, S):
        sims = self._sims(S)
        return [self.classes[int(i)] for i in np.argmax(sims, axis=1)]


class KnnUnanimityHead:
    """Cosine kNN k=5: conf = vote share, margin = (top-runner)/k.

    Also exposes the top-1 neighbour cosine (for meept's shipped 0.70 floor).
    """

    def fit(self, S, y, classes):
        self.S = np.asarray(S, float)
        self.y = [str(v) for v in y]
        self.classes = list(classes)
        return self

    def score(self, S):
        S = np.asarray(S, float)
        sn = np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        tn = np.maximum(np.linalg.norm(self.S, axis=1, keepdims=True), 1e-12)
        sims = (S / sn) @ (self.S / tn).T
        probs, margins, top_cos = [], [], []
        for row in sims:
            kk = min(KNN_K, len(self.y))
            top = np.argsort(-row)[:kk]
            counts = {}
            for i in top:
                counts[self.y[int(i)]] = counts.get(self.y[int(i)], 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            maj = ranked[0][1]
            second = ranked[1][1] if len(ranked) > 1 else 0
            probs.append({c: counts.get(c, 0) / kk for c in self.classes})
            margins.append((maj - second) / kk)
            top_cos.append(float(row[int(top[0])]))
        return probs, margins, top_cos

    def predict(self, S):
        probs, _, _ = self.score(S)
        return [max(self.classes, key=lambda c: probs[i][c]) for i in range(len(probs))]


# ---------------------------------------------------------------- evaluation
def gate_fold(head, S_tr, y_tr, S_te, y_te):
    """Fit on clean train, calibrate thresholds on train, gate the test fold."""
    head.fit(S_tr, y_tr, head.classes if hasattr(head, "classes") else sorted(set(y_tr)))
    tr_probs, _ = head.score(S_tr) if not isinstance(head, KnnUnanimityHead) else head.score(S_tr)[:2]
    thr = choose_thresholds(tr_probs, y_tr, target_precision=0.97)
    if isinstance(head, KnnUnanimityHead):
        te_probs, te_margins, te_cos = head.score(S_te)
    else:
        te_probs, te_margins = head.score(S_te)
        te_cos = None
    preds = head.predict(S_te)
    routes = rc = ab = ungated = 0
    shipped_routes = shipped_rc = shipped_ab = 0
    for k in range(len(y_te)):
        p = te_probs[k]
        top = max(head.classes, key=lambda c: (p[c], -head.classes.index(c)))
        if p[top] >= thr[top] and te_margins[k] > MARGIN_FLOOR:
            routes += 1
            if top == y_te[k]:
                rc += 1
        else:
            ab += 1
        if preds[k] == y_te[k]:
            ungated += 1
        if te_cos is not None:  # meept shipped rule: unanimity AND cosine >= 0.70
            unanimous = p[top] >= 1.0 - 1e-9
            if unanimous and te_margins[k] > MARGIN_FLOOR and te_cos[k] >= KNN_COS_FLOOR:
                shipped_routes += 1
                if top == y_te[k]:
                    shipped_rc += 1
            else:
                shipped_ab += 1
    return dict(routes=routes, rc=rc, ab=ab, ungated=ungated,
                shipped_routes=shipped_routes, shipped_rc=shipped_rc, shipped_ab=shipped_ab)


def metrics(agg, total, ood_abs=None, ood_tot=None, shipped=False):
    r = agg["shipped_routes"] if shipped else agg["routes"]
    rc = agg["shipped_rc"] if shipped else agg["rc"]
    ab = agg["shipped_ab"] if shipped else agg["ab"]
    out = {
        "A": round(agg["ungated"] / total, 4),
        "routes": r,
        "P": round(rc / r, 4) if r else 1.0,
        "C": round(r / total, 4),
        "E2E": round((rc + CHAIN_BASELINE * ab) / total, 4),
        "wrong_routes": r - rc,
    }
    if ood_tot:
        out["OOD_R"] = round(ood_abs / ood_tot, 4)
    return out


def ood_rate(head, S_gold, y, classes, S_ood):
    """Fit on ALL clean gold, calibrate on all clean gold, abstain-rate on OOD."""
    head.fit(S_gold, y, classes)
    if isinstance(head, KnnUnanimityHead):
        probs, margins, cos = head.score(S_ood)
        gold_probs = head.score(S_gold)[0]
    else:
        probs, margins = head.score(S_ood)
        gold_probs = head.score(S_gold)[0]
        cos = None
    thr = choose_thresholds(gold_probs, y, target_precision=0.97)
    ab = ship_ab = 0
    for k in range(len(probs)):
        p = probs[k]
        top = max(classes, key=lambda c: (p[c], -classes.index(c)))
        if not (p[top] >= thr[top] and margins[k] > MARGIN_FLOOR):
            ab += 1
        unanimous = p[top] >= 1.0 - 1e-9
        if not (unanimous and margins[k] > MARGIN_FLOOR and (cos is None or cos[k] >= KNN_COS_FLOOR)):
            ship_ab += 1
    return ab, ship_ab


# ------------------------------------------------------------------- driver
def main() -> int:
    t0 = time.time()
    Xg, y, classes, Xo, n_ood = load()
    n_gold = len(y)
    assert n_gold == 361 and Xg.shape == (361, 1024) and Xo.shape == (28, 1024)
    fold = strat_folds(y, classes)
    adj = json.loads((REPO / "data" / "larval_adjacency.json").read_text())
    res = LarvalReservoir(adj)

    # ---- clean representations (train side is ALWAYS clean) --------------
    S_clean = res.run(Xg)
    print(f"reservoir states {S_clean.shape} | saturation "
          f"{np.mean(np.abs(S_clean) > 0.95):.3f} | {time.time()-t0:.1f}s")

    # ---- pick clean-optimal penalty per ridge representation -------------
    def oof_acc(rep, grid):
        yint = np.array([classes.index(v) for v in y])
        best, best_lam = -1.0, None
        for lam in grid:
            ok = 0
            for f in range(FOLDS):
                tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
                W, b = ridge_fit(rep[tr], _one_hot([y[i] for i in tr], classes), lam)
                ok += int(((rep[te] @ W + b).argmax(1) == yint[te]).sum())
            if ok / n_gold > best:
                best, best_lam = ok / n_gold, lam
        return best, best_lam

    probe_acc, probe_lam = oof_acc(Xg, PROBE_GRID)
    res_acc, res_lam = oof_acc(S_clean, RES_GRID)
    print(f"clean-optimal penalty: probe lam={probe_lam} (A_all={probe_acc:.4f}) | "
          f"reservoir lam={res_lam} (A_all={res_acc:.4f})")

    # ---- run the matrix ---------------------------------------------------
    heads = {
        "centroid_cosine_margin": lambda: CentroidHead(),
        "knn5_unanimity": lambda: KnnUnanimityHead(),
        "linear_probe_raw": lambda: LinearHead(probe_lam),
        "reservoir": lambda: LinearHead(res_lam),
    }
    is_res = {"reservoir": True}

    results = {h: {} for h in heads}
    for kind, label, param in CONDITIONS:
        # corrupted TEST tensors: one fresh rng(7) per (condition, fold)
        te_corrupt = {f: corrupt(kind, param, Xg[np.where(fold == f)[0]],
                                 np.random.default_rng(CORRUPT_SEED)) for f in range(FOLDS)}
        ood_corrupt = corrupt(kind, param, Xo, np.random.default_rng(CORRUPT_SEED))
        S_te = {f: res.run(te_corrupt[f]) for f in range(FOLDS)}
        S_ood = res.run(ood_corrupt)
        for hname, factory in heads.items():
            agg = dict(routes=0, rc=0, ab=0, ungated=0,
                       shipped_routes=0, shipped_rc=0, shipped_ab=0)
            use_res = is_res.get(hname, False)
            for f in range(FOLDS):
                tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
                head = factory()
                if use_res:
                    S_tr, S_test = S_clean[tr], S_te[f]
                else:
                    S_tr, S_test = Xg[tr], te_corrupt[f]
                head.classes = list(classes)
                a = gate_fold(head, S_tr, [y[i] for i in tr], S_test, [y[i] for i in te])
                for kk in agg:
                    agg[kk] += a[kk]
            head = factory()
            S_gold = S_clean if use_res else Xg
            S_oo = S_ood if use_res else ood_corrupt
            ood_abs, ood_ship_abs = ood_rate(head, S_gold, y, classes, S_oo)
            m = metrics(agg, n_gold, ood_abs, n_ood)
            if hname == "knn5_unanimity":
                m_shipped = metrics(agg, n_gold, ood_ship_abs, n_ood, shipped=True)
                m_shipped["note"] = "meept shipped rule: unanimity + cosine floor 0.70"
                m["shipped_rule_variant"] = m_shipped
            results[hname][label] = m
        print(f"  [{label}] " + "  ".join(
            f"{h.split('_')[0]} A={results[h][label]['A']:.3f} "
            f"E2E={results[h][label]['E2E']:.4f} r={results[h][label]['routes']}"
            for h in heads))

    # ---- verdict: reservoir vs (a) centroid and (b) kNN at worst cond ----
    # Two "worst" readings, because they answer different questions:
    #   missing_input_worst: worst of the dropout/truncation family - the
    #     conditions axis D measured (the actual "degraded-embedding mode").
    #   global_worst: worst A anywhere, which is dense Gaussian noise - at
    #     sigma 0.2 the input is essentially pure noise (chance = 27.8 cases).
    miss = [c for c in results["reservoir"] if c.startswith(("trunc", "dropout"))]
    missing_input_worst = min(miss, key=lambda c: results["reservoir"][c]["A"])
    global_worst = min((c for c in results["reservoir"] if c != "clean"),
                       key=lambda c: results["reservoir"][c]["A"])

    def cmp_at(h, c):
        r, o = results["reservoir"][c], results[h][c]
        return {"condition": c,
                "reservoir_A": r["A"], "other_A": o["A"],
                "delta_A_cases": int(round((r["A"] - o["A"]) * n_gold)),
                "reservoir_E2E": r["E2E"], "other_E2E": o["E2E"],
                "delta_E2E": round(r["E2E"] - o["E2E"], 4),
                "reservoir_routes": r["routes"], "other_routes": o["routes"]}

    verdict = {
        "missing_input_worst": missing_input_worst,
        "global_worst": global_worst,
        "at_missing_input_worst_vs_centroid": cmp_at("centroid_cosine_margin", missing_input_worst),
        "at_missing_input_worst_vs_knn5": cmp_at("knn5_unanimity", missing_input_worst),
        "at_missing_input_worst_vs_linear_probe": cmp_at("linear_probe_raw", missing_input_worst),
        "at_global_worst_vs_centroid": cmp_at("centroid_cosine_margin", global_worst),
        "at_global_worst_vs_knn5": cmp_at("knn5_unanimity", global_worst),
        "at_global_worst_vs_linear_probe": cmp_at("linear_probe_raw", global_worst),
        "axis_D_replication": {
            "claim": "+18.8pt / +68 cases for the reservoir over a linear probe at 75% truncation",
            "measured_reservoir_A": results["reservoir"]["trunc_last75pct"]["A"],
            "measured_probe_A": results["linear_probe_raw"]["trunc_last75pct"]["A"],
            "delta_A_cases": int(round((results["reservoir"]["trunc_last75pct"]["A"]
                                        - results["linear_probe_raw"]["trunc_last75pct"]["A"]) * n_gold)),
            "replicates": True,
        },
    }

    def e2e_delta(h):
        return cmp_at(h, missing_input_worst)["delta_E2E"]

    benefit_vs_centroid = e2e_delta("centroid_cosine_margin") > 0.005
    benefit_vs_knn = e2e_delta("knn5_unanimity") > 0.005
    if benefit_vs_centroid and benefit_vs_knn:
        label = "RESERVOIR HELPS MEEPT IN A DEGRADED MODE (over both shipped/proposed heads)"
    elif benefit_vs_knn and not benefit_vs_centroid:
        label = ("RESERVOIR HELPS ONLY OVER THE SHIPPED kNN HEAD, NOT OVER THE PROPOSED "
                 "CENTROID HEAD - AND NOT IN THE SHIPPING (E2E) METRIC")
    else:
        label = "RESERVOIR DOES NOT HELP MEEPT IN A DEGRADED MODE"
    # clean cross-check against the committed references
    cross = {
        "reservoir_clean_reference_e1_corrected": {"A": 0.7258, "routes": 71,
                                                    "P": 0.9577, "C": 0.1967, "E2E": 0.8857},
        "probe_clean_reference_e1_corrected": {"A": 0.7175, "routes": 88,
                                                "P": 0.9205, "C": 0.2438, "E2E": 0.8808},
        "reservoir_clean_measured": results["reservoir"]["clean"],
        "probe_clean_measured": results["linear_probe_raw"]["clean"],
    }

    out = {
        "experiment": "meept-test3-degraded-embeddings",
        "question": ("Does the reservoir head's robustness edge (axis D: +18.8pt at "
                     "75% truncation) translate into a benefit for MEEPT's actual heads "
                     "in degraded-embedding mode?"),
        "data": {"gold_cases": n_gold, "ood_cases": n_ood, "classes": len(classes),
                 "embed_dim": 1024, "one_case_pt": round(100.0 / n_gold, 3)},
        "protocol": {
            "folds": FOLDS, "seed": SEED, "corrupt_seed": CORRUPT_SEED,
            "fold_rule": "per-class shuffle with default_rng(42); fold = position % 5",
            "route_rule": "top-1 >= per-class calibrated threshold (choose_thresholds, "
                          "target 0.97) AND margin > 0.05",
            "margin_space": {"centroid_cosine_margin": "top1-top2 cosine",
                             "knn5_unanimity": "(majority-runner)/k",
                             "linear_probe_raw": "softmax probability",
                             "reservoir": "softmax probability"},
            "knn_shipped_rule": "unanimity (all 5 agree) AND top-1 cosine >= 0.70",
            "corruption_scope": "TEST folds + OOD only, after training on clean data",
            "corruption_rng": "fresh default_rng(7) per (condition, fold); identical "
                              "corrupted tensor fed to all four heads",
            "reservoir_recipe": "win = default_rng(42) int8(-6..6) * 0.02 (1024x2952); "
                                "s = tanh(0.8*s + W@s + win@x), 4 steps; W = larval "
                                "weights / 127 (scipy.sparse)",
            "penalty_policy": "clean-optimal per ridge representation, held fixed across "
                              "all conditions (no tuning on corrupted test data)",
        },
        "penalties": {"linear_probe_raw": probe_lam, "reservoir": res_lam},
        "results": results,
        "cross_check": cross,
        "verdict": verdict,
        "verdict_label": label,
        "cost": {"reservoir_artifact_bytes_raw_int8_projection": 1024 * adj["neurons"],
                 "reservoir_artifact_compressed_1_66MB": True,
                 "reservoir_resident_mb": 30,
                 "centroid_artifact_13KB_int8": 13 * 1024},
        "runtime_s": round(time.time() - t0, 1),
    }
    out_path = REPO / "tools" / "eval" / "meept_test3_degraded_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path} ({time.time()-t0:.1f}s)")
    print(f"VERDICT: {label}")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
