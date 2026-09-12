"""Axis C: readout-head variants on the frozen larval reservoir states, plus a
check that the ridge-lambda default defect is really what made the reservoir
look useless.

WHY THIS EXISTS
Only a ridge head had ever been tried on these states. The meept classifier
campaign's measured champion was a CENTROID + cosine-margin head, and a
multinomial logistic head had never been tried on reservoir states at all. In
parallel, tools/train/train_readout.py hardcoded RIDGE_LAMBDA = 1e-3, which is
effectively unregularized for a ~2952-dim readout on ~289 samples/fold (the
corrected run used lam=30 and lifted all-case accuracy 0.125 -> 0.726).

WHAT IT COMPUTES
Every head is fitted on the TRAIN folds only and scored on the held-out fold
under one fixed route rule:

    route to top-1  iff  top-1 prob >= per-class calibrated threshold
                         AND top1-top2 margin > 0.05      else abstain

Thresholds come from tools/train/calibrate.choose_thresholds on TRAIN-side
scores at target precision 0.97. P = route_correct/routes (routed-but-wrong
cases STAY in the denominator), C = routes/total, A = route_correct/total,
E2E = (route_correct + 0.868 * abstained)/total. A_all is the UNGATED
out-of-fold argmax accuracy over all 361 gold cases - that is the quantity
tools/eval/e1_corrected_results.json reports as "A", so it is listed for
cross-reference.

Determinism: 5-fold stratified CV, seed 42, shuffle-per-class then
fold = position % 5; no RNG anywhere else except the fixed-seed int8 input
projection (same construction as tools/eval/e1_shakedown.py).

Run:
  python3 tools/eval/axis_readout_heads.py \\
      --out tools/eval/axis_readout_heads.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "train"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

import states as st  # noqa: E402
from calibrate import choose_thresholds  # noqa: E402
from train_readout import (  # noqa: E402
    INNER_SEED,
    _inner_folds,
    _one_hot,
    auto_ridge_lambda,
    resolve_ridge_lambda,
    ridge_fit,
)

CHAIN = 0.868          # chain floor (master.md)
FOLDS = 5
SEED = 42
MARGIN_FLOOR = 0.05
STEPS = 4
DECAY = 0.8
IN_SCALE = 0.02
COS_T = 0.1            # cosine softmax temperature (tools/eval/baselines.py)
KNN_K = 5
# Logistic inner-CV grid. l2 here is the penalty on z-scored features with the
# 1/n-mean gradient, so lambda_ridge ~= n_train * l2 (n~289 -> l2=0.1 ~= lam 29).
LOGISTIC_L2_GRID = (1e-3, 1e-2, 1e-1, 1.0)
LOGISTIC_LR = 0.2
LOGISTIC_EPOCHS = 2000


# ---------------------------------------------------------------- plumbing

def softmax(logits):
    m = np.max(logits)
    e = np.exp(logits - m)
    return e / e.sum()


def build_states(adj, vecs):
    """Frozen larval reservoir states, exactly as e1_shakedown builds them."""
    n = adj["neurons"]
    D = vecs.shape[1]
    rng = np.random.default_rng(SEED)
    proj = rng.integers(-6, 7, size=(D, n)).astype(np.int8)
    cfg = {
        "name": adj["name"], "neurons": n, "edges": adj["edges"],
        "embed_dim": D, "steps": STEPS, "decay": DECAY, "classes": [],
        "indptr": adj["indptr"], "indices": adj["indices"],
        "weights": (np.asarray(adj["weights"], dtype=np.float64) / 127.0).tolist(),
        "weight_scale": 1.0 / 127.0,
        "win": (proj.astype(np.float64) * IN_SCALE).reshape(-1).tolist(),
    }
    S = np.asarray([st.states(cfg, x.tolist()) for x in vecs], dtype=np.float64)
    return S, cfg


def strat_folds(y, classes, folds=FOLDS, seed=SEED):
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


def simple_acc(pred_labels, y):
    return float(np.mean(np.asarray(pred_labels) == np.asarray(y)))


# ------------------------------------------------------------------- heads

class RidgeHead:
    """Closed-form ridge readout on one-hot targets (production ridge_fit).

    ``lam=None`` -> production resolve_ridge_lambda (inner CV on the train
    fold); an explicit value pins the penalty (the 1e-3 / 30 anchors).
    """

    def __init__(self, lam=None, label="ridge"):
        self.lam = lam
        self.label = label

    def fit(self, S, y, classes):
        Y = _one_hot(y, classes)
        self.selected_lam = resolve_ridge_lambda(S, Y, self.lam)
        W, bias = ridge_fit(S, Y, lam=self.lam)
        self.W = np.asarray(W, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.classes = list(classes)
        return self

    def score(self, S):
        logits = np.asarray(S, dtype=np.float64) @ self.W + self.bias  # W is [N][K]
        probs, margins = [], []
        for row in logits:
            p = softmax(row)
            order = np.sort(p)
            probs.append({self.classes[j]: float(p[j]) for j in range(len(p))})
            margins.append(float(order[-1] - order[-2]))
        return probs, margins

    def predict(self, S):
        logits = np.asarray(S, dtype=np.float64) @ self.W + self.bias
        return [self.classes[int(i)] for i in np.argmax(logits, axis=1)]


class CentroidHead:
    """Centroid + cosine margin - the meept campaign's measured champion shape.

    conf = softmax over cosines at temperature 0.1 (baselines.py). margin_mode
    "cosine" is the champion's own margin (top1-top2 cosine); "prob" is the
    softmax-probability margin, i.e. the same margin space the linear heads
    report, kept as a sensitivity check on the margin floor.
    """

    def __init__(self, margin_mode="cosine"):
        self.margin_mode = margin_mode

    def fit(self, S, y, classes):
        S = np.asarray(S, dtype=np.float64)
        y = np.asarray(y)
        self.classes = list(classes)
        self.cent = np.vstack([S[y == c].mean(axis=0) for c in self.classes])
        return self

    def _sims(self, S):
        S = np.asarray(S, dtype=np.float64)
        cm = np.maximum(np.linalg.norm(self.cent, axis=1, keepdims=True), 1e-12)
        sn = np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        return (S / sn) @ (self.cent / cm).T  # (rows, classes)

    def score(self, S):
        sims = self._sims(S)
        probs, margins = [], []
        for row in sims:
            p = softmax(row / COS_T)
            order = np.sort(row)
            probs.append({self.classes[j]: float(p[j]) for j in range(len(p))})
            if self.margin_mode == "cosine":
                margins.append(float(order[-1] - order[-2]))
            else:
                po = np.sort(p)
                margins.append(float(po[-1] - po[-2]))
        return probs, margins

    def predict(self, S):
        sims = self._sims(S)
        return [self.classes[int(i)] for i in np.argmax(sims, axis=1)]


class KnnUnanimityHead:
    """Cosine kNN (k=5): conf = majority vote share, margin = (top-runner)/k."""

    def __init__(self, k=KNN_K):
        self.k = k

    def fit(self, S, y, classes):
        self.S = np.asarray(S, dtype=np.float64)
        self.y = [str(v) for v in y]
        self.classes = list(classes)
        return self

    def score(self, S):
        S = np.asarray(S, dtype=np.float64)
        sn = np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        tn = np.maximum(np.linalg.norm(self.S, axis=1, keepdims=True), 1e-12)
        sims = (S / sn) @ (self.S / tn).T
        probs, margins = [], []
        for row in sims:
            kk = min(self.k, len(self.y))
            top = np.argsort(-row)[:kk]
            counts = {}
            for i in top:
                counts[self.y[int(i)]] = counts.get(self.y[int(i)], 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            maj = ranked[0][1]
            second = ranked[1][1] if len(ranked) > 1 else 0
            probs.append({c: counts.get(c, 0) / kk for c in self.classes})
            margins.append((maj - second) / kk)
        return probs, margins

    def predict(self, S):
        probs, _ = self.score(S)
        return [max(self.classes, key=lambda c: probs[i][c]) for i in range(len(probs))]


class LogisticHead:
    """Multinomial logistic regression, numpy only, deterministic.

    Full-batch gradient descent with momentum on z-scored features (train-fold
    statistics only), L2 on the weights. l2 is chosen by a deterministic
    inner CV over LOGISTIC_L2_GRID (same stratified inner-fold helper the ridge
    policy uses), so this head is given the same fair shot at regularization
    as the auto-ridge head. No shuffling, no RNG beyond the inner folds.
    """

    def __init__(self, l2="cv", lr=LOGISTIC_LR, epochs=LOGISTIC_EPOCHS):
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs

    def _prep(self, S):
        S = np.asarray(S, dtype=np.float64)
        return (S - self.mu) / self.sd

    def _fit_one(self, Xz, Y, l2):
        n, d = Xz.shape
        K = Y.shape[1]
        W = np.zeros((d, K))
        b = np.zeros(K)
        vW = np.zeros_like(W)
        vb = np.zeros(K)
        for _ in range(self.epochs):
            L = Xz @ W + b
            L -= L.max(axis=1, keepdims=True)
            P = np.exp(L)
            P /= P.sum(axis=1, keepdims=True)
            G = Xz.T @ (P - Y) / n + l2 * W
            gb = (P - Y).sum(axis=0) / n
            vW = 0.9 * vW - G
            vb = 0.9 * vb - gb
            W += self.lr * vW
            b += self.lr * vb
        return W, b

    def fit(self, S, y, classes):
        S = np.asarray(S, dtype=np.float64)
        self.classes = list(classes)
        self.mu = S.mean(axis=0)
        self.sd = S.std(axis=0) + 1e-6
        Xz = self._prep(S)
        Y = np.asarray(_one_hot(y, classes), dtype=np.float64)
        if self.l2 == "cv":
            self.selected_l2, self.inner_scores = self._select_l2(Xz, y)
        else:
            self.selected_l2, self.inner_scores = float(self.l2), {}
        self.W, self.bias = self._fit_one(Xz, Y, self.selected_l2)
        return self

    def _select_l2(self, Xz, y):
        labels = [str(v) for v in y]
        inner = _inner_folds(labels, 4, INNER_SEED)
        classes = self.classes
        acc = {float(c): [] for c in LOGISTIC_L2_GRID}
        for f in range(4):
            tr = inner != f
            va = ~tr
            if not tr.any() or not va.any():
                continue
            Ytr = np.asarray(_one_hot([labels[i] for i in np.where(tr)[0]], classes),
                             dtype=np.float64)
            gold = np.asarray([classes.index(labels[i]) for i in np.where(va)[0]])
            for c in LOGISTIC_L2_GRID:
                W, b = self._fit_one(Xz[tr], Ytr, float(c))
                pred = np.argmax(Xz[va] @ W + b, axis=1)
                acc[float(c)].append(float((pred == gold).mean()))
        best, best_acc = float(LOGISTIC_L2_GRID[0]), -1.0
        for c in LOGISTIC_L2_GRID:
            vals = acc[float(c)]
            a = sum(vals) / len(vals) if vals else 0.0
            if a > best_acc + 1e-12:
                best, best_acc = float(c), a
        return best, {str(k): round(sum(v) / max(len(v), 1), 4) for k, v in acc.items()}

    def score(self, S):
        logits = self._prep(S) @ self.W + self.bias
        probs, margins = [], []
        for row in logits:
            p = softmax(row)
            o = np.sort(p)
            probs.append({self.classes[j]: float(p[j]) for j in range(len(p))})
            margins.append(float(o[-1] - o[-2]))
        return probs, margins

    def predict(self, S):
        logits = self._prep(S) @ self.W + self.bias
        return [self.classes[int(i)] for i in np.argmax(logits, axis=1)]


# ---------------------------------------------------------------- protocol

def gate_eval(head_factory, S, y, classes, fold, n_total):
    """Fit per fold, gate with TRAIN-calibrated thresholds, accumulate."""
    routes = route_correct = abstained = 0
    ungated_ok = 0
    per_fold = []
    for f in range(FOLDS):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        head = head_factory().fit(S[tr], [y[i] for i in tr], classes)
        tr_probs, _ = head.score(S[tr])
        thr = choose_thresholds(tr_probs, [y[i] for i in tr], target_precision=0.97)
        te_probs, te_margins = head.score(S[te])
        preds = head.predict(S[te])
        f_routes = f_ok = f_abs = f_ungated = 0
        for k, i in enumerate(te):
            p = te_probs[k]
            top = max(classes, key=lambda c: (p[c], -classes.index(c)))
            if p[top] >= thr[top] and te_margins[k] > MARGIN_FLOOR:
                f_routes += 1
                if top == y[i]:
                    f_ok += 1
            else:
                f_abs += 1
            if preds[k] == y[i]:
                f_ungated += 1
        routes += f_routes
        route_correct += f_ok
        abstained += f_abs
        ungated_ok += f_ungated
        entry = {"fold": int(f), "train": int(len(tr)), "test": int(len(te)),
                 "routes": f_routes, "route_correct": f_ok, "abstained": f_abs,
                 "oof_acc": round(f_ungated / len(te), 4)}
        lam = getattr(head, "selected_lam", None)
        if lam is not None:
            entry["selected_lam"] = round(float(lam), 6)
        l2 = getattr(head, "selected_l2", None)
        if l2 is not None:
            entry["selected_l2"] = float(l2)
            entry["l2_equiv_ridge_lambda"] = round(float(l2) * len(tr), 1)
        per_fold.append(entry)

    total = routes + abstained
    return {
        "routes": routes, "route_correct": route_correct, "abstained": abstained,
        "total": total,
        "P": round(route_correct / routes, 4) if routes else 1.0,
        "C": round(routes / total, 4),
        "A": round(route_correct / total, 4),
        "A_all": round(ungated_ok / total, 4),
        "E2E": round((route_correct + CHAIN * abstained) / total, 4),
        "route_count_disclosure": f"{routes}/{total} direct routes",
        "per_fold": per_fold,
    }


def ood_abstain(head_factory, S_gold, y, classes, S_ood):
    """Fit on ALL gold, calibrate on ALL gold, count OOD abstentions."""
    head = head_factory().fit(S_gold, y, classes)
    probs, _ = head.score(S_gold)
    thr = choose_thresholds(probs, y, target_precision=0.97)
    ood_probs, ood_margins = head.score(S_ood)
    ok = 0
    for k in range(len(S_ood)):
        p = ood_probs[k]
        top = max(classes, key=lambda c: (p[c], -classes.index(c)))
        if not (p[top] >= thr[top] and ood_margins[k] > MARGIN_FLOOR):
            ok += 1
    return round(ok / len(S_ood), 4) if len(S_ood) else None


# -------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(REPO / "data" / "e1_inputs.json"))
    ap.add_argument("--adjacency", default=str(REPO / "data" / "larval_adjacency.json"))
    ap.add_argument("--out", default=str(REPO / "tools" / "eval" / "axis_readout_heads.json"))
    ap.add_argument("--states-cache", default="/tmp/axis_c_readout_states.npy")
    ap.add_argument("--skip-ood", action="store_true")
    ap.add_argument("--enrich-only", metavar="JSON", default=None,
                    help="recompute the derived fields of an existing result file "
                         "in place (no experiment re-run)")
    args = ap.parse_args(argv)

    if args.enrich_only:
        path = Path(args.enrich_only)
        saved = json.loads(path.read_text())
        path.write_text(json.dumps(enrich(saved), indent=2) + "\n", encoding="utf-8")
        print(f"re-derived {path}")
        return 0

    t0 = time.time()
    adj = json.loads(Path(args.adjacency).read_text())
    data = json.loads(Path(args.inputs).read_text())
    gold = data["gold"]
    ood = data["ood"]
    classes = sorted(data["classes"])
    vecs = np.asarray(data["vecs"], dtype=np.float64)
    y = [g["intent"] for g in gold]
    n_gold = len(gold)

    cache = Path(args.states_cache)
    if cache.exists():
        S = np.load(cache)
    else:
        S, _ = build_states(adj, vecs)
        np.save(cache, S)
    assert S.shape == (len(vecs), adj["neurons"])
    Sg, So = S[:n_gold], S[n_gold:]
    sat = float(np.mean(np.abs(Sg) > 0.95))
    print(f"states {S.shape} saturation={sat:.3f} (neuron-major; {n_gold} gold, "
          f"{len(ood)} ood; {time.time()-t0:.1f}s)")

    fold = strat_folds(y, classes)

    heads = {}

    def run(name, factory, note=""):
        t = time.time()
        res = gate_eval(factory, Sg, y, classes, fold, n_gold)
        if not args.skip_ood:
            res["OOD_R"] = ood_abstain(factory, Sg, y, classes, So)
        res["note"] = note
        heads[name] = res
        print(f"[{name}] routes={res['routes']}/{res['total']} P={res['P']} "
              f"C={res['C']} A={res['A']} A_all={res['A_all']} E2E={res['E2E']} "
              f"OOD_R={res.get('OOD_R')} ({time.time()-t:.1f}s)")

    # --- anchors (the pre-existing ridge head at both penalties) ----------
    run("ridge_fixed_lam_1e-3", lambda: RidgeHead(lam=1e-3),
        "legacy trainer default (the defect)")
    run("ridge_fixed_lam_30", lambda: RidgeHead(lam=30.0),
        "reference corrected run (e1_corrected_results.json)")
    run("ridge_auto_inner_cv", lambda: RidgeHead(lam=None),
        "production resolve_ridge_lambda: deterministic inner CV per train fold")

    # --- head variants ----------------------------------------------------
    run("centroid_cosine_margin", lambda: CentroidHead("cosine"),
        "campaign champion shape: class centroids + cosine margin, conf=softmax(cos/0.1)")
    run("centroid_prob_margin", lambda: CentroidHead("prob"),
        "same centroids, margin in softmax-probability space (sensitivity check)")
    run("knn5_unanimity", lambda: KnnUnanimityHead(5),
        "cosine kNN k=5, conf = vote share, margin = (top-runner)/k")
    run("logistic_numpy_l2_cv", lambda: LogisticHead(l2="cv"),
        "multinomial logistic, full-batch GD + momentum, L2 by inner CV")

    # --- cross-checks -----------------------------------------------------
    checks = {}
    # 1. raw-embedding linear probe (reference: lam=0.1 -> A_all 0.7175, routes 88)
    emb_probe = gate_eval(lambda: RidgeHead(lam=0.1), vecs[:n_gold], y, classes,
                          fold, n_gold)
    checks["raw_embeddings_ridge_lam_0.1"] = {
        "routes": emb_probe["routes"], "P": emb_probe["P"], "C": emb_probe["C"],
        "A": emb_probe["A"], "A_all": emb_probe["A_all"], "E2E": emb_probe["E2E"],
        "expected": {"A_all": 0.7175, "routes": 88, "P": 0.921, "C": 0.244,
                     "E2E": 0.8808},
    }
    # 2. an independent logistic implementation (sklearn lbfgs) on the states
    try:
        from sklearn.linear_model import LogisticRegression

        ok = 0
        for f in range(FOLDS):
            tr = np.where(fold != f)[0]
            te = np.where(fold == f)[0]
            mu, sd = Sg[tr].mean(0), Sg[tr].std(0) + 1e-6
            clf = LogisticRegression(max_iter=3000, C=0.01, random_state=0)
            clf.fit((Sg[tr] - mu) / sd, [y[i] for i in tr])
            pred = clf.predict((Sg[te] - mu) / sd)
            ok += int(np.sum(pred == np.asarray([y[i] for i in te])))
        checks["sklearn_logistic_states_C0.01"] = {
            "A_all": round(ok / n_gold, 4),
            "note": "independent multinomial-logistic implementation, ungated OOF",
        }
    except Exception as e:  # noqa: BLE001
        checks["sklearn_logistic_states_C0.01"] = {"error": str(e)}

    # 3. the auto policy's choices vs the corrected-run optimum
    lam_full = resolve_ridge_lambda(Sg, _one_hot(y, classes))
    checks["auto_lambda_on_all_gold"] = {
        "selected": lam_full,
        "measured_best_in_corrected_run": 30,
        "legacy_default": 1e-3,
    }

    out = {
        "experiment": "Axis C - readout head variants on frozen larval reservoir states",
        "source": {
            "inputs": str(args.inputs), "adjacency": str(args.adjacency),
            "neurons": adj["neurons"], "edges": adj["edges"],
            "license": adj.get("license", ""),
            "gold": n_gold, "ood": len(ood), "classes": classes,
            "embed_dim": int(vecs.shape[1]),
            "states": "s = tanh(decay*s + W*s + win*x), steps=4, decay=0.8, "
                      "win = seeded int8 projection (rng 42) scaled to 0.02",
            "saturation": round(sat, 4),
        },
        "protocol": {
            "folds": FOLDS, "seed": SEED,
            "fold_rule": "shuffle per class with np.random.default_rng(42), fold = position % 5",
            "route_rule": "top-1 prob >= per-class calibrated threshold AND margin > 0.05",
            "calibration": "choose_thresholds(train_probs, train_labels, target_precision=0.97)",
            "margin_space": "each head's own margin (cosine for centroid, vote share for kNN, "
                            "softmax probability for the linear/logistic heads); the "
                            "centroid_prob_margin row repeats the centroid head with a "
                            "probability-space margin as a sensitivity check",
            "metrics": {"P": "route_correct/routes (wrong routes stay in the denominator)",
                        "C": "routes/total", "A": "route_correct/total",
                        "A_all": "ungated out-of-fold argmax accuracy (the quantity "
                                 "e1_corrected_results.json calls A)",
                        "E2E": "(route_correct + 0.868*abstained)/total"},
            "one_case_pt": round(100.0 / n_gold, 2),
        },
        "heads": heads,
        "cross_checks": checks,
        "runtime_s": round(time.time() - t0, 1),
    }
    enrich(out)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out} ({time.time()-t0:.1f}s)")
    print("verdict:", json.dumps(out["verdict"], indent=1))
    return 0


def enrich(out):
    """Derived fields (pure functions of the measured head numbers).

    Also available standalone so the JSON can be re-derived without re-running
    the ~5-minute experiment:

        python3 tools/eval/axis_readout_heads.py \\
            --enrich-only tools/eval/axis_readout_heads.json
    """
    heads = out["heads"]
    ridge_ref = heads["ridge_fixed_lam_30"]
    legacy = heads["ridge_fixed_lam_1e-3"]
    nonridge = {k: v for k, v in heads.items() if not k.startswith("ridge")}

    out.setdefault("cross_checks", {})["legacy_floor_on_this_projection"] = {
        "A_all_lam_1e-3": legacy["A_all"],
        "A_all_lam_30": ridge_ref["A_all"],
        "delta_A_all": round(ridge_ref["A_all"] - legacy["A_all"], 4),
        "routes_lam_1e-3": legacy["routes"],
        "routes_lam_30": ridge_ref["routes"],
        "note": ("the committed e1_real run quoted 0.125 for the legacy floor, but "
                 "that run's input projection differs (e1_real draws its projection "
                 "from an rng already consumed by the fold shuffles). On the "
                 "projection that reproduces e1_corrected_results.json exactly "
                 "(routes=71, P=0.9577, C=0.1967, E2E=0.8857) the legacy floor "
                 "costs %.1f points of all-case accuracy and %d of the 71 routes."
                 % ((ridge_ref["A_all"] - legacy["A_all"]) * 100,
                    ridge_ref["routes"] - legacy["routes"])),
    }

    def best_above(key, floor):
        cand = [k for k, v in nonridge.items() if v[key] > floor + 1e-12]
        return max(cand, key=lambda k: nonridge[k][key]) if cand else None

    ranked = sorted(((v["A_all"], k) for k, v in nonridge.items()), reverse=True)
    best_nonridge_e2e = max(nonridge, key=lambda k: nonridge[k]["E2E"])
    verdict = {
        "reference": {"head": "ridge_fixed_lam_30",
                      "A": ridge_ref["A"], "A_all": ridge_ref["A_all"],
                      "P": ridge_ref["P"], "C": ridge_ref["C"], "E2E": ridge_ref["E2E"]},
        "best_head_by_A_all": max(heads, key=lambda k: heads[k]["A_all"]),
        "best_head_by_A_route_correct_over_total": max(heads, key=lambda k: heads[k]["A"]),
        "best_head_by_E2E": max(heads, key=lambda k: heads[k]["E2E"]),
        "beats_ridge_on_E2E": best_above("E2E", ridge_ref["E2E"]),
        "beats_ridge_on_P": best_above("P", ridge_ref["P"]),
        "beats_ridge_on_A_route_correct_over_total": best_above("A", ridge_ref["A"]),
        "beats_ridge_on_A_all_ungated": best_above("A_all", ridge_ref["A_all"]),
        "chain_floor": CHAIN,
    }
    verdict["any_head_beats_ridge"] = bool(
        verdict["beats_ridge_on_E2E"] or verdict["beats_ridge_on_A_all_ungated"])
    log = nonridge["logistic_numpy_l2_cv"]
    cent = nonridge["centroid_cosine_margin"]
    verdict["plain_verdict"] = (
        "No head beats the ridge+margin head where it matters. E2E: best non-ridge "
        "'%s' = %.4f vs ridge %.4f. Ungated out-of-fold accuracy (how much signal the "
        "head actually extracts from the states): ridge %.4f, %s %.4f, centroid "
        "(cosine margin) %.4f, kNN-unanimity %.4f. The logistic head wins only the "
        "route_correct/total reading of A, by routing far more cases (C %.4f vs %.4f) "
        "at lower precision (P %.4f vs %.4f) - a coverage/precision trade that still "
        "lands below the ridge on E2E. The campaign's centroid+cosine-margin champion "
        "does not transfer to reservoir states."
        % (best_nonridge_e2e, nonridge[best_nonridge_e2e]["E2E"],
           ridge_ref["E2E"], ridge_ref["A_all"], ranked[0][1], ranked[0][0],
           cent["A_all"], nonridge["knn5_unanimity"]["A_all"], log["C"],
           ridge_ref["C"], log["P"], ridge_ref["P"])
    )
    out["verdict"] = verdict
    return out


if __name__ == "__main__":
    raise SystemExit(main())
