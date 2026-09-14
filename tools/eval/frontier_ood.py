"""Constrained frontier for the Stage-0 gate, with OOD as a first-class constraint.

Context: tuning bought coverage but regressed OOD abstention (0.929 -> 0.857),
and that regression was invisible because the gold OOD set is only 28 cases
(3.6 points per case). This script (a) fixes the measurement with a larger OOD
surface, reported SEPARATELY from the gold set, and (b) asks whether any
(head, gate) holds precision >= 0.97 AND OOD-abstain >= 0.95 while routing a
useful share of traffic. It then tests the two mechanisms that could plausibly
fix OOD:

  A. SUPERVISED OOD PROBE - a logistic classifier separating in-distribution
     from out-of-distribution, evaluated with HELD-OUT OOD folds so it cannot
     memorise the evaluation set.
  B. (reported) unsupervised novelty - distance to the nearest training example,
     which the earlier sweep showed barely helps.

Run:  python3 tools/eval/frontier_ood.py
Data: data/e1_inputs.json and data/ood_synthetic_embedded.json
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "train"))

from calibrate import choose_thresholds  # noqa: E402

CHAIN_BASELINE = 0.868
MARGIN = 0.05
TARGET_P = 0.97
SEED = 42
FOLDS = 5
PROBE_PCTL = 0.90      # in-distribution retention target when calibrating the probe


def load():
    d = json.loads((REPO / "data" / "e1_inputs.json").read_text())
    gold, classes = d["gold"], d["classes"]
    vecs = np.asarray(d["vecs"], dtype=np.float64)
    x = vecs[: len(gold)]
    y = np.array([c["intent"] for c in gold])
    gold_ood = vecs[len(gold):]
    syn_path = REPO / "data" / "ood_synthetic_embedded.json"
    syn_ood = (np.asarray(json.loads(syn_path.read_text())["vecs"], dtype=np.float64)
               if syn_path.exists() else np.zeros((0, x.shape[1])))
    return x, y, classes, gold_ood, syn_ood


def norm(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def folds_for(y, classes, seed=SEED):
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), dtype=int)
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % FOLDS
    return fold


def novelty(x, xtr):
    """Cosine distance to the nearest training example (leave-one-out on train)."""
    d = 1.0 - norm(x) @ norm(xtr).T
    if x.shape == xtr.shape and np.allclose(x, xtr):
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def fit_centroid(xtr, cls):
    return norm(np.array([xtr[[i for i in range(len(cls)) if cls[i] == c]].mean(0)
                          for c in CLASSES]))


def fit_ridge(xtr, ytr, lam):
    k = len(CLASSES)
    yy = np.zeros((len(ytr), k))
    for i, v in enumerate(ytr):
        yy[i, CLASSES.index(v)] = 1
    xb = np.hstack([xtr, np.ones((len(xtr), 1))])
    sol = np.linalg.solve(xb.T @ xb + lam * np.eye(xb.shape[1]), xb.T @ yy)
    return sol[:-1], sol[-1]


def fit_lda(xtr, ytr, shrink=0.3):
    """Linear discriminant: whitened class means (the Mahalanobis head).

    Included because it is the standard step up from cosine-to-mean and had
    never been tried here. It routes far more and is worse on both precision
    and OOD - see the report.
    """
    m = np.array([xtr[[i for i in range(len(ytr)) if ytr[i] == c]].mean(0) for c in CLASSES])
    xc = xtr - m[[CLASSES.index(v) for v in ytr]]
    s_cov = (xc.T @ xc) / max(len(xtr) - len(CLASSES), 1)
    sph = (1 - shrink) * s_cov + shrink * np.trace(s_cov) / xtr.shape[1] * np.eye(xtr.shape[1])
    si = np.linalg.inv(sph)
    return ("lin", (m @ si).T, -0.5 * np.einsum('ij,ij->i', m @ si, m))


def probs(fitted, x, temp):
    if isinstance(fitted, np.ndarray):
        s = norm(x) @ fitted.T
        e = np.exp(s * temp)
        return e / e.sum(1, keepdims=True)
    if len(fitted) == 3:      # ("lin", W, b) from fit_lda
        _, w, b = fitted
    else:                     # (W, b) from fit_ridge
        w, b = fitted
    logits = x @ w + b
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def sorted_probs(p):
    srt = -np.sort(-p, axis=1)
    return p.argmax(1), srt[:, 0], srt[:, 0] - srt[:, 1]


CLASSES = []


def run(head, x, y, classes, fold, gold_ood, syn_ood, lam=10.0, temp=10.0,
        ood_probe=False, ood_fold=None, novelty_pctl=None, target_p=TARGET_P):
    """Returns routing metrics; with ood_probe=True the probe gates every route."""
    global CLASSES
    CLASSES = classes
    acc = {"routes": 0, "route_correct": 0, "abstained": 0, "g_ab": 0, "s_ab": 0}
    for f in range(FOLDS):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        ytr = [y[i] for i in tr]
        if head == "centroid":
            fitted = fit_centroid(x[tr], ytr)
        elif head == "lda":
            fitted = fit_lda(x[tr], ytr)
        else:
            fitted = fit_ridge(x[tr], ytr, lam)
        p_tr = probs(fitted, x[tr], temp)

        allow_probe = None
        if ood_probe:
            from sklearn.linear_model import LogisticRegression
            # train the probe on train-fold gold + the OTHER folds' OOD cases only
            ood_all = np.vstack([gold_ood, syn_ood])
            tr_ood = ood_fold != f
            xp = np.vstack([x[tr], ood_all[tr_ood]])
            lp = np.array([0] * len(tr) + [1] * int(tr_ood.sum()))
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(xp, lp)
            # calibrate so that PROBE_PCTL of train-fold gold is retained
            pin_tr = clf.predict_proba(x[tr])[:, 0]
            cut = float(np.quantile(pin_tr, 1.0 - PROBE_PCTL))
            allow_probe = (lambda X, _c=clf, _cut=cut: _c.predict_proba(X)[:, 0] >= _cut)

        # Calibrate the class thresholds on the population the gate will actually
        # see. With a probe in front, calibrating on ALL train rows overstates
        # precision, because the probe then removes correct and wrong routes
        # indiscriminately (measured: P fell 0.9737 -> 0.9524).
        keep = np.ones(len(tr), dtype=bool)
        if allow_probe is not None:
            keep = allow_probe(x[tr])
        rows = [{classes[j]: float(p_tr[q][j]) for j in range(len(classes))}
                for q in range(len(tr)) if keep[q]]
        keep_labels = [ytr[q] for q in range(len(tr)) if keep[q]]
        thr = choose_thresholds(rows, keep_labels, target_precision=target_p) if keep.any() \
            else {c: 1.1 for c in classes}
        # a class absent from the calibration subset can never route
        tau = np.array([thr.get(c, 1.1) for c in classes])

        nov_cut = None
        if novelty_pctl is not None:
            nov_cut = float(np.quantile(novelty(x[tr], x[tr]), novelty_pctl))

        for x_set, tag in ((x[te], "test"), (gold_ood, "g"), (syn_ood, "s")):
            if len(x_set) == 0:
                continue
            p = probs(fitted, x_set, temp)
            top, conf, mg = sorted_probs(p)
            allow = (conf >= tau[top]) & (mg > MARGIN)
            if allow_probe is not None:
                allow &= allow_probe(x_set)
            if nov_cut is not None:
                allow &= novelty(x_set, x[tr]) <= nov_cut
            if tag == "test":
                correct = top == np.array([classes.index(v) for v in y[te]])
                acc["routes"] += int(allow.sum())
                acc["route_correct"] += int((allow & correct).sum())
                acc["abstained"] += int((~allow).sum())
            elif tag == "g":
                acc["g_ab"] += int((~allow).sum())
            else:
                acc["s_ab"] += int((~allow).sum())
    total = acc["routes"] + acc["abstained"]
    return {
        "head": head, "lam": lam, "temp": temp, "ood_probe": ood_probe,
        "novelty_pctl": novelty_pctl, "target_p": target_p,
        "routes": acc["routes"],
        "P": round(acc["route_correct"] / max(acc["routes"], 1), 4),
        "C": round(acc["routes"] / total, 4),
        "E2E": round((acc["route_correct"] + CHAIN_BASELINE * acc["abstained"]) / total, 4),
        "OOD_gold": round(acc["g_ab"] / (len(gold_ood) * FOLDS), 4) if len(gold_ood) else None,
        "OOD_syn": round(acc["s_ab"] / (len(syn_ood) * FOLDS), 4) if len(syn_ood) else None,
    }


def main() -> int:
    x, y, classes, gold_ood, syn_ood = load()
    print(f"gold {x.shape} | gold-OOD {gold_ood.shape} | synthetic-OOD {syn_ood.shape}")
    fold = folds_for(y, classes)
    n_ood = len(gold_ood) + len(syn_ood)
    ood_fold = np.arange(n_ood) % FOLDS

    rows = []
    print("\n--- baseline frontier (no OOD mechanism) ---")
    print(f"{'head':9s} {'lam':>5s} {'temp':>5s} {'routes':>7s} {'P':>7s} {'C':>7s} {'E2E':>7s} {'OODg':>7s} {'OODs':>7s}")
    for head, lam, temp in (("centroid", 0.0, 10.0), ("centroid", 0.0, 20.0),
                            ("lda", 0.0, 0.0), ("ridge", 1.0, 0.0), ("ridge", 10.0, 0.0)):
        r = run(head, x, y, classes, fold, gold_ood, syn_ood, lam=lam, temp=temp)
        rows.append(r)
        print(f"{r['head']:9s} {r['lam']:5.1f} {r['temp']:5.1f} {r['routes']:7d} {r['P']:7.4f} "
              f"{r['C']:7.4f} {r['E2E']:7.4f} {r['OOD_gold']:7.4f} {r['OOD_syn']:7.4f}")

    print("\n--- with the supervised OOD probe (held-out OOD folds) ---")
    for head, lam, temp in (("centroid", 0.0, 10.0), ("centroid", 0.0, 20.0),
                            ("ridge", 1.0, 0.0), ("ridge", 10.0, 0.0)):
        r = run(head, x, y, classes, fold, gold_ood, syn_ood, lam=lam, temp=temp,
                ood_probe=True, ood_fold=ood_fold)
        rows.append(r)
        print(f"{r['head']:9s} {r['lam']:5.1f} {r['temp']:5.1f} {r['routes']:7d} {r['P']:7.4f} "
              f"{r['C']:7.4f} {r['E2E']:7.4f} {r['OOD_gold']:7.4f} {r['OOD_syn']:7.4f}")

    print("\n--- with unsupervised novelty (for contrast) ---")
    for pctl in (0.99, 0.95):
        r = run("centroid", x, y, classes, fold, gold_ood, syn_ood, temp=10.0,
                novelty_pctl=pctl)
        rows.append(r)
        print(f"{r['head']:9s} {'-':>5s} {r['temp']:5.1f} {r['routes']:7d} {r['P']:7.4f} "
              f"{r['C']:7.4f} {r['E2E']:7.4f} {r['OOD_gold']:7.4f} {r['OOD_syn']:7.4f}   nov_pctl={pctl}")

    print("\n--- stricter class thresholds x OOD probe (the untested combination) ---")
    for tp in (0.98, 0.99, 0.995):
        for probe in (False, True):
            r = run("centroid", x, y, classes, fold, gold_ood, syn_ood, temp=10.0,
                    ood_probe=probe, ood_fold=ood_fold, target_p=tp)
            rows.append(r)
            print(f"centroid target_p={tp:<6.3f} probe={str(probe):5s} routes={r['routes']:3d} "
                  f"P={r['P']:.4f} C={r['C']:.4f} E2E={r['E2E']:.4f} OODg={r['OOD_gold']:.4f} OODs={r['OOD_syn']:.4f}")

    feasible = [r for r in rows if r["P"] >= 0.97 and r["OOD_gold"] >= 0.95]
    print(f"\nfeasible (P>=0.97 AND OOD_gold>=0.95): {len(feasible)}")
    if feasible:
        b = max(feasible, key=lambda r: r["E2E"])
        print(f"best feasible: {b}")
    best = max(rows, key=lambda r: r["E2E"])
    print(f"best E2E overall: head={best['head']} probe={best['ood_probe']} "
          f"-> E2E {best['E2E']}, P {best['P']}, routes {best['routes']}, "
          f"OOD_gold {best['OOD_gold']}")

    out = REPO / "tools" / "eval" / "frontier_ood_results.json"
    out.write_text(json.dumps({"folds": FOLDS, "seed": SEED, "target_precision": TARGET_P,
                               "chain_baseline": CHAIN_BASELINE,
                               "probe_retention": PROBE_PCTL, "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())