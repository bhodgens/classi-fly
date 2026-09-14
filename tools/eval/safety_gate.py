"""Safety-first routing gate (owner decision 'OOD policy: safety').

Trades coverage for safety: the gate routes a case only when ALL of
  (a) the centroid/cosine head's top-1 probability clears that class's
      calibrated threshold (target precision 0.97),
  (b) the top1-top2 probability margin exceeds 0.05, and
  (c) a supervised OOD probe (logistic, in-dist vs OOD) calls the case
      in-distribution with probability >= its calibrated threshold
      (calibrated so 95% of train in-dist rows are retained).
Any miss -> abstain (fail-safe: the chain baseline handles the case).

CRITICAL calibration detail (measured in the frontier run): the class
thresholds must be calibrated on the subset of train rows that PASS the OOD
probe, not on all train rows. Calibrating on all rows overstates precision
(measured P fell 0.9737 -> 0.9524) because the probe removes correct and
wrong routes indiscriminately afterwards.

Fail-safe property: under garbage (pure-noise) input the gate must ABSTAIN,
not route - the cosine-margin centroid head family fails safe; the
softmax-margin head does not (it routed 77 noise cases at P 0.22, sigma 0.2).

Run:  python3 tools/eval/safety_gate.py
Writes: tools/eval/safety_gate_results.json
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
PROBE_RETENTION = 0.95   # in-dist rows the OOD probe must retain
SEED = 42
FOLDS = 5
TEMP = 10.0


def norm(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def folds_for(y, classes, seed=SEED):
    """Stratified fold assignment (same protocol as frontier_ood.py)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), dtype=int)
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % FOLDS
    return fold


class SafetyGate:
    """Centroid + cosine-margin head behind a supervised OOD probe.

    fit(x_gold, y_gold, x_ood_train): learn class means, fit the logistic
    OOD probe, calibrate the probe threshold at PROBE_RETENTION on the
    in-dist train rows, then calibrate per-class route thresholds with
    choose_thresholds on the subset of train rows that PASS the probe.
    """

    def __init__(self, classes=None, target_precision=TARGET_P, temp=TEMP,
                 margin=MARGIN, probe_retention=PROBE_RETENTION,
                 precise_thresholds=False):
        if classes is None:
            d = json.loads((REPO / "data" / "e1_inputs.json").read_text())
            classes = d["classes"]
        self.classes = list(classes)
        self.target_precision = target_precision
        self.precise_thresholds = precise_thresholds
        self.temp = temp
        self.margin = margin
        self.probe_retention = probe_retention
        self.centroids = None
        self.ood_clf = None
        self.ood_threshold = None
        self.thresholds = {}
        self.calibrated_on = 0

    # -- head ---------------------------------------------------------------
    def _probs(self, x):
        s = norm(x) @ self.centroids.T * self.temp
        s = s - s.max(1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(1, keepdims=True)

    @staticmethod
    def _sorted_probs(p):
        srt = -np.sort(-p, axis=1)
        return p.argmax(1), srt[:, 0], srt[:, 0] - srt[:, 1]

    # -- OOD probe ----------------------------------------------------------
    def _ood_in_prob(self, x):
        return self.ood_clf.predict_proba(x)[:, 0]

    @staticmethod
    def _lane_prec(rows, labels, cls, t):
        """'score_c >= t routes to cls' precision (no FP credit)."""
        tp = fp = 0
        for row, lab in zip(rows, labels):
            crossed = row.get(cls, 0.0) >= t
            if lab == cls:
                if crossed:
                    tp += 1
            elif crossed:
                fp += 1
        return 1.0 if tp + fp == 0 else tp / (tp + fp)

    def _choose_thresholds(self, rows, labels):
        """Per-class thresholds on the calibration rows.

        precise_thresholds=True uses the exact per-class score as the sole
        candidate (zero tolerance), so a threshold calibrated at P>=0.97
        cannot be met by a test row scoring BELOW every calibrated class
        score - the quantile grid can't guarantee that.
        """
        from calibrate import choose_thresholds
        if self.precise_thresholds:
            classes_present = sorted(set(labels))
            out = {}
            for cls in classes_present:
                own = sorted(r.get(cls, 0.0) for r, lab in zip(rows, labels) if lab == cls)
                for t in own:
                    if self._lane_prec(rows, labels, cls, t) >= self.target_precision:
                        out[cls] = t
                        break
                else:
                    out[cls] = own[-1]
            return out
        return choose_thresholds(rows, labels, target_precision=self.target_precision)

    # -- API ----------------------------------------------------------------
    def fit(self, x_gold, y_gold, x_ood_train):
        from sklearn.linear_model import LogisticRegression

        classes = self.classes
        y_idx = np.array([classes.index(v) for v in y_gold])
        # centroid head: normalised class means, cosine margin
        self.centroids = norm(np.vstack(
            [x_gold[y_idx == k].mean(0) for k in range(len(classes))]))
        # supervised OOD probe
        xp = np.vstack([x_gold, x_ood_train])
        lp = np.array([0] * len(x_gold) + [1] * len(x_ood_train))
        self.ood_clf = LogisticRegression(max_iter=2000, C=1.0).fit(xp, lp)
        pin = self._ood_in_prob(x_gold)
        # retain PROBE_RETENTION of in-dist train rows
        self.ood_threshold = float(np.quantile(pin, 1.0 - self.probe_retention))
        # CRITICAL: calibrate class thresholds on the probe-PASSING subset only
        p = self._probs(x_gold)
        top, conf, _ = self._sorted_probs(p)
        keep = pin >= self.ood_threshold
        rows = [{classes[j]: float(p[i][j]) for j in range(len(classes))}
                for i in np.where(keep)[0]]
        labels = [y_gold[i] for i in np.where(keep)[0]]
        self.calibrated_on = int(keep.sum())
        thr = self._choose_thresholds(rows, labels) if keep.any() else {}
        # a class absent from the calibration subset can never route
        self.thresholds = {c: thr.get(c, 1.1) for c in classes}
        return self

    def predict(self, x):
        """Returns a list of dicts: class, confidence, margin, ood_score, routed."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        p = self._probs(x)
        top, conf, mg = self._sorted_probs(p)
        ood_in = self._ood_in_prob(x)
        tau = np.array([self.thresholds[c] for c in self.classes])
        routed = (conf >= tau[top]) & (mg > self.margin) & (ood_in >= self.ood_threshold)
        return [{"class": self.classes[top[i]], "confidence": float(conf[i]),
                 "margin": float(mg[i]), "ood_score": float(ood_in[i]),
                 "routed": bool(routed[i])} for i in range(len(x))]


def load():
    d = json.loads((REPO / "data" / "e1_inputs.json").read_text())
    gold, classes = d["gold"], d["classes"]
    vecs = np.asarray(d["vecs"], dtype=np.float64)
    x = vecs[: len(gold)]
    y = [c["intent"] for c in gold]
    gold_ood = vecs[len(gold):]
    syn_path = REPO / "data" / "ood_synthetic_embedded.json"
    syn_ood = (np.asarray(json.loads(syn_path.read_text())["vecs"], dtype=np.float64)
               if syn_path.exists() else np.zeros((0, x.shape[1])))
    real_path = REPO / "data" / "ood_real_embedded.json"
    real_ood = (np.asarray(json.loads(real_path.read_text())["vecs"], dtype=np.float64)
                if real_path.exists() else None)
    return x, y, classes, gold_ood, syn_ood, real_ood


def noise_fail_safe(classes, dim=1024, n=200, sigma=0.2, seed=SEED):
    """Pure-noise embeddings must ABSTAIN for every case, any sigma."""
    rng = np.random.default_rng(seed)
    x, y, _, gold_ood, syn_ood, _ = load()
    gate = SafetyGate(classes).fit(x, y, np.vstack([gold_ood, syn_ood]))
    for sg in (sigma, 0.5, 1.0):
        out = gate.predict(rng.normal(0.0, sg, (n, dim)))
        if any(r["routed"] for r in out):
            return False, f"routed noise at sigma={sg}"
    return True, "abstained at all noise levels"


def evaluate(x, y, classes, fold, gold_ood, syn_ood, ood_sources=None,
             gate_cls=SafetyGate, target_p=TARGET_P):
    """5-fold protocol: P/C/E2E with routed-but-wrong in the P denominator."""
    acc = {"routes": 0, "route_correct": 0, "abstained": 0}
    y = np.asarray(y)
    ood_ab = {name: 0 for name in (ood_sources or {})}
    ood_tot = {name: 0 for name in (ood_sources or {})}
    for f in range(FOLDS):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        # NOTE: the probe sees only the OTHER folds' OOD cases, so it cannot
        # memorise the OOD cases it is evaluated on.
        n_ood = len(gold_ood) + len(syn_ood)
        ood_fold = np.arange(n_ood) % FOLDS
        ood_train = np.vstack([gold_ood, syn_ood])[ood_fold != f]
        gate = gate_cls(classes, target_precision=target_p).fit(x[tr], [y[i] for i in tr], ood_train)
        res = gate.predict(x[te])
        for r in res:
            acc["routes" if r["routed"] else "abstained"] += 1
        correct = np.array([r["routed"] and r["class"] == v
                            for r, v in zip(res, y[te])])
        acc["route_correct"] += int(correct.sum())
        for name, x_set in (ood_sources or {}).items():
            if x_set is None or len(x_set) == 0:
                continue
            rr = gate.predict(x_set)
            ood_ab[name] += sum(1 for r in rr if not r["routed"])
            ood_tot[name] += len(rr)
    total = acc["routes"] + acc["abstained"]
    return {
        "routes": acc["routes"],
        "P": round(acc["route_correct"] / max(acc["routes"], 1), 4),
        "C": round(acc["routes"] / max(total, 1), 4),
        "E2E": round((acc["route_correct"] + CHAIN_BASELINE * acc["abstained"]) / max(total, 1), 4),
        "ood_abstain": {k: round(v / max(ood_tot[k], 1), 4) for k, v in ood_ab.items()},
    }


class BadCalGate(SafetyGate):
    """Ablation: same gate but thresholds calibrated on ALL train rows.

    This is the exact mistake the last run made: it overstates precision
    (measured P fell 0.9737 -> 0.9524 in the frontier sweep) because the
    probe then removes correct and wrong routes indiscriminately.
    Uses precise (zero-tolerance) thresholds so the calibration-subset
    difference is not masked by the quantile grid's tolerance.
    """

    def __init__(self, classes, **kw):
        kw["precise_thresholds"] = True
        super().__init__(classes, **kw)

    def fit(self, x_gold, y_gold, x_ood_train):
        from sklearn.linear_model import LogisticRegression
        cls = self.classes
        y_idx = np.array([cls.index(v) for v in y_gold])
        self.centroids = norm(np.vstack(
            [x_gold[y_idx == k].mean(0) for k in range(len(cls))]))
        xp = np.vstack([x_gold, x_ood_train])
        lp = np.array([0] * len(x_gold) + [1] * len(x_ood_train))
        self.ood_clf = LogisticRegression(max_iter=2000, C=1.0).fit(xp, lp)
        pin = self._ood_in_prob(x_gold)
        self.ood_threshold = float(np.quantile(pin, 1.0 - self.probe_retention))
        p = self._probs(x_gold)
        rows_all = [{cls[j]: float(p[i][j]) for j in range(len(cls))}
                    for i in range(len(x_gold))]
        thr = self._choose_thresholds(rows_all, list(y_gold))
        self.thresholds = {c: thr.get(c, 1.1) for c in cls}
        self.calibrated_on = len(x_gold)
        return self


def main() -> int:
    x, y, classes, gold_ood, syn_ood, real_ood = load()
    print(f"gold {x.shape} | gold-OOD {gold_ood.shape} | synthetic-OOD {syn_ood.shape}"
          + (f" | real-OOD {real_ood.shape}" if real_ood is not None
             else " | real-OOD PENDING (data/ood_real_embedded.json absent)"))
    fold = folds_for(y, classes)
    ood_sources = {"gold": gold_ood, "synthetic": syn_ood}
    if real_ood is not None:
        ood_sources["real"] = real_ood

    rows = []
    for target_p in (0.97, 0.95):
        r = evaluate(x, y, classes, fold, gold_ood, syn_ood,
                     ood_sources=ood_sources, target_p=target_p)
        r["target_precision"] = target_p
        rows.append(r)
        print(f"target_p={target_p}: routes={r['routes']} P={r['P']} C={r['C']} "
              f"E2E={r['E2E']} ood={r['ood_abstain']}")

    # noise fail-safe
    ok, msg = noise_fail_safe(classes)
    print(f"noise fail-safe: {ok} ({msg})")

    # ablation: calibrating on ALL train rows instead of the probe-passing subset
    r_bad = evaluate(x, y, classes, fold, gold_ood, syn_ood,
                     ood_sources=ood_sources, gate_cls=BadCalGate, target_p=0.97)
    r_bad["target_precision"] = 0.97
    r_bad["variant"] = "calibrate_on_all_train_rows"
    rows.append(r_bad)
    print(f"ablation (calibrate on ALL rows): P={r_bad['P']} C={r_bad['C']}")

    out = REPO / "tools" / "eval" / "safety_gate_results.json"
    out.write_text(json.dumps({
        "gate": "safety-first (centroid cosine margin + supervised OOD probe)",
        "folds": FOLDS, "seed": SEED, "target_precision": TARGET_P,
        "chain_baseline": CHAIN_BASELINE, "probe_retention": PROBE_RETENTION,
        "ood_real_pending": real_ood is None, "rows": rows,
        "noise_fail_safe": {"passed": ok, "detail": msg},
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
