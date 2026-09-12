"""Linear readout training on frozen reservoir states (leaf 04 Tasks 2 and 4).

Preferred solver: closed-form ridge regression on one-hot targets,

    W = (S^T S + lambda I)^-1 S^T Y        (deterministic, no lr to tune)

with the bias folded in as an extra all-ones column. The penalty is resolved
per problem (``resolve_ridge_lambda``): a narrow readout keeps the legacy
RIDGE_LAMBDA floor, a wide one gets a deterministic inner-CV choice, because
a hardcoded floor is effectively unregularization once the feature count
dwarfs the sample count. Requires numpy; without it the trainer falls back to
multinomial logistic regression with a small learning rate floor of 1e-3 (a
head trained at a backbone-style lr never learns - the "lr underdose" failure
mode this leaf tests against).

The reservoir itself is NEVER trained: it is read from a .fly artifact or an
adjacency dict and stays frozen. Training is CPU-only and deterministic
(sorted classes, fixed iteration order, closed form or fixed seed).
"""

import json
import math
import sys
from pathlib import Path

try:
    import numpy as _np
except ImportError:  # stdlib fallback; never install anything
    _np = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate import choose_thresholds
from states import load_states_config, states

LR_FLOOR = 1e-3
# The legacy floor penalty. It is the right value for a NARROW readout
# (features <= WIDE_READOUT_DIM) and the wrong value for a wide one: at
# ~2952 features on ~289 samples/fold it is effectively unregularized, and
# the resulting readout made a working reservoir look useless (measured
# all-case accuracy 0.125 -> 0.726 once the penalty was corrected; see
# tools/eval/e1_corrected_results.json). It is therefore no longer the
# blanket default for wide readouts - resolve_ridge_lambda() picks the
# penalty per problem (deterministic inner CV).
RIDGE_LAMBDA = 1e-3
# At/above this feature count a readout is "wide" relative to the sample
# counts this trainer sees, so the caller-supplied floor is overridden by
# the inner-CV choice. Narrow readouts (the tiny fixtures, the orientation
# repro) keep the legacy value exactly.
WIDE_READOUT_DIM = 64
# Candidate penalties, ascending. Spans the measured optima: 0.1 for a
# 1024-dim raw-embedding probe, 10-30 for 2-3k-dim reservoir states.
RIDGE_LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
INNER_FOLDS = 4
INNER_SEED = 42
# Fewest rows for which the inner CV is meaningful (~4 rows per inner fold).
# Below this the documented heuristic is used instead; ties inside the inner
# CV still break toward the smallest candidate, so degenerate data degrades
# to the legacy floor on its own.
MIN_AUTO_ROWS = 4 * INNER_FOLDS
# Documented heuristic fallback (used only when the sample count is too
# small for the inner CV to be meaningful): shrink in proportion to the
# parameter-to-sample ratio.
HEURISTIC_SCALE = 1.0
REPORT_KEYS = ("train_accuracy", "thresholds", "pair_count")


def sorted_classes(pairs):
    """Sorted unique labels - fixed class order keeps outputs deterministic."""
    return sorted({label for _, label in pairs})


def _one_hot(y, classes):
    idx = {c: i for i, c in enumerate(classes)}
    Y = [[0.0] * len(classes) for _ in y]
    for row, label in zip(Y, y):
        row[idx[label]] = 1.0
    return Y


def _eval_states(pairs, cfg):
    return [(states(cfg, x), label) for x, label in pairs]


def _softmax_probs(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    z = sum(exps)
    return [e / z for e in exps]


def _inner_folds(labels, folds, seed):
    """Stratified inner-fold ids: shuffle per class, position % folds."""
    if _np is None:
        raise RuntimeError("_inner_folds requires numpy")
    n = len(labels)
    fold = _np.zeros(n, dtype=int)
    rng = _np.random.default_rng(seed)
    arr = _np.asarray(labels)
    for cls in sorted(set(labels)):
        idx = _np.where(arr == cls)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % folds
    return fold


def heuristic_ridge_lambda(n_features, n_samples):
    """Documented fallback penalty: shrink with the parameter-to-sample ratio.

    Used only when the sample count is too small for a meaningful inner CV,
    and only for wide readouts. Coarse by construction (the measured optima
    are data-dependent), but never the effectively-unregularized 1e-3.
    """
    return max(RIDGE_LAMBDA, HEURISTIC_SCALE * float(n_features) / max(int(n_samples), 1))


def auto_ridge_lambda(S, Y, candidates=RIDGE_LAMBDA_GRID, folds=INNER_FOLDS, seed=INNER_SEED):
    """Deterministic inner-CV ridge penalty for the given design matrix/targets.

    The nested split is a stratified k-fold over the rows it is given (the
    caller passes TRAIN-fold rows only, so no leak). Each candidate's score is
    the mean inner-validation argmax accuracy; ties go to the smaller penalty
    (ascending scan, strict improvement required). The candidate loop reuses
    one symmetric eigendecomposition per inner fold, so 10 candidates cost one
    ``eigh`` plus matmuls rather than ten 3k x 3k solves.

    Narrow readouts (features < WIDE_READOUT_DIM) or too-few rows get the
    legacy RIDGE_LAMBDA, which keeps the small-fixture behaviour byte-identical.
    """
    if _np is None:
        return RIDGE_LAMBDA
    X = _np.asarray(S, dtype=_np.float64)
    Yv = _np.asarray(Y, dtype=_np.float64)
    if X.ndim != 2 or Yv.ndim != 2 or X.shape[0] != Yv.shape[0]:
        raise ValueError("S and Y must be 2-D with matching row counts")
    n, d = X.shape
    if d < WIDE_READOUT_DIM or n < MIN_AUTO_ROWS:
        return RIDGE_LAMBDA

    labels = [int(i) for i in _np.argmax(Yv, axis=1)]
    inner = _inner_folds(labels, folds, seed)
    acc = {float(c): [] for c in candidates}
    for f in range(folds):
        tr = inner != f
        va = ~tr
        if not tr.any() or not va.any():
            continue
        Xtr = _np.hstack([X[tr], _np.ones((int(tr.sum()), 1))])
        A0 = Xtr.T @ Xtr
        G = Xtr.T @ Yv[tr]
        w, V = _np.linalg.eigh(A0)  # A0 symmetric positive semidefinite
        VtG = V.T @ G
        Xv = _np.hstack([X[va], _np.ones((int(va.sum()), 1))])
        gold = [_np.argmax(row) for row in Yv[va]]
        for c in candidates:
            sol = V @ (VtG / (w + float(c))[:, None])
            pred = _np.argmax(Xv @ sol, axis=1)
            acc[float(c)].append(float((pred == gold).mean()))

    best, best_acc = float(RIDGE_LAMBDA), -1.0
    for c in candidates:  # ascending: first strict improvement wins
        vals = acc[float(c)]
        a = sum(vals) / len(vals) if vals else 0.0
        if a > best_acc + 1e-12:
            best, best_acc = float(c), a
    return best


def resolve_ridge_lambda(S, Y, lam=None):
    """The penalty actually used: explicit ``lam`` wins, else the wide-readout
    inner-CV choice (or the documented heuristic when there are too few rows),
    else the legacy floor for narrow readouts."""
    if lam is not None:
        return float(lam)
    if _np is None:
        return RIDGE_LAMBDA
    X = _np.asarray(S, dtype=_np.float64)
    if X.ndim != 2 or X.shape[1] < WIDE_READOUT_DIM:
        return RIDGE_LAMBDA
    if X.shape[0] < MIN_AUTO_ROWS:
        return heuristic_ridge_lambda(X.shape[1], X.shape[0])
    return auto_ridge_lambda(S, Y)


def ridge_fit(S, Y, lam=None):
    """Closed-form ridge with bias column: returns (W [N][K], bias [K]).

    ``lam=None`` (the default) resolves the penalty per problem via
    resolve_ridge_lambda: narrow readouts keep RIDGE_LAMBDA, wide ones get a
    deterministic inner-CV choice. Pass an explicit ``lam`` to override.
    """
    if _np is None:
        raise RuntimeError("ridge_fit requires numpy; use logistic_train")
    lam = resolve_ridge_lambda(S, Y, lam)
    X = _np.asarray(S, dtype=_np.float64)
    Yv = _np.asarray(Y, dtype=_np.float64)
    Xb = _np.hstack([X, _np.ones((X.shape[0], 1))])  # bias column
    A = Xb.T @ Xb + lam * _np.eye(Xb.shape[1])
    B = Xb.T @ Yv
    sol = _np.linalg.solve(A, B)  # (N+1, K); deterministic LAPACK path
    return sol[:-1, :].tolist(), sol[-1, :].tolist()


def logistic_train(S, y, classes, lr=LR_FLOOR, epochs=400, l2=1e-4):
    """Stdlib multinomial logistic regression fallback.

    lr defaults to the 1e-3 head floor (backbone-scale rates underdose the
    head; see the leaf's underdose test). Deterministic: fixed sample and
    class order, no shuffling.
    """
    K = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    Y = _one_hot(y, classes)
    n = len(S)
    dim = len(S[0]) if S else 0
    W = [[0.0] * K for _ in range(dim)]
    bias = [0.0] * K
    if lr < LR_FLOOR:
        lr = LR_FLOOR  # head floor: below this the head underdoses
    for _ in range(epochs):
        gradW = [[0.0] * K for _ in range(dim)]
        gradb = [0.0] * K
        for s, yv in zip(S, Y):
            logits = [sum(w * v for w, v in zip(row, s)) + b for row, b in zip(W, bias)]
            probs = _softmax_probs(logits)
            for j in range(K):
                d = probs[j] - yv[j]
                gradb[j] += d
                for i, si in enumerate(s):
                    gradW[i][j] += d * si
        for j in range(K):
            bias[j] -= lr * gradb[j] / n
            for i in range(dim):
                gradW[i][j] = gradW[i][j] / n + l2 * W[i][j]
                W[i][j] -= lr * gradW[i][j]
    return W, bias


def fit_readout(S, y, classes, lam=None):
    """Train the linear readout on frozen states.

    S: list of state vectors; y: labels; classes: fixed class order.
    Returns (W [N][K] nested lists, bias [K]). Deterministic: ridge closed
    form when numpy is present, fixed-order logistic otherwise.

    ``lam=None`` resolves the ridge penalty per problem (inner CV for wide
    readouts; the legacy floor for narrow ones) - see resolve_ridge_lambda.
    """
    Y = _one_hot(y, classes)
    if _np is not None:
        return ridge_fit(S, Y, lam=lam)
    return logistic_train(S, y, classes)


def train_from_pairs(fly_path, pairs_path, target_precision=0.97, lam=None):
    """Full offline training run: states -> readout -> thresholds.

    ``lam=None`` (default) resolves the ridge penalty from the training
    states themselves via resolve_ridge_lambda, so a wide readout no longer
    silently trains at the effectively-unregularized 1e-3 floor.

    Returns (export_dict, report_dict) ready for JSON serialization. Both
    keep their exact Contract 3 key sets (deliberately no extra fields: the
    committed export/report consumers pin them); call
    ``resolve_ridge_lambda(S, Y)`` (or pass ``lam=``) if the chosen penalty
    needs to be logged.
    """
    cfg = load_states_config(fly_path)
    classes = []
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pairs.append((rec["embedding"], rec["label"]))
    if not pairs:
        raise ValueError("no training pairs")
    classes = sorted_classes(pairs)
    if len(classes) < 2:
        raise ValueError("need at least 2 classes")

    S_pairs = _eval_states(pairs, cfg)
    S = [s for s, _ in S_pairs]
    y = [lab for _, lab in S_pairs]

    if _np is not None:
        W, bias = ridge_fit(S, _one_hot(y, classes), lam=lam)
    else:
        W, bias = logistic_train(S, y, classes)

    acc, mean_loss = score_readout(W, bias, S, y, classes)
    ln_k = math.log(len(classes))
    if mean_loss >= ln_k:
        raise RuntimeError(
            "readout failed to learn: mean loss %.4f >= ln(K) %.4f "
            "(lr underdose or degenerate data)" % (mean_loss, ln_k)
        )

    scores = _class_scores(W, bias, S, classes)
    thresholds = choose_thresholds(scores, y, target_precision=target_precision)

    export = {
        "classes": classes,
        "W": W,
        "bias": bias,
        "threshold": [float(thresholds[c]) for c in classes],
        "weight_scale": _max_abs_scale(W),
        "readout_scale": _max_abs_scale(W),
    }
    report = {
        "train_accuracy": acc,
        "thresholds": thresholds,
        "pair_count": len(pairs),
    }
    return export, report


def _logits(W, bias, s):
    """logits[j] = W[:, j] . s + bias[j] for W stored [N][K] (Contract 3).

    W is neuron-major (one row per neuron, one column per class); bias is
    length K. The prior implementation zipped W's rows with bias, treating
    rows as class-major - correct only by accident of shape, wrong for every
    N != K (and for N == K too, since W is not symmetric).
    """
    return [sum(W[i][j] * v for i, v in enumerate(s)) + bias[j]
            for j in range(len(bias))]


def score_readout(W, bias, S, y, classes):
    """Train accuracy and mean cross-entropy of a fitted readout."""
    idx = {c: i for i, c in enumerate(classes)}
    correct = 0
    total = 0.0
    for s, label in zip(S, y):
        logits = _logits(W, bias, s)
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        logz = math.log(sum(exps))
        total += -(logits[idx[label]] - m - logz)
        if max(range(len(classes)), key=lambda j: logits[j]) == idx[label]:
            correct += 1
    return correct / max(len(S), 1), total / max(len(S), 1)


def _class_scores(W, bias, S, classes):
    """Per-row softmax over lanes: the route-probability rows calibration consumes."""
    rows = []
    for s in S:
        logits = _logits(W, bias, s)
        probs = _softmax_probs(logits)
        rows.append({classes[j]: probs[j] for j in range(len(classes))})
    return rows


def _max_abs_scale(W, floor=1e-6):
    """float64 W -> int8-friendly scale (max |w| / 127, floored)."""
    m = max((abs(v) for row in W for v in row), default=0.0)
    return max(m / 127.0, floor)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    import argparse

    p = argparse.ArgumentParser(description="Train the classi-fly linear readout")
    p.add_argument("--fly", required=True, help="path to the .fly artifact")
    p.add_argument("--pairs", required=True, help="pairs.jsonl ({embedding,label} per line)")
    p.add_argument("--out", required=True, help="readout export path (Contract 3 JSON)")
    p.add_argument("--report", help="optional readout_report.json path")
    p.add_argument("--lr", type=float, default=LR_FLOOR, help="logistic head lr (numpy absent)")
    p.add_argument("--epochs", type=int, default=200, help="logistic epochs (numpy absent)")
    p.add_argument(
        "--calib", default="quantile", choices=["quantile"], help="calibration method"
    )
    p.add_argument("--target-precision", type=float, default=0.97)
    p.add_argument(
        "--ridge-lambda",
        type=float,
        default=None,
        help="ridge penalty override; default = auto (inner CV for wide readouts)",
    )
    args = p.parse_args(argv)

    export, report = train_from_pairs(
        args.fly, args.pairs, target_precision=args.target_precision, lam=args.ridge_lambda
    )
    Path(args.out).write_text(json.dumps(export) + "\n", encoding="utf-8")
    if args.report:
        Path(args.report).write_text(json.dumps(report) + "\n", encoding="utf-8")
    # Print the report and nothing else.
    print(json.dumps(report))


if __name__ == "__main__":
    main()
