"""Linear readout training on frozen reservoir states (leaf 04 Tasks 2 and 4).

Preferred solver: closed-form ridge regression on one-hot targets,

    W = (S^T S + lambda I)^-1 S^T Y        (deterministic, no lr to tune)

with the bias folded in as an extra all-ones column. Requires numpy; without
it the trainer falls back to multinomial logistic regression with a small
learning rate floor of 1e-3 (a head trained at a backbone-style lr never
learns - the "lr underdose" failure mode this leaf tests against).

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
RIDGE_LAMBDA = 1e-3
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


def ridge_fit(S, Y, lam=RIDGE_LAMBDA):
    """Closed-form ridge with bias column: returns (W [N][K], bias [K])."""
    if _np is None:
        raise RuntimeError("ridge_fit requires numpy; use logistic_train")
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


def fit_readout(S, y, classes, lam=RIDGE_LAMBDA):
    """Train the linear readout on frozen states.

    S: list of state vectors; y: labels; classes: fixed class order.
    Returns (W [N][K] nested lists, bias [K]). Deterministic: ridge closed
    form when numpy is present, fixed-order logistic otherwise.
    """
    Y = _one_hot(y, classes)
    if _np is not None:
        return ridge_fit(S, Y, lam=lam)
    return logistic_train(S, y, classes)


def train_from_pairs(fly_path, pairs_path, target_precision=0.97, lam=RIDGE_LAMBDA):
    """Full offline training run: states -> readout -> thresholds.

    Returns (export_dict, report_dict) ready for JSON serialization.
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


def score_readout(W, bias, S, y, classes):
    """Train accuracy and mean cross-entropy of a fitted readout."""
    idx = {c: i for i, c in enumerate(classes)}
    correct = 0
    total = 0.0
    for s, label in zip(S, y):
        logits = [sum(w * v for w, v in zip(row, s)) + b for row, b in zip(W, bias)]
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
        logits = [sum(w * v for w, v in zip(row, s)) + b for row, b in zip(W, bias)]
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
    args = p.parse_args(argv)

    export, report = train_from_pairs(
        args.fly, args.pairs, target_precision=args.target_precision
    )
    Path(args.out).write_text(json.dumps(export) + "\n", encoding="utf-8")
    if args.report:
        Path(args.report).write_text(json.dumps(report) + "\n", encoding="utf-8")
    # Print the report and nothing else.
    print(json.dumps(report))


if __name__ == "__main__":
    main()
