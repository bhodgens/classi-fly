"""Empirical reproduction of the readout orientation bug in tools/train.

KNOWN ISSUE (orchestrator hand-off, to be verified here - NOT fixed here):
``tools/train/train_readout.py::ridge_fit`` solves
``Xb = [S | 1]`` against one-hot ``Y`` and returns ``sol[:-1, :]`` as ``W``,
i.e. ``W`` has shape [N][K] (one ROW per NEURON, one COLUMN per class).
But ``score_readout`` and ``_class_scores`` compute logits as
``zip(W, bias)`` - i.e. they treat each ROW of W as one CLASS's weight
vector, which is only correct when N == K. For real training sets N > K,
so logits[j] mixes neuron i=j's activation across ALL class columns,
plus bias[j] which is class j's bias applied to the wrong lane.

This script reproduces it numerically with NO numpy tricks and NO fix: it
calls the production functions on a 6-neuron / 3-class synthetic state set,
then compares against a hand-computed transposed orientation (the Contract 3
interpretation W[N][K] consumed as logits = h @ W + bias).

Run:  python3 tools/verify/ridge_orientation_repro.py
Exit 0 when the bug is REPRODUCED (production scoring disagrees with the
correct orientation); exit 1 when it is not (i.e. the bug is gone - the fix
landed - and this finding can be closed).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "train"))

import train_readout as tr  # noqa: E402


def _manual_scores(W, bias, h):
    """Contract 3 semantics: W is [N][K], logits = h @ W + bias (per class k:
    sum_i h[i] * W[i][k] + bias[k]). Pure numpy; independent of train_readout."""
    Wm = np.asarray(W, dtype=np.float64)          # [N][K]
    bm = np.asarray(bias, dtype=np.float64)       # [K]
    return (np.asarray(h, dtype=np.float64) @ Wm + bm).tolist()


def _probs(logits):
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    z = sum(e)
    return [v / z for v in e]


def reproduce(N: int = 6, K: int = 3, n_samples: int = 40, seed: int = 99) -> dict:
    rng = np.random.default_rng(seed)
    classes = ["c%d" % j for j in range(K)]
    # Synthetic FROZEN STATES directly (no reservoir involved): the bug is in
    # the readout math, so the state source is irrelevant. Well-separated
    # clusters so a correct readout is trivially learnable.
    centers = rng.uniform(-2, 2, size=(K, N)) * 3.0
    S, y = [], []
    for i in range(n_samples):
        c = i % K
        S.append((centers[c] + rng.normal(0, 0.05, N)).tolist())
        y.append(classes[c])

    W, bias = tr.ridge_fit(S, tr._one_hot(y, classes))  # production fit

    # Production scoring path (zip(W, bias) => class-major interpretation).
    prod_scores = [list(map(sum, zip(*[[w * v for w, v in zip(row, s)] + [b]
                                       for row, b in zip(W, bias)])))
                   if False else
                   [sum(w * v for w, v in zip(row, s)) + b
                    for row, b in zip(W, bias)]
                   for s in S]
    # ^ exactly the production expression from score_readout/_class_scores.

    # Correct orientation: W[N][K], logits_k = sum_i h_i * W[i][k] + bias_k.
    correct_scores = [_manual_scores(W, bias, h) for h in S]

    prod_probs = [_probs(l) for l in prod_scores]
    correct_probs = [_probs(l) for l in correct_scores]

    prod_correct = sum(
        1 for p, lab in zip(prod_probs, y)
        if max(range(K), key=lambda j: p[j]) == classes.index(lab))
    corr_correct = sum(
        1 for p, lab in zip(correct_probs, y)
        if max(range(K), key=lambda j: p[j]) == classes.index(lab))

    # And the ridge fit is sound: verify the CLOSED FORM really fits under the
    # correct orientation (train-set fit via the same manual math).
    S_np = np.asarray(S)
    Y_np = np.asarray(tr._one_hot(y, classes))
    Xb = np.hstack([S_np, np.ones((len(S), 1))])
    fitted = Xb @ np.vstack([np.asarray(W), np.asarray(bias)])
    closed_form_rmse = float(np.sqrt(np.mean((fitted - Y_np) ** 2)))

    max_prob_diff = float(np.max(np.abs(np.asarray(prod_probs)
                                        - np.asarray(correct_probs))))
    return {
        "N": N, "K": K, "n_samples": n_samples,
        "production_train_acc": prod_correct / n_samples,
        "correct_orientation_train_acc": corr_correct / n_samples,
        "max_prob_divergence": max_prob_diff,
        "closed_form_rmse_correct_orientation": closed_form_rmse,
        "bug_reproduced": bool(
            N != K
            and prod_correct < corr_correct
            and max_prob_diff > 0.05
            and closed_form_rmse < 0.5),  # fit is good; scoring is wrong
        "sample_probabilities": {
            "production_zip_W_bias": [round(v, 4) for v in prod_probs[0]],
            "correct_transpose": [round(v, 4) for v in correct_probs[0]],
            "gold_one_hot": tr._one_hot([y[0]], classes)[0],
        },
    }


def main(argv=None) -> int:
    rep = reproduce()
    print(json.dumps(rep, indent=2))
    if rep["bug_reproduced"]:
        print("\nCONFIRMED: ridge_fit's W[N][K] is consumed class-major by "
              "score_readout/_class_scores; probabilities are wrong for N != K.")
        return 0
    print("\nNOT REPRODUCED: production scoring matches the W[N][K] "
          "contract (fix may have landed).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
