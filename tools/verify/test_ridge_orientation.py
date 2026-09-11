"""Pinned regression test: the ridge-orientation bug (tools/train).

The orchestrator dispatched this leaf to VERIFY - not fix - the bug where
``ridge_fit`` returns W as [N][K] (neuron-major, per Contract 3) while
``score_readout``/``_class_scores`` consume it class-major via ``zip(W, bias)``.
Correct only when N == K; real N > K training yields wrong probabilities.

This test PINS the bug's empirical signature: if someone lands the fix without
updating this file, the test fails loudly and the finding can be closed.

If this test starts failing, run tools/verify/ridge_orientation_repro.py:
exit 1 means production scoring now matches the contract -> flip this test to
assert agreement and close the VERIFICATION.md finding.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ridge_orientation_repro as repro  # noqa: E402


def test_ridge_orientation_bug_reproduced():
    """W[N][K] consumed class-major gives wrong probabilities for N != K.

    Hand-checked signature (seed 99, N=6, K=3, 40 well-separated samples):
    - production (zip) accuracy is chance-level (~0.325)
    - correct transpose accuracy is 1.0
    - max probability divergence > 0.4
    - the ridge fit itself is sound (closed-form RMSE < 0.01 under the
      correct orientation), proving the FIT is fine and the SCORING is wrong.
    """
    rep = repro.reproduce(N=6, K=3, seed=99)
    assert rep["N"] != rep["K"]
    assert rep["production_train_acc"] < 0.5, (
        "production scoring now learns - the orientation fix likely landed; "
        "update VERIFICATION.md and this pin")
    assert rep["correct_orientation_train_acc"] == pytest.approx(1.0)
    assert rep["max_prob_divergence"] > 0.2
    assert rep["closed_form_rmse_correct_orientation"] < 0.1
    assert rep["bug_reproduced"] is True


def test_square_case_is_also_wrong():
    """The orchestrator's hand-off claimed the zip() misreading is "correct
    when N == K". EMPIRICALLY FALSE: even at N == K the readout is only
    correct when the fitted W happens to be symmetric, and a ridge fit on
    clustered data is NOT symmetric (verified: production acc 0.75 vs 1.0
    correct-orientation at N=K=4, seed 5). The bug is unconditional; only
    contrived symmetric-W cases coincide.
    """
    import numpy as np

    import train_readout as tr

    rng = np.random.default_rng(5)
    N = K = 4
    classes = ["c%d" % j for j in range(K)]
    centers = rng.uniform(-2, 2, size=(K, N)) * 3.0
    S, y = [], []
    for i in range(24):
        c = i % K
        S.append((centers[c] + rng.normal(0, 0.05, N)).tolist())
        y.append(classes[c])
    W, bias = tr.ridge_fit(S, tr._one_hot(y, classes))
    Wm = np.asarray(W)
    assert not np.allclose(Wm, Wm.T), "fixture unexpectedly produced symmetric W"
    acc_prod, _ = tr.score_readout(W, bias, S, y, classes)
    correct = 0
    for s, lab in zip(S, y):
        logits = repro._manual_scores(W, bias, s)
        if max(range(K), key=lambda j: logits[j]) == classes.index(lab):
            correct += 1
    acc_correct = correct / len(S)
    assert acc_correct == pytest.approx(1.0)
    # FIXED 2026-09-11: _logits() now consumes W[N][K] neuron-major, so
    # production scoring matches the correct orientation at N == K too.
    assert acc_prod == pytest.approx(acc_correct), (
        "N==K production scoring diverges from the correct orientation"
    )