"""Tests pinning the safety-first gate's bars (owner decision 'OOD policy: safety').

Pinned properties:
  1. FAIL-SAFE: pure-noise embeddings are abstained for EVERY case, at
     several noise levels.
  2. Precision >= 0.95 on the 5-fold protocol (routed-but-wrong in the
     denominator).
  3. OOD abstain >= 0.95 on every available held-out OOD source.
  4. The critical calibration detail: calibrating class thresholds on the
     probe-PASSING subset is what holds precision - calibrating on ALL
     train rows (the last run's mistake) must be worse or equal.

Run:  python3 -m pytest tools/eval/test_safety_gate.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))
sys.path.insert(0, str(REPO / "tools" / "train"))

from safety_gate import (  # noqa: E402
    BadCalGate, SafetyGate, evaluate, folds_for, load, noise_fail_safe)

MIN_P = 0.95
MIN_OOD_ABSTAIN = 0.95


@pytest.fixture(scope="module")
def data():
    x, y, classes, gold_ood, syn_ood, real_ood = load()
    fold = folds_for(y, classes)
    ood_sources = {"gold": gold_ood, "synthetic": syn_ood}
    if real_ood is not None:
        ood_sources["real"] = real_ood
    return x, y, classes, fold, gold_ood, syn_ood, ood_sources


@pytest.fixture(scope="module")
def eval_results(data):
    """Run the 5-fold protocol once for the real gate and the all-rows ablation."""
    x, y, classes, fold, gold_ood, syn_ood, ood_sources = data
    good = evaluate(x, y, classes, fold, gold_ood, syn_ood,
                    ood_sources=ood_sources, target_p=0.97)
    bad = evaluate(x, y, classes, fold, gold_ood, syn_ood, ood_sources=ood_sources,
                   gate_cls=BadCalGate, target_p=0.97)
    return good, bad


def test_noise_fail_safe_every_case():
    """Pure-noise embeddings must be abstained for EVERY case (all sigmas)."""
    ok, msg = noise_fail_safe(classes=None, n=200)
    assert ok, msg


def test_noise_fail_safe_individual(data):
    """Same property asserted per-case on a directly fitted gate."""
    x, y, classes, _, gold_ood, syn_ood, _ = data
    rng = np.random.default_rng(123)
    gate = SafetyGate(classes).fit(x, y, np.vstack([gold_ood, syn_ood]))
    for sigma in (0.2, 0.5, 1.0):
        out = gate.predict(rng.normal(0.0, sigma, (100, x.shape[1])))
        assert not any(r["routed"] for r in out), f"routed noise at sigma={sigma}"


def test_precision_floor(eval_results):
    """Precision >= 0.95 on the 5-fold protocol."""
    good, _ = eval_results
    assert good["P"] >= MIN_P, f"P={good['P']} < {MIN_P} (routes={good['routes']})"


def test_ood_abstain_floor(eval_results):
    """OOD abstain >= 0.95 on every available held-out OOD source."""
    good, _ = eval_results
    for src, rate in good["ood_abstain"].items():
        assert rate >= MIN_OOD_ABSTAIN, f"OOD source '{src}' abstain {rate} < {MIN_OOD_ABSTAIN}"


def test_calibration_on_probe_passing_subset_is_what_holds_precision(eval_results):
    """Calibrating on the probe-PASSING subset must beat (or tie) calibrating
    on ALL train rows - the exact mistake that overstated precision last run."""
    good, bad = eval_results
    assert good["P"] >= bad["P"], (
        f"probe-subset calibration P={good['P']} should be >= all-rows "
        f"calibration P={bad['P']}")
