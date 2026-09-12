"""Tests for tools/train/train_readout.py (leaf 04 Tasks 2 and 4).

The key assertion: the trainer must actually learn (loss < ln(K), accuracy
well above chance) so the lr-underdose failure mode (a fresh head that never
learns, pinned at ln(K)) is explicitly tested against.
"""

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixture
import flybytes
import numpy as _np
from train_readout import (
    RIDGE_LAMBDA,
    RIDGE_LAMBDA_GRID,
    REPORT_KEYS,
    _one_hot,
    fit_readout,
    heuristic_ridge_lambda,
    logistic_train,
    resolve_ridge_lambda,
    ridge_fit,
    train_from_pairs,
)
from calibrate import choose_thresholds


def _separable_states(n_per_class=20, dim=None):
    """Deterministic separable synthetic states (two tight clusters)."""
    dim = dim or fixture.NEURONS
    from states import states as _states, load_states_config

    cfg = load_states_config(fixture.tiny_artifact())
    X, y = [], []
    for i in range(n_per_class):
        x = [0.55, 0.35 + 0.01 * (i % 3), -0.20, 0.10]
        y.append("alpha")
        X.append(_states(cfg, x))
        x2 = [-0.45 + 0.01 * (i % 2), -0.30, 0.25, -0.08]
        y.append("beta")
        X.append(_states(cfg, x2))
    return X, y


class TestFitReadout(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _separable_states()
        self.K = len(fixture.CLASSES)

    def test_reaches_above_chance_accuracy(self):
        W, bias = fit_readout(self.X, self.y, fixture.CLASSES)
        acc, mean_loss = _eval(W, bias, self.X, self.y, fixture.CLASSES)
        self.assertGreater(acc, 0.90)
        self.assertLess(mean_loss, math.log(self.K), "loss pinned at ln(K): head never learned")
        self.assertLess(mean_loss, math.log(self.K) - 0.1)

    def test_deterministic_closed_form(self):
        W1, b1 = fit_readout(self.X, self.y, fixture.CLASSES)
        W2, b2 = fit_readout(self.X, self.y, fixture.CLASSES)
        self.assertEqual(W1, W2)
        self.assertEqual(b1, b2)

    def test_logistic_fallback_also_learns(self):
        W, bias = logistic_train(
            self.X, self.y, fixture.CLASSES, lr=1e-3, epochs=400, l2=1e-4
        )
        acc, mean_loss = _eval(W, bias, self.X, self.y, fixture.CLASSES)
        self.assertGreater(acc, 0.90)
        self.assertLess(mean_loss, math.log(self.K), "lr-underdose: loss pinned at ln(K)")


def _eval(W, bias, X, y, classes):
    idx = {c: i for i, c in enumerate(classes)}
    correct = 0
    total_loss = 0.0
    K = len(classes)
    for x, label in zip(X, y):
        logits = [
            sum(w * s for w, s in zip(row, x)) + b for row, b in zip(W, bias)
        ]
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        z = sum(exps)
        probs = [e / z for e in exps]
        pred = classes[max(range(K), key=lambda i: probs[i])]
        if pred == label:
            correct += 1
        total_loss += -math.log(max(probs[idx[label]], 1e-12))
    n = len(y)
    return correct / n, total_loss / n


class TestRidgeLambdaPolicy(unittest.TestCase):
    """Pins the wide-readout ridge penalty policy (repo defect fix).

    The defect: RIDGE_LAMBDA was a hardcoded 1e-3, which is effectively
    unregularized for a ~2952-dim readout on ~289 samples/fold and made a
    working reservoir look useless (measured all-case accuracy 0.125 -> 0.726
    once corrected). The policy: the penalty is resolved per problem -
    deterministic inner CV for wide readouts, the legacy floor for narrow
    ones, a documented ratio heuristic when there are too few rows to split.
    """

    @staticmethod
    def _wide_problem(seed=12, dim=400, per_class=15, signal=3, noise=2.0,
                      test_n=80):
        """Overcomplete problem: a 3-dim class signal buried in 397 pure-noise
        features with FEWER rows than columns, so lambda=1e-3 overfits."""
        rng = _np.random.default_rng(seed)
        centers = rng.normal(0, 1.5, size=(2, signal))

        def gen(n):
            X = _np.zeros((n, dim))
            y = []
            for i in range(n):
                c = i % 2
                y.append("c%d" % c)
                X[i, :signal] = centers[c] + rng.normal(0, noise, signal)
                X[i, signal:] = rng.normal(0, 1.0, dim - signal)
            return X, y

        Xtr, ytr = gen(per_class * 2)
        Xte, yte = gen(test_n)
        return Xtr, ytr, Xte, yte

    @staticmethod
    def _held_out_acc(Xtr, Ytr, Xte, yte, classes, lam):
        W, bias = ridge_fit(Xtr, Ytr, lam=lam)
        Xb = _np.hstack([Xte, _np.ones((Xte.shape[0], 1))])
        pred = _np.argmax(Xb @ _np.vstack([_np.asarray(W), _np.asarray(bias)]), axis=1)
        return float((_np.asarray(classes)[pred] == _np.asarray(yte)).mean())

    def test_wide_readout_penalty_is_not_the_legacy_floor(self):
        Xtr, ytr, Xte, yte = self._wide_problem()
        classes = ["c0", "c1"]
        Ytr = _one_hot(ytr, classes)
        lam = resolve_ridge_lambda(Xtr, Ytr)
        self.assertIn(lam, RIDGE_LAMBDA_GRID)
        self.assertGreater(lam, RIDGE_LAMBDA, "wide readout kept the 1e-3 floor")
        # deterministic: same design matrix -> same penalty, every call
        self.assertEqual(lam, resolve_ridge_lambda(Xtr, Ytr))
        # and the point of the fix: the resolved penalty does not do worse
        # (here: better) than the legacy floor on rows it never saw
        acc_auto = self._held_out_acc(Xtr, Ytr, Xte, yte, classes, lam)
        acc_legacy = self._held_out_acc(Xtr, Ytr, Xte, yte, classes, RIDGE_LAMBDA)
        self.assertGreaterEqual(acc_auto, acc_legacy - 1e-9)
        self.assertGreater(acc_auto, acc_legacy)

    def test_narrow_readout_keeps_the_legacy_floor(self):
        """Back-compat: the tiny fixtures and the orientation repro are narrow,
        so their fits must be byte-identical to the pre-fix behaviour."""
        X, y = _separable_states(dim=6)
        classes = fixture.CLASSES
        Y = _one_hot(y, classes)
        self.assertEqual(resolve_ridge_lambda(X, Y), RIDGE_LAMBDA)
        self.assertEqual(ridge_fit(X, Y), ridge_fit(X, Y, lam=RIDGE_LAMBDA))

    def test_fit_readout_uses_the_resolved_penalty(self):
        Xtr, ytr, _, _ = self._wide_problem()
        Ytr = _one_hot(ytr, ["c0", "c1"])
        lam = resolve_ridge_lambda(Xtr, Ytr)
        W_auto, b_auto = fit_readout(Xtr, ytr, ["c0", "c1"])
        W_exp, b_exp = fit_readout(Xtr, ytr, ["c0", "c1"], lam=lam)
        self.assertEqual(W_auto, W_exp)
        self.assertEqual(b_auto, b_exp)

    def test_too_few_rows_falls_back_to_documented_heuristic(self):
        X = _np.zeros((6, 80))  # wide but unsplittable
        Y = _one_hot(["a"] * 6, ["a"])
        self.assertEqual(resolve_ridge_lambda(X, Y), heuristic_ridge_lambda(80, 6))
        self.assertGreater(heuristic_ridge_lambda(2952, 289), RIDGE_LAMBDA)

    def test_explicit_lambda_still_wins(self):
        Xtr, ytr, _, _ = self._wide_problem()
        Ytr = _one_hot(ytr, ["c0", "c1"])
        self.assertEqual(resolve_ridge_lambda(Xtr, Ytr, lam=7.5), 7.5)
        self.assertEqual(
            fit_readout(Xtr, ytr, ["c0", "c1"], lam=0.25),
            fit_readout(Xtr, ytr, ["c0", "c1"], lam=0.25),
        )


class TestWideReadoutEndToEnd(unittest.TestCase):
    """train_from_pairs on a WIDE .fly artifact: the auto penalty path must run
    and both outputs must keep their exact Contract 3 key sets."""

    def setUp(self):
        self.tmp = Path("/tmp") / "classi-fly-train-test" / "wide"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.fly = self.tmp / "wide.fly"
        self.pairs = self.tmp / "pairs.jsonl"
        rng = _np.random.default_rng(3)
        n, emb = 100, 8
        indptr, indices, weights = [0], [], []
        for _ in range(n):
            for _ in range(3):
                indices.append(int(rng.integers(0, n)))
                weights.append(int(rng.integers(-8, 9)))
            indptr.append(len(indices))
        artifact = {
            "name": "wide", "neurons": n, "edges": len(indices), "embed_dim": emb,
            "steps": 4, "decay": 0.8, "classes": ["c0", "c1"], "indptr": indptr,
            "indices": indices, "weights": weights, "weight_scale": 1.0 / 127.0,
            "in_w": [int(v) for v in rng.integers(-6, 7, size=emb * n)],
            "in_scale": 0.02, "source": "synthetic", "license": "none",
            "attribution": "n/a",
        }
        flybytes.write_fly(self.fly, artifact)
        with self.pairs.open("w") as f:
            for i in range(40):
                c = i % 2
                x = [0.0] * emb
                x[0] = 0.5 if c == 0 else -0.5
                f.write(json.dumps({"embedding": x, "label": "c%d" % c}) + "\n")

    def test_wide_auto_path_keeps_both_contracts(self):
        export, report = train_from_pairs(str(self.fly), str(self.pairs))
        self.assertEqual(
            set(export.keys()),
            {"classes", "W", "bias", "threshold", "weight_scale", "readout_scale"},
        )
        self.assertEqual(set(report.keys()), set(REPORT_KEYS))
        self.assertEqual(report["pair_count"], 40)
        self.assertGreater(report["train_accuracy"], 0.9)


class TestEndToEndExport(unittest.TestCase):
    """Leaf Task 4: run the CLI against synthetic pairs, check both files."""

    def setUp(self):
        tmp = Path("/tmp") / "classi-fly-train-test" / "e2e"
        tmp.mkdir(parents=True, exist_ok=True)
        fly = tmp / "tiny.fly"
        flybytes.write_fly(fly, fixture.tiny_artifact())
        pairs = tmp / "pairs.jsonl"
        with pairs.open("w") as f:
            for x_emb, label in zip(
                [
                    [0.55, 0.35, -0.20, 0.10],
                    [0.52, 0.36, -0.21, 0.11],
                    [0.58, 0.33, -0.19, 0.09],
                    [-0.45, -0.30, 0.25, -0.08],
                    [-0.42, -0.32, 0.27, -0.07],
                    [-0.48, -0.28, 0.23, -0.09],
                ],
                ["alpha"] * 3 + ["beta"] * 3,
            ):
                f.write(json.dumps({"embedding": x_emb, "label": label}) + "\n")
        self.out = tmp / "readout.json"
        self.report = tmp / "readout_report.json"
        import subprocess

        r = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "train_readout.py"),
                "--fly",
                str(fly),
                "--pairs",
                str(pairs),
                "--out",
                str(self.out),
                "--report",
                str(self.report),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stderr.strip(), "", "unexpected stderr from CLI")
        self.assertEqual(
            set(json.loads(r.stdout).keys()),
            {"train_accuracy", "thresholds", "pair_count"},
            "CLI must print the report and nothing else",
        )

    def test_export_matches_contract3(self):
        export = json.loads(self.out.read_text())
        self.assertEqual(
            set(export.keys()),
            {"classes", "W", "bias", "threshold", "weight_scale", "readout_scale"},
        )
        self.assertEqual(export["classes"], fixture.CLASSES)
        self.assertEqual(len(export["W"]), fixture.NEURONS)
        self.assertEqual(len(export["W"][0]), len(fixture.CLASSES))
        self.assertEqual(len(export["bias"]), len(fixture.CLASSES))
        self.assertEqual(len(export["threshold"]), len(fixture.CLASSES))
        self.assertIsInstance(export["weight_scale"], float)
        self.assertIsInstance(export["readout_scale"], float)
        self.assertGreater(export["weight_scale"], 0.0)
        self.assertGreater(export["readout_scale"], 0.0)

    def test_report_fields(self):
        report = json.loads(self.report.read_text())
        self.assertEqual(set(report.keys()), {"train_accuracy", "thresholds", "pair_count"})
        self.assertEqual(report["pair_count"], 6)
        self.assertIsInstance(report["train_accuracy"], float)
        self.assertEqual(set(report["thresholds"].keys()), set(fixture.CLASSES))
        self.assertGreater(report["train_accuracy"], 0.99)


if __name__ == "__main__":
    unittest.main()
