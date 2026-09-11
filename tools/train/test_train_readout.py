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
from train_readout import fit_readout, logistic_train
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
