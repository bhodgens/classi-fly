"""Tests for tools/train/calibrate.py (leaf 04 Task 3)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate import choose_thresholds

# Fixture design (deterministic, no RNG):
# - "easy": its lane is never contaminated by other classes, so the lowest
#   quantile of its own scores already meets the precision target.
# - "confusable": a third class leaks INTO the confusable lane just above its
#   lowest own scores, so only a high quantile of its own scores clears the
#   leak band and meets the precision target.
EASY_N = 25
CONF_N = 25
LEAK_N = 30


def _fixture_rows():
    scores, labels = [], []
    for i in range(EASY_N):
        scores.append({"easy": 0.30 + 0.008 * i, "confusable": 0.01, "third": 0.02})
        labels.append("easy")
    for i in range(CONF_N):
        scores.append({"easy": 0.02, "confusable": 0.90 + 0.004 * i, "third": 0.02})
        labels.append("confusable")
    # Leak rows score 0.9005..0.944 in the confusable lane - inside the
    # confusable class's own band (0.90..0.996), above its bottom.
    for i in range(LEAK_N):
        scores.append(
            {"easy": 0.02, "confusable": 0.9005 + 0.0015 * i, "third": 0.30 + 0.001 * i}
        )
        labels.append("third")
    return scores, labels


class TestChooseThresholds(unittest.TestCase):
    def test_easy_class_gets_lower_threshold(self):
        scores, labels = _fixture_rows()
        th = choose_thresholds(scores, labels, target_precision=0.97)
        self.assertEqual(set(th.keys()), {"easy", "confusable", "third"})
        self.assertLess(
            th["easy"],
            th["confusable"],
            "easy class must get a lower threshold than the confusable one",
        )
        # easy: nothing ever leaks into its lane -> threshold sits at the
        # bottom of its own observed scores (its 0th quantile).
        self.assertAlmostEqual(th["easy"], 0.30, places=9)
        # confusable: threshold must sit above the leak band (max leak 0.885).
        self.assertGreater(th["confusable"], 0.885)
        self.assertLessEqual(th["confusable"], 0.90 + 0.004 * (CONF_N - 1))

    def test_thresholds_stay_within_observed_range(self):
        scores, labels = _fixture_rows()
        th = choose_thresholds(scores, labels, target_precision=0.97)
        easy_scores = sorted(r["easy"] for r, l in zip(scores, labels) if l == "easy")
        self.assertGreaterEqual(th["easy"], easy_scores[0])

    def test_quantile_not_constant(self):
        """Thresholds must come from data, not a hard-coded constant."""
        scores, labels = [], []
        for i in range(30):
            scores.append({"only": 0.80 + 0.001 * i})
            labels.append("only")
        th = choose_thresholds(scores, labels, target_precision=0.97)
        self.assertAlmostEqual(th["only"], 0.80, places=9)

    def test_deterministic(self):
        scores, labels = _fixture_rows()
        t1 = choose_thresholds(scores, labels, target_precision=0.97)
        t2 = choose_thresholds(scores, labels, target_precision=0.97)
        self.assertEqual(t1, t2)


if __name__ == "__main__":
    unittest.main()
