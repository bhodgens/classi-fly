"""Tests for tools/train/states.py (leaf 04 Task 1)."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixture
import flybytes
from states import load_states_config, states

TMP = Path("/tmp") / "classi-fly-train-test"


def run_states_from_fly(path, embeddings):
    cfg = load_states_config(path)
    return [states(cfg, x) for x in embeddings]


class TestStates(unittest.TestCase):
    def test_fly_fixture_output_shape_and_range(self):
        path = TMP / "tiny.fly"
        path.parent.mkdir(parents=True, exist_ok=True)
        flybytes.write_fly(path, fixture.tiny_artifact())
        outs = run_states_from_fly(
            path,
            [
                [0.5, 0.25, -0.1, 0.05],
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.5, 0.25, -0.25],
            ],
        )
        self.assertEqual(len(outs), 3)
        for s in outs:
            self.assertEqual(len(s), fixture.NEURONS)
            for v in s:
                self.assertGreaterEqual(v, -1.0)
                self.assertLessEqual(v, 1.0)
                self.assertFalse(math.isnan(v))

    def test_adjacency_dict_matches_fly_fixture(self):
        art = fixture.tiny_artifact()
        cfg_dict = load_states_config(art)
        path = TMP / "tiny_match.fly"
        path.parent.mkdir(parents=True, exist_ok=True)
        flybytes.write_fly(path, art)
        cfg_fly = load_states_config(path)
        x = [0.5, 0.25, -0.1, 0.05]
        self.assertEqual(states(cfg_dict, x), states(cfg_fly, x))

    def test_deterministic_and_zero_start(self):
        art = fixture.tiny_artifact()
        cfg = load_states_config(art)
        x = [0.3, 0.1, -0.2, 0.4]
        a = states(cfg, x)
        b = states(cfg, x)
        self.assertEqual(a, b)
        z = states(cfg, [0.0] * fixture.EMBED_DIM)
        self.assertEqual(z, [0.0] * fixture.NEURONS)

    def test_self_consistent_golden_vector(self):
        """Golden vector, independently computed.

        2 neurons, W=[[0,1],[0,0]]/10 (weight_scale 0.1 -> stored ints
        [0,1,0,0]), win=[1.0,0.5] (in_w [10,5] * in_scale 0.1),
        decay=0.8, steps=3, x=[0.5].
        Recurrence s = tanh(decay*s + W*s + win*x), s0=0, iterated in double
        precision (verified with an independent one-off script):
        [0.8049862596769853, 0.5261753745004829].
        """
        art = {
            "neurons": 2,
            "edges": 1,
            "embed_dim": 1,
            "steps": 3,
            "decay": 0.8,
            "classes": ["a"],
            "indptr": [0, 1, 1],
            "indices": [1],
            "weights": [1],
            "weight_scale": 0.1,
            "in_w": [10, 5],
            "in_scale": 0.1,
        }
        cfg = load_states_config(art)
        got = states(cfg, [0.5])
        exp = [0.8049862596769853, 0.5261753745004829]
        for g, e in zip(got, exp):
            self.assertAlmostEqual(g, e, places=12)


if __name__ == "__main__":
    unittest.main()
