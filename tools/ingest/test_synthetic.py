"""Tests for tools/ingest/synthetic.py (leaf 03-synthetic-reservoir).

Runs under pytest or unittest:
    python3 -m pytest tools/ingest/test_synthetic.py -q
    python3 -m unittest tools.ingest.test_synthetic   (from repo root, or)
    python3 tools/ingest/test_synthetic.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from synthetic import generate, spectral_radius_estimate  # noqa: E402

SCRIPT = os.path.join(_HERE, "synthetic.py")

EXPECTED_KEYS = {
    "name", "neurons", "edges", "indptr", "indices", "weights",
    "source", "license", "attribution",
}


class TestDeterminism(unittest.TestCase):
    def test_deterministic(self):
        a = generate(seed=7, n=64, fan_in=4, inhib_frac=0.2)
        b = generate(seed=7, n=64, fan_in=4, inhib_frac=0.2)
        self.assertEqual(a["indptr"], b["indptr"])
        self.assertEqual(a["indices"], b["indices"])
        self.assertEqual(a["weights"], b["weights"])

    def test_different_seed_differs(self):
        a = generate(seed=1, n=64, fan_in=4, inhib_frac=0.2)
        b = generate(seed=2, n=64, fan_in=4, inhib_frac=0.2)
        self.assertNotEqual(a["indices"], b["indices"])

    def test_params_differ_differs(self):
        a = generate(seed=7, n=64, fan_in=4, inhib_frac=0.2)
        b = generate(seed=7, n=64, fan_in=5, inhib_frac=0.2)
        self.assertNotEqual(a["indices"], b["indices"])

    def test_cli_byte_identical(self):
        # The CLI path must reproduce generate() exactly, byte for byte.
        argv = [sys.executable, SCRIPT, "--seed", "7", "--neurons", "64",
                "--fan-in", "4", "--inhib-frac", "0.2"]
        outs = []
        with tempfile.TemporaryDirectory() as d:
            for i in (0, 1):
                path = os.path.join(d, f"adj_{i}.json")
                subprocess.run(argv + ["--out", path], check=True)
                with open(path, "rb") as f:
                    outs.append(f.read())
        self.assertEqual(outs[0], outs[1])
        obj = json.loads(outs[0])
        ref = generate(seed=7, n=64, fan_in=4, inhib_frac=0.2)
        self.assertEqual(obj["indptr"], ref["indptr"])
        self.assertEqual(obj["indices"], ref["indices"])
        self.assertEqual(obj["weights"], ref["weights"])


class TestCSRShape(unittest.TestCase):
    def test_csr_shape(self):
        a = generate(seed=3, n=32, fan_in=5, inhib_frac=0.2)
        self.assertEqual(len(a["indptr"]), 33)
        self.assertEqual(len(a["indices"]), len(a["weights"]))
        self.assertEqual(len(a["indices"]), a["edges"])
        self.assertEqual(a["indptr"][0], 0)
        self.assertEqual(a["indptr"][-1], a["edges"])

    def test_contract_fields(self):
        a = generate(seed=3, n=32, fan_in=5, inhib_frac=0.2)
        self.assertEqual(set(a.keys()), EXPECTED_KEYS)
        self.assertEqual(a["neurons"], 32)
        self.assertEqual(a["edges"], 32 * 5)
        self.assertEqual(a["source"], "synthetic")
        self.assertEqual(a["license"], "none")
        self.assertEqual(a["attribution"],
                         "seed-generated (classi-fly); no third-party data")

    def test_no_self_loops_and_fan_in_respected(self):
        n, fan_in = 48, 6
        a = generate(seed=11, n=n, fan_in=fan_in, inhib_frac=0.2)
        self.assertEqual(a["edges"], n * fan_in)
        for i in range(n):
            row = a["indices"][a["indptr"][i]:a["indptr"][i + 1]]
            self.assertEqual(len(row), fan_in)
            self.assertEqual(len(set(row)), fan_in)  # distinct targets
            self.assertNotIn(i, row)                # no self-loop
            self.assertEqual(row, sorted(row))      # CSR row order

    def test_weights_nonzero_and_signed(self):
        a = generate(seed=5, n=256, fan_in=6, inhib_frac=0.5)
        self.assertTrue(all(w != 0.0 for w in a["weights"]))
        self.assertTrue(any(w > 0 for w in a["weights"]))
        self.assertTrue(any(w < 0 for w in a["weights"]))

    def test_no_inhibition_all_positive(self):
        a = generate(seed=5, n=32, fan_in=4, inhib_frac=0.0)
        self.assertTrue(all(w > 0 for w in a["weights"]))

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            generate(seed=1, n=8, fan_in=8, inhib_frac=0.2)  # fan_in >= n
        with self.assertRaises(ValueError):
            generate(seed=1, n=1, fan_in=1, inhib_frac=0.2)
        with self.assertRaises(ValueError):
            generate(seed=1, n=8, fan_in=2, inhib_frac=1.5)


class TestStability(unittest.TestCase):
    def test_spectral_radius_below_target_defaults(self):
        # Default parameters (n=2048, fan_in=6, inhib_frac=0.2, target 0.9).
        a = generate(seed=7)
        self.assertEqual(a["neurons"], 2048)
        est = spectral_radius_estimate(a)
        self.assertLess(est, 0.99)
        self.assertLess(est, 0.9 + 1e-9)

    def test_spectral_radius_respects_custom_target(self):
        a = generate(seed=9, n=128, fan_in=4, inhib_frac=0.2, target=0.5)
        self.assertLess(spectral_radius_estimate(a), 0.5 + 1e-9)

    def test_forced_unsafe_target_raises(self):
        with self.assertRaises(ValueError):
            generate(seed=7, n=64, fan_in=4, inhib_frac=0.2, target=1.0)
        with self.assertRaises(ValueError):
            generate(seed=7, n=64, fan_in=4, inhib_frac=0.2, target=1.7)
        with self.assertRaises(ValueError):
            generate(seed=7, n=64, fan_in=4, inhib_frac=0.2, target=0.0)

    def test_estimate_is_deterministic(self):
        a = generate(seed=7, n=64, fan_in=4, inhib_frac=0.2)
        self.assertEqual(spectral_radius_estimate(a),
                         spectral_radius_estimate(a))


if __name__ == "__main__":
    unittest.main()
