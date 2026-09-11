"""Adversarial verification tooling (leaf 08).

These tools intentionally do NOT import the eval harness's scoring code: the
whole point is to recompute the headline numbers from per-case data with an
independent formula. They read `results.json` fixtures and `.fly` artifacts
from disk and say what they find.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (
    os.path.join(_HERE, os.pardir, "eval"),
    os.path.join(_HERE, os.pardir, "train"),
):
    sys.path.insert(0, _p)
