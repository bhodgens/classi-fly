"""Tests for tools/verify/quant_drift.py (leaf 08 Task 2).

Builds real .fly twins (float reference vs int8 matrix-mode) on the fly and
checks agreement on the probe battery using the Python reference reader.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flyio  # noqa: E402
import quant_drift as qd  # noqa: E402


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    d = tmp_path_factory.mktemp("quant")
    f = qd.build_float_fly(d / "float_ref.fly")
    i = qd.build_int8_fly(d / "int8.fly")
    return f, i


def test_twins_load_and_share_topology(twins):
    f, i = twins
    rf, ri = flyio.Reservoir.Load(f), flyio.Reservoir.Load(i)
    assert rf._m.n == ri._m.n == 6
    assert rf._m.classes == ri._m.classes == ["a", "b"]
    assert rf._m.steps == ri._m.steps
    # Different input modes (seed-expanded float vs stored int8 matrix).
    assert rf._m.in_w is None and ri._m.in_w is not None


def test_float_vs_int8_agree_on_all_probes(twins):
    f, i = twins
    rep = qd.compare(f, i, embed_dim=4)
    assert rep["probes"] >= 64
    assert rep["class_disagreements"] == 0, qd.format_report(rep)
    assert rep["route_disagreements"] == 0, qd.format_report(rep)
    assert rep["max_state_delta"] <= qd.MAX_STATE_DELTA, qd.format_report(rep)
    assert rep["max_prob_delta"] <= qd.MAX_PROB_DELTA, qd.format_report(rep)
    assert rep["classify_ok"] and rep["state_ok"] and rep["prob_ok"]


def test_probe_battery_is_deterministic():
    a = qd.probe_embeddings()
    b = qd.probe_embeddings()
    assert np.array_equal(a, b)
    assert a.shape[1] == 4 and a.shape[0] >= 64


def test_fixture_artifact_still_agrees_with_itself(tools_dir):
    """Sanity: the shipped eval fixture, loaded twice, is bit-stable."""
    p = tools_dir / "eval_fixture.fly"
    r1 = flyio.Reservoir.Load(p)
    r2 = flyio.Reservoir.Load(p)
    v = np.full(8, 0.5)
    a, b = r1.Classify(v), r2.Classify(v)
    assert (a.Class, a.Abstained) == (b.Class, b.Abstained)
    assert a.Confidence == b.Confidence


@pytest.fixture()
def tools_dir():
    return Path(__file__).resolve().parent.parent / "eval"


def test_cli_default_twins_pass(capsys):
    rc = qd.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "VERDICT: PASS" in out
