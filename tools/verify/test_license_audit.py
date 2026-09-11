"""Tests for tools/verify/license_audit.py (leaf 08 Task 3)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import license_audit as la  # noqa: E402
from flybytes import write_fly  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent


def _artifact(name, license_, source, attribution=""):
    return {
        "name": name, "neurons": 2, "embed_dim": 1, "steps": 1,
        "classes": ["a", "b"], "weight_scale": 0.5,
        "indptr": [0, 1, 2], "indices": [0, 1], "weights": [1, 1],
        "source": source, "license": license_, "attribution": attribution,
    }


def test_classify_license_policy():
    assert la.classify_license("CC-BY-NC-4.0") == "BENCHMARK-ONLY"
    assert la.classify_license("cc-by-nc") == "BENCHMARK-ONLY"
    assert la.classify_license("synthetic") == "SHIPPABLE"
    assert la.classify_license("none") == "SHIPPABLE"
    assert la.classify_license("CC0-1.0") == "SHIPPABLE"
    assert la.classify_license("CC-BY-4.0") == "SHIPPABLE"
    assert la.classify_license("CC-BY") == "SHIPPABLE"
    assert la.classify_license("some-new-license") == "UNKNOWN"  # fail-closed


def test_nc_artifact_is_not_shippable(tmp_path):
    p = write_fly(tmp_path / "nc.fly",
                  _artifact("flywire-test", "CC-BY-NC-4.0", "flywire", "FlyWire"))
    header = la.read_fly_header(p)
    assert header["license"] == "CC-BY-NC-4.0"
    rep = la.audit(extra_dirs=[tmp_path])
    row = [a for a in rep["artifacts"] if a["path"] == str(p)][0]
    assert row["verdict"] == "BENCHMARK-ONLY"
    assert rep["ok"] is False


def test_synthetic_artifact_is_shippable(tmp_path):
    p = write_fly(tmp_path / "syn.fly",
                  _artifact("syn-test", "none", "synthetic"))
    rep = la.audit(extra_dirs=[tmp_path])
    row = [a for a in rep["artifacts"] if a["path"] == str(p)][0]
    assert row["verdict"] == "SHIPPABLE"


def test_ccby_requires_attribution_in_header(tmp_path):
    p = write_fly(tmp_path / "by.fly",
                  _artifact("larval-test", "CC-BY-4.0", "larval-connectome",
                            attribution="Winding et al. 2023, Science 379:eadd9330"))
    rep = la.audit(extra_dirs=[tmp_path])
    row = [a for a in rep["artifacts"] if a["path"] == str(p)][0]
    assert row["verdict"] == "SHIPPABLE"
    # Same license but NO attribution: fail-closed, not shippable.
    q = write_fly(tmp_path / "by_noattr.fly",
                  _artifact("larval-noattr", "CC-BY-4.0", "larval-connectome"))
    rep2 = la.audit(extra_dirs=[tmp_path])
    row2 = [a for a in rep2["artifacts"] if a["path"] == str(q)][0]
    assert row2["verdict"] == "UNKNOWN"


def test_notice_cross_check(tmp_path):
    """A CC-BY artifact whose source NOTICE does not cover => finding."""
    p = write_fly(tmp_path / "odd.fly",
                  _artifact("odd-source", "CC-BY-4.0", "some-unlisted-source",
                            attribution="someone"))
    rep = la.audit(extra_dirs=[tmp_path])
    # 'some-unlisted-source' maps to no NOTICE family; unknown sources are not
    # flagged (they are simply not one of the four known families), but the
    # flywire family MUST be covered by the real NOTICE if present.
    assert rep["notice_findings"] == []
    flywire_like = write_fly(tmp_path / "fw.fly",
                             _artifact("fw", "CC-BY-NC-4.0", "flywire"))
    rep2 = la.audit(extra_dirs=[tmp_path])
    row = [a for a in rep2["artifacts"] if a["path"] == str(flywire_like)][0]
    assert row["notice_covers_source"] is True  # real NOTICE mentions FlyWire


def test_notice_gap_is_a_finding(tmp_path, monkeypatch):
    """Point the audit at a NOTICE that omits FlyWire: must flag it."""
    notice = tmp_path / "NOTICE"
    notice.write_text("classi-fly NOTICE\n\nSynthetic reservoirs only.\n",
                      encoding="utf-8")
    write_fly(tmp_path / "fw2.fly", _artifact("fw2", "CC-BY-NC-4.0", "flywire"))
    rep = la.audit(extra_dirs=[tmp_path], notice_path=notice)
    assert any("flywire" in f for f in rep["notice_findings"])
    assert rep["ok"] is False


def test_repo_artifacts_audit_green():
    """The real repo tree: every committed .fly is shippable and NOTICE covers
    the sources. (eval_fixture.fly is synthetic, CC0-1.0.)"""
    rep = la.audit()
    assert rep["artifacts"], "expected at least the eval fixture .fly"
    for a in rep["artifacts"]:
        assert a["verdict"] == "SHIPPABLE", json.dumps(a, indent=2)
        assert a["notice_covers_source"], json.dumps(a, indent=2)
    assert rep["ok"] is True


def test_walk_finds_fixture():
    paths = la.find_fly_files()
    assert any(p.name == "eval_fixture.fly" for p in paths)


def test_cli_green(capsys):
    rc = la.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "VERDICT: PASS" in out
