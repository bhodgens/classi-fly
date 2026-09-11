"""Tests for tools/verify/recompute.py (leaf 08 Task 1).

The fixture here is HAND-BUILT with known counts so the expected P/C/E2E/OOD_R
can be computed by hand, independently of any harness code.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import recompute  # noqa: E402


def _hand_rows():
    """8 cases, hand-checked:
    - routes: rows 0,1,2,3,4 -> 5 routes; abstained: rows 5,6,7 -> 3
    - routed_correct: rows 0,1,2 correct; row 3 wrong; row 4 wrong -> 3
    - OOD rows: 5 (abstained) and 6 (abstained) and 7? no: 7 is not OOD.
      Let's say row 5 OOD abstain, row 6 OOD abstain, row 7 in-dist abstain.
    Hand metrics (chain credit 0.868):
      P   = 3/5 = 0.6
      C   = 5/8 = 0.625
      E2E = (3 + 0.868*3)/8 = 5.604/8 = 0.7005
      OOD_R = 2/2 = 1.0
    """
    def row(i, pred, gold, abstained, ood=False):
        return {"pred": pred, "gold": gold, "abstained": abstained, "ood": ood,
                "conf": 0.0 if abstained else 0.9, "fold": 0,
                "text_sha256": "h%d" % i}
    return [
        row(0, "a", "a", False), row(1, "b", "b", False), row(2, "c", "c", False),
        row(3, "a", "b", False),                       # routed, wrong
        row(4, "c", "a", False),                       # routed, wrong
        row(5, "a", "a", True, ood=True),              # OOD abstain
        row(6, "b", "b", True, ood=True),              # OOD abstain
        row(7, "c", "c", True, ood=False),             # in-dist abstain
    ]


HAND = {
    "P": 3 / 5,
    "C": 5 / 8,
    "E2E": (3 + 0.868 * 3) / 8,
    "OOD_R": 1.0,
    "routes": 5,
    "total": 8,
}


def test_recompute_matches_hand_calculation():
    got = recompute.recompute(_hand_rows())
    for k in ("P", "C", "E2E", "OOD_R"):
        assert got[k] == pytest.approx(HAND[k], abs=1e-12), k
    assert got["routes"] == HAND["routes"]
    assert got["total"] == HAND["total"]
    assert got["ood_total"] == 2 and got["ood_abstained"] == 2
    assert got["abstained"] == 3 and got["gate_correct"] == 3


def test_e2e_is_deterministic_not_sampled():
    """The deterministic formula must hold EXACTLY, and re-running with the
    same rows yields bit-identical E2E (a sampled draw would not)."""
    rows = _hand_rows()
    a = recompute.recompute(rows, chain_baseline=0.868)
    b = recompute.recompute(list(reversed(rows)), chain_baseline=0.868)
    assert a["E2E"] == b["E2E"]
    assert a["E2E"] == pytest.approx((3 + 0.868 * 3) / 8, abs=1e-15)


def test_recompute_results_agrees_on_a_honest_fixture():
    """A results.json whose headline matches its per-case data: full agreement."""
    rows = _hand_rows()
    results = {
        "modes": {
            "head": {"P": HAND["P"], "C": HAND["C"], "E2E": HAND["E2E"],
                     "OOD_R": HAND["OOD_R"], "routes": 5, "total": 8,
                     "predictions": rows},
        },
        "chain_baseline": 0.868,
    }
    findings = recompute.recompute_results(results)
    assert findings, "expected per-metric findings"
    assert all(f["agree"] for f in findings), json.dumps(findings, indent=2)


def test_recompute_flags_a_cooked_headline():
    """THE adversarial case: a results.json whose E2E is inflated beyond what
    its own per-case rows support. Must be flagged as a disagreement."""
    rows = _hand_rows()
    results = {
        "modes": {
            "head": {"P": 0.99, "C": HAND["C"], "E2E": 0.99,  # cooked E2E + P
                     "OOD_R": HAND["OOD_R"], "routes": 5, "total": 8,
                     "predictions": rows},
        },
        "chain_baseline": 0.868,
    }
    findings = recompute.recompute_results(results)
    bad = {f["metric"] for f in findings if not f["agree"]}
    assert "E2E" in bad, "cooked E2E not flagged"
    assert "P" in bad, "cooked P not flagged"
    assert "C" not in bad and "OOD_R" not in bad


def test_recompute_flags_route_count_undisclosure():
    """A headline that hides the route count (routes field lying about the
    per-case data) is flagged - route-count disclosure is mandatory."""
    rows = _hand_rows()
    results = {
        "modes": {
            "head": {"P": HAND["P"], "C": HAND["C"], "E2E": HAND["E2E"],
                     "OOD_R": HAND["OOD_R"], "routes": 8,  # lie: 3 are abstains
                     "total": 8, "predictions": rows},
        },
        "chain_baseline": 0.868,
    }
    findings = recompute.recompute_results(results)
    bad = [f for f in findings if f["metric"] == "routes" and not f["agree"]]
    assert bad and bad[0]["recomputed"] == 5 and bad[0]["harness"] == 8


def test_margin_label_calls_subonecase_noise():
    # 0.8735 vs floor 0.868 on 2 routes: (0.0055)*2 = 0.011 cases -> noise.
    label = recompute.margin_label(2, 0.8735)
    assert "SUB-ONE-CASE MARGIN" in label and "noise" in label
    # A real margin on many routes is not noise.
    assert "noise" not in recompute.margin_label(48, 0.95)


def test_verdict_lines_use_independent_bars():
    results = {
        "modes": {
            "judge": {"predictions": _hand_rows(), "total": 8},
        },
        "chain_baseline": 0.868,
    }
    lines = recompute.verdict_lines(results)
    text = "\n".join(lines)
    assert "judge" in text
    assert "87.35" in text and "86.80" in text  # both bars named
    assert "routes/total: 5/8" in text          # route count disclosed


def test_cli_on_synthetic_results_fixture(tmp_path, capsys):
    fixture = Path(__file__).resolve().parent / "testdata" / "synthetic_results.json"
    rc = recompute.main([str(fixture)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "all metrics agree to 1e-9" in out
    assert "routes/total" in out


def test_cli_fails_on_tampered_results(tmp_path, capsys):
    fixture = json.loads(
        (Path(__file__).resolve().parent / "testdata" / "synthetic_results.json")
        .read_text(encoding="utf-8"))
    fixture["modes"]["judge"]["E2E"] = 0.999  # tamper
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(fixture), encoding="utf-8")
    rc = recompute.main([str(p)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "DISAGREEMENTS" in out
