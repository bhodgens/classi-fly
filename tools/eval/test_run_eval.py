import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import embeddings as emb
import flyio
import run_eval

FIXTURE_FLY = Path(__file__).resolve().parent / "eval_fixture.fly"


def _corpus(n_per=6, ood_text="delta completely novel request"):
    items = []
    for name, words in (("alpha", ["parser", "lexer", "token"]),
                        ("beta", ["sprint", "backlog", "roadmap"]),
                        ("gamma", ["review", "diff", "merge"])):
        for w in words * 2:
            items.append({"text": name + " " + w, "intent": name,
                          "ood": False, "source": "fixture"})
    items.append({"text": ood_text, "intent": "alpha", "ood": True, "source": "fixture"})
    return items


def test_fixture_fly_loads_and_classifies():
    res = flyio.Reservoir.Load(FIXTURE_FLY)
    info = res.Info()
    assert info.Name == "eval-fixture"
    assert info.Neurons == 8 and info.EmbedDim == 8 and info.Classes == ["alpha", "beta", "gamma"]
    vec = emb.deterministic_reference_vector("alpha parser", dim=8)
    r = res.Classify(vec)
    assert r.Class == "alpha" and not r.Abstained
    with pytest.raises(ValueError):
        res.Classify([0.0] * 5)  # dim mismatch is an error (Contract 2)


def test_run_eval_head_keys_and_deterministic_e2e():
    items = _corpus()
    fake = emb.DeterministicFakeEndpoint(dim=8)
    results = run_eval.run_eval(FIXTURE_FLY, items, mode="head",
                                cache_dir=None, client=fake)
    for key in ("P", "C", "E2E", "OOD_R", "routes", "total", "predictions"):
        assert key in results, "missing metric key %r" % key
    assert results["total"] == len(items)
    assert results["routes"] <= results["total"]
    assert 0.0 <= results["P"] <= 1.0 and 0.0 <= results["C"] <= 1.0
    assert len(results["predictions"]) == len(items)
    # E2E recomputed INDEPENDENTLY from the counts must equal the reported value.
    expected_e2e = (results["gate_correct"] + 0.868 * results["abstained"]) / results["total"]
    assert results["E2E"] == pytest.approx(expected_e2e, abs=1e-12)
    # OOD cases are scored: every prediction row appears, OOD row included.
    assert sum(1 for p in results["predictions"] if p["ood"]) == 1
    assert results["ood_total"] == 1
    # OOD_R derives from that OOD row's abstention.
    ood_row = [p for p in results["predictions"] if p["ood"]][0]
    assert results["OOD_R"] == (1.0 if ood_row["abstained"] else 0.0)
    # Determinism: same inputs -> identical metrics.
    fake2 = emb.DeterministicFakeEndpoint(dim=8)
    again = run_eval.run_eval(FIXTURE_FLY, items, mode="head", client=fake2)
    assert again["E2E"] == results["E2E"]
    assert again["routes"] == results["routes"]
    # Predictions carry text hashes, never raw text.
    blob = json.dumps(results)
    assert "parser" not in blob


def test_run_eval_judge_and_all():
    items = _corpus()
    fake = emb.DeterministicFakeEndpoint(dim=8)
    judge = run_eval.run_eval(FIXTURE_FLY, items, mode="judge", client=fake)
    for key in ("P", "C", "E2E", "OOD_R", "routes", "total", "predictions"):
        assert key in judge
    # Judge gates are stricter: judge routes <= head routes on the same data.
    head = run_eval.run_eval(FIXTURE_FLY, items, mode="head", client=fake)
    assert judge["routes"] <= head["routes"]
    both = run_eval.run_eval(FIXTURE_FLY, items, mode="all", client=fake)
    assert set(both["modes"]) == {"head", "judge"}
    assert both["chain_baseline"] == 0.868
    for m in ("head", "judge"):
        for key in ("P", "C", "E2E", "OOD_R", "routes", "total"):
            assert key in both["modes"][m]


def test_fold_assignment_stratified_and_fixed():
    labels = ["a"] * 10 + ["b"] * 5
    a1, used, note = run_eval.fold_indices(labels, 5, seed=42)
    assert used == 5 and note is None
    assert len(a1) == len(labels)
    assert all(0 <= f < 5 for f in a1)
    # Per label, folds are round-robin over the shuffled indices: balanced.
    for lbl in ("a", "b"):
        counts = [0] * 5
        for i, f in enumerate(a1):
            if labels[i] == lbl:
                counts[f] += 1
        assert max(counts) - min(counts) <= 1
    a2, _, _ = run_eval.fold_indices(labels, 5, seed=42)
    assert a1 == a2  # seed 42 -> identical assignment
    # Stratification: no fold is single-label while both labels exist.
    fold_labels = {}
    for i, f in enumerate(a1):
        fold_labels.setdefault(f, set()).add(labels[i])
    small = run_eval.fold_indices(["a", "b"], 5, seed=42)
    assert small[1] == 2 and "leave-one-out" in small[2]


def test_small_corpus_falls_back_to_leave_one_out():
    items = [{"text": "alpha parser", "intent": "alpha", "ood": False},
             {"text": "beta roadmap", "intent": "beta", "ood": False},
             {"text": "gamma diff", "intent": "gamma", "ood": True}]
    fake = emb.DeterministicFakeEndpoint(dim=8)
    results = run_eval.run_eval(FIXTURE_FLY, items, mode="head", client=fake)
    assert results["folds_used"] == 3
    assert "leave-one-out" in (results["folds_note"] or "")
    assert results["total"] == 3


def test_write_results_discloses_routes(tmp_path):
    items = _corpus()
    fake = emb.DeterministicFakeEndpoint(dim=8)
    results = run_eval.run_eval(FIXTURE_FLY, items, mode="all", client=fake)
    out = tmp_path / "results.json"
    run_eval.write_results(out, results)
    assert out.exists()
    report = tmp_path / "report.md"
    assert report.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    for m in ("head", "judge"):
        assert "routes" in data["modes"][m] and "total" in data["modes"][m]
    table = report.read_text(encoding="utf-8")
    assert "routes/total" in table
    assert "routes" in table
    for m in ("head", "judge"):
        assert "%d routes" % data["modes"][m]["routes"] in table  # route count named


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        run_eval.run_eval(FIXTURE_FLY, [{"text": "x", "intent": "alpha"}], mode="nope")
