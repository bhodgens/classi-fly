"""Tests for the hemibrain neuPrint response -> CSR transform (offline, no network)."""

import pytest

from hemibrain import main, response_to_csr

RECORDED = {
    "data": {
        "connectivity": {
            "neuron": [101, 102, 103],
            "partners": [[103, 102], [101], []],
            "weights": [[-2, 4], [1], []],
        }
    }
}


def test_transform_shape():
    out = response_to_csr(RECORDED)
    assert out["neurons"] == 3
    assert out["edges"] == 3
    # columns are sorted within each row regardless of partner input order
    assert out["indptr"] == [0, 2, 3, 3]
    assert out["indices"] == [1, 2, 0]
    assert out["weights"] == [4, -2, 1]


def test_hemibrain_provenance_exact():
    out = response_to_csr(RECORDED)
    assert out["source"] == "hemibrain"
    assert out["license"] == "CC-BY"
    assert out["attribution"] == "Janelia FlyEM Hemibrain (CC-BY)"


def test_unwrapped_response_equivalent():
    bare = {"connectivity": RECORDED["data"]["connectivity"]}
    assert response_to_csr(bare) == response_to_csr(RECORDED)


def test_partner_outside_universe_raises():
    bad = {
        "data": {
            "connectivity": {
                "neuron": [1],
                "partners": [[99]],
                "weights": [[3]],
            }
        }
    }
    with pytest.raises(ValueError):
        response_to_csr(bad)


def test_length_mismatch_raises():
    bad = {
        "data": {
            "connectivity": {
                "neuron": [1, 2],
                "partners": [[2]],
                "weights": [[1], [2]],
            }
        }
    }
    with pytest.raises(ValueError):
        response_to_csr(bad)


def test_zero_weight_partner_dropped():
    resp = {
        "data": {
            "connectivity": {
                "neuron": [1, 2],
                "partners": [[2], []],
                "weights": [[0], []],
            }
        }
    }
    out = response_to_csr(resp)
    assert out["edges"] == 0
    assert out["indptr"] == [0, 0, 0]
    assert out["indices"] == []
    assert out["weights"] == []


def test_cli_dry_run_no_network_no_file(tmp_path):
    dst = tmp_path / "adj.json"
    rc = main(
        [
            "--token",
            "recorded-fixture-token",
            "--region",
            "MB",
            "--out",
            str(dst),
        ]
    )
    assert rc == 0
    assert not dst.exists()
