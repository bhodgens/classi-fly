"""Tests for the larval connectome CSV -> CSR transform (offline, fixtures only)."""

import json

import pytest

from larval import csv_to_csr, main
from registry import SOURCES

LABELED_CSV = (
    ",alpha,beta,gamma\n"
    "alpha,0,5,-2\n"
    "beta,3,0,0\n"
    "gamma,0,-1,7\n"
)


def test_basic_shape():
    out = csv_to_csr(LABELED_CSV)
    assert out["neurons"] == 3
    assert out["edges"] == 5
    assert out["indptr"] == [0, 2, 3, 5]
    assert out["indices"] == [1, 2, 0, 1, 2]
    assert out["weights"] == [5, -2, 3, -1, 7]


def test_indptr_contract():
    out = csv_to_csr(LABELED_CSV)
    assert out["indptr"][0] == 0
    assert len(out["indptr"]) == out["neurons"] + 1
    assert len(out["indices"]) == out["edges"]
    assert len(out["weights"]) == out["edges"]


def test_weights_carry_sign_as_ints():
    out = csv_to_csr(LABELED_CSV)
    assert -2 in out["weights"]
    assert -1 in out["weights"]
    assert all(isinstance(w, int) for w in out["weights"])


def test_bare_matrix_without_header_or_labels():
    out = csv_to_csr("0,4\n-6,0\n")
    assert out["neurons"] == 2
    assert out["indptr"] == [0, 1, 2]
    assert out["weights"] == [4, -6]


def test_integral_floats_become_ints():
    out = csv_to_csr("0,2.0\n0,0\n")
    assert out["weights"] == [2]
    assert isinstance(out["weights"][0], int)


def test_provenance_strings_exact():
    out = csv_to_csr(LABELED_CSV, name="larval-v1")
    assert out["source"] == "larval-connectome"
    assert out["license"] == "CC-BY-4.0"
    assert out["attribution"] == "Winding et al. 2023, Science 379:eadd9330"
    assert out["name"] == "larval-v1"
    assert SOURCES["larval"] == {
        "source": "larval-connectome",
        "license": "CC-BY-4.0",
        "attribution": "Winding et al. 2023, Science 379:eadd9330",
    }


def test_malformed_numeric_row_raises():
    with pytest.raises(ValueError):
        csv_to_csr("0,1,2\n1,x,2\n")


def test_non_square_matrix_raises():
    with pytest.raises(ValueError):
        csv_to_csr("0,1,2\n3,0,4\n")


def test_cli_round_trip(tmp_path):
    src = tmp_path / "in.csv"
    dst = tmp_path / "adj.json"
    src.write_text(LABELED_CSV, encoding="utf-8")
    rc = main(["--csv", str(src), "--out", str(dst)])
    assert rc == 0
    written = json.loads(dst.read_text(encoding="utf-8"))
    assert written == csv_to_csr(LABELED_CSV)
