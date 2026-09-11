"""License-guard tests: FlyWire must never silently produce a shippable artifact."""

import pytest

from flywire import build_export
from registry import SOURCES, NonCommercialError, guard


def test_flywire_requires_ack():
    with pytest.raises(NonCommercialError):
        guard("flywire")


def test_flywire_ack_returns_entry():
    entry = guard("flywire", allow_noncommercial=True)
    assert entry["source"] == "flywire"
    assert entry["license"] == "CC-BY-NC-4.0"


def test_flywire_build_requires_ack():
    rows = [[0, 2], [0, 0]]
    with pytest.raises(NonCommercialError):
        build_export(rows)
    out = build_export(rows, allow_noncommercial=True)
    assert out["name"] == "flywire"
    assert out["source"] == "flywire"
    assert out["license"] == "CC-BY-NC-4.0"
    assert out["indptr"] == [0, 1, 1]
    assert out["indices"] == [1]
    assert out["weights"] == [2]


def test_commercial_sources_unblocked():
    assert guard("larval")["license"] == "CC-BY-4.0"
    assert guard("hemibrain")["license"] == "CC-BY"


def test_registry_strings_exact():
    assert SOURCES["larval"] == {
        "source": "larval-connectome",
        "license": "CC-BY-4.0",
        "attribution": "Winding et al. 2023, Science 379:eadd9330",
    }
    assert SOURCES["hemibrain"]["attribution"] == "Janelia FlyEM Hemibrain (CC-BY)"


def test_flywire_license_exact():
    assert SOURCES["flywire"]["license"] == "CC-BY-NC-4.0"


def test_unknown_source_rejected():
    with pytest.raises(KeyError):
        guard("fafb")
