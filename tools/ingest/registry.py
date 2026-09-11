"""Single registry of connectome sources: exact license strings + NC guard.

Every ingest path takes its provenance from SOURCES via guard()/csr_export(),
so the strings in Contract 1 are defined in exactly one place.
"""

from __future__ import annotations

SOURCES = {
    "larval": {
        "source": "larval-connectome",
        "license": "CC-BY-4.0",
        "attribution": "Winding et al. 2023, Science 379:eadd9330",
    },
    "hemibrain": {
        "source": "hemibrain",
        "license": "CC-BY",
        "attribution": "Janelia FlyEM Hemibrain (CC-BY)",
    },
    # Benchmark use only: CC BY-NC 4.0 forbids commercial use. Never ship.
    "flywire": {
        "source": "flywire",
        "license": "CC-BY-NC-4.0",
        "attribution": "FlyWire (FAFB/BANC/MCNS/MANC), CC BY-NC 4.0 - benchmark only",
    },
}

# Sources whose license forbids commercial use; exports are gated.
NON_COMMERCIAL = frozenset({"flywire"})


class NonCommercialError(RuntimeError):
    """A non-commercial source was used without explicit acknowledgement."""


def entry(source):
    """Return the provenance dict for a known source."""
    try:
        return SOURCES[source]
    except KeyError:
        raise KeyError(f"unknown connectome source: {source!r}") from None


def guard(source, allow_noncommercial=False):
    """Return the registry entry, refusing unacknowledged non-commercial sources."""
    provenance = entry(source)
    if source in NON_COMMERCIAL and not allow_noncommercial:
        raise NonCommercialError(
            f"source {source!r} is licensed {provenance['license']} "
            "(non-commercial); pass allow_noncommercial=True (or "
            "--allow-noncommercial) to acknowledge benchmark-only use"
        )
    return provenance


def csr_export(
    source,
    name,
    neurons,
    indptr,
    indices,
    weights,
    allow_noncommercial=False,
):
    """Assemble the shared CSR export dict with guard-enforced provenance."""
    provenance = guard(source, allow_noncommercial=allow_noncommercial)
    return {
        "name": name,
        "neurons": neurons,
        "edges": len(weights),
        "indptr": indptr,
        "indices": indices,
        "weights": weights,
        **provenance,
    }
