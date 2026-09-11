"""FlyWire ingest: benchmark-only CSR export, guarded by the NC license gate.

FlyWire data is CC BY-NC 4.0: commercial use is forbidden, so exports are
blocked unless explicitly acknowledged (never shippable).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

import registry
from registry import NonCommercialError  # re-export for CLI consumers


def build_export(matrix_rows, name="flywire", allow_noncommercial=False):
    """CSR export from an iterable of square-matrix rows (FlyWire, NC-gated)."""
    indptr = [0]
    indices, weights = [], []
    ncols = None
    nrows = 0

    for raw in matrix_rows:
        if ncols is None:
            ncols = len(raw)
        elif len(raw) != ncols:
            raise ValueError("matrix is not square")
        for col_no, cell in enumerate(raw):
            value = int(cell)
            if value != 0:
                indices.append(col_no)
                weights.append(value)
        indptr.append(len(indices))
        nrows += 1

    if ncols is None:
        raise ValueError("empty matrix")
    if nrows != ncols:
        raise ValueError("matrix is not square")

    return registry.csr_export(
        "flywire",
        name,
        ncols,
        indptr,
        indices,
        weights,
        allow_noncommercial=allow_noncommercial,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build a FlyWire CSR adjacency export (CC-BY-NC-4.0, benchmark "
            "only). Requires --allow-noncommercial."
        )
    )
    parser.add_argument("--csv", required=True, help="square connectivity matrix CSV")
    parser.add_argument("--out", required=True, help="path for the adjacency JSON export")
    parser.add_argument("--name", default="flywire", help="export name")
    parser.add_argument(
        "--allow-noncommercial",
        action="store_true",
        help="acknowledge the non-commercial license (benchmark use only)",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.csv, encoding="utf-8", newline="") as fh:
            export = build_export(
                list(csv.reader(fh)),
                name=args.name,
                allow_noncommercial=args.allow_noncommercial,
            )
    except (OSError, ValueError, NonCommercialError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with open(args.out, "w", encoding="utf-8") as out:
        json.dump(export, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
