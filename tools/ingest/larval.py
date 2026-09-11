"""Larval connectome ingest: signed connectivity CSV -> CSR adjacency export.

Provenance: Winding et al. 2023, Science 379:eadd9330 (CC-BY-4.0).
The CSV is consumed row-by-row; no dense matrix is materialised.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys

import registry


def _is_number(cell):
    try:
        float(cell)
    except ValueError:
        return False
    return True


def _as_int_weight(cell, row_no, col_no):
    value = float(cell)
    if not value.is_integer():
        raise ValueError(
            f"non-integer weight at row {row_no}, column {col_no}: {cell!r}"
        )
    return int(value)


def csv_to_csr_stream(rows, name="larval-v1"):
    """Build the CSR export from an iterable of CSV row lists (streaming)."""
    indptr = [0]
    indices, weights = [], []
    ncols = None
    labeled = False
    nrows = 0

    for row_no, raw in enumerate(rows, start=1):
        if not any(field.strip() for field in raw):
            continue
        if ncols is None:
            ncols = len(raw)
        elif len(raw) != ncols:
            raise ValueError(
                f"row {row_no} has {len(raw)} fields, expected {ncols}"
            )
        if not nrows:
            # A leading header row (e.g. ",alpha,beta,gamma") has non-numeric
            # label cells in the value region; skip it entirely.
            if not all(_is_number(cell) for cell in raw[1:]):
                continue
            labeled = not _is_number(raw[0])
        start_col = 1 if labeled else 0
        for col_no, cell in enumerate(raw[start_col:], start=start_col):
            value = _as_int_weight(cell, row_no, col_no)
            if value == 0:
                continue
            indices.append(col_no - start_col)
            weights.append(value)
        indptr.append(len(indices))
        nrows += 1

    if ncols is None:
        raise ValueError("empty CSV")
    if labeled:
        neurons = ncols - 1
        if nrows != neurons:
            raise ValueError("matrix is not square")
    else:
        neurons = ncols
        if nrows != neurons:
            raise ValueError("matrix is not square")

    return registry.csr_export("larval", name, neurons, indptr, indices, weights)


def csv_to_csr(text, name="larval-v1"):
    """Pure-function entry point: parse a signed connectivity CSV from text."""
    return csv_to_csr_stream(csv.reader(io.StringIO(text)), name=name)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Convert the larval signed connectivity CSV to a CSR adjacency "
            "export (CC-BY-4.0)."
        )
    )
    parser.add_argument("--csv", required=True, help="path to the signed connectivity CSV")
    parser.add_argument("--out", required=True, help="path for the adjacency JSON export")
    parser.add_argument("--name", default="larval-v1", help="export name")
    args = parser.parse_args(argv)

    try:
        with open(args.csv, encoding="utf-8", newline="") as fh:
            export = csv_to_csr_stream(csv.reader(fh), name=args.name)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with open(args.out, "w", encoding="utf-8") as out:
        json.dump(export, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
