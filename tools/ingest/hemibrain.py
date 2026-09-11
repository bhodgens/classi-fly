"""Hemibrain ingest via neuPrint: recorded/queried response -> CSR adjacency export.

Provenance: Janelia FlyEM Hemibrain (CC-BY). Network access only in --live
mode; the transform itself is pure and tested offline against fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys

import registry

DEFAULT_ENDPOINT = "https://neuprint.janelia.org"


def response_to_csr(response, name="hemibrain-MB-v1", allow_noncommercial=False):
    """Transform a neuPrint connectivity response into the shared CSR export."""
    conn = response.get("data", response).get("connectivity", {})
    neuron_ids = conn.get("neuron", [])
    partners = conn.get("partners", [])
    weights = conn.get("weights", [])

    if not (len(neuron_ids) == len(partners) == len(weights)):
        raise ValueError("connectivity arrays must have equal length")

    order = {body_id: idx for idx, body_id in enumerate(neuron_ids)}
    indptr = [0]
    indices, wvals = [], []

    for row, (targets, wrow) in enumerate(zip(partners, weights)):
        pairs = []
        for target, weight in zip(targets, wrow):
            if target not in order:
                raise ValueError(
                    f"partner {target} at row {row} is not in the neuron list"
                )
            weight = int(weight)
            if weight != 0:
                pairs.append((order[target], weight))
        # CSR convention: column indices sorted within each row.
        for col, weight in sorted(pairs):
            indices.append(col)
            wvals.append(weight)
        indptr.append(len(indices))

    return registry.csr_export(
        "hemibrain",
        name,
        len(neuron_ids),
        indptr,
        indices,
        wvals,
        allow_noncommercial=allow_noncommercial,
    )


def fetch_connectivity(token, region, endpoint=DEFAULT_ENDPOINT):
    """Live neuPrint query. Only called when --live is passed."""
    import urllib.request  # local import: never loaded by tests

    query = (
        "MATCH (n:Neuron)<-[:ConnectsTo]-(m:Neuron) "
        f"WHERE n.bodyId IN [{region}] AND m.bodyId IN [{region}] "
        "RETURN n.bodyId, m.bodyId, m.type, n.type, "
        "e.weight, e.roiInfo"
    )
    body = json.dumps({"cypher": query, "dataset": "hemibrain:v1.2.1"}).encode()
    req = urllib.request.Request(
        f"{endpoint}/api/custom/neuprintquery",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Query neuPrint for hemibrain connectivity and emit a CSR "
            "adjacency export (CC-BY)."
        )
    )
    parser.add_argument("--token", required=True, help="neuPrint API token")
    parser.add_argument("--region", required=True, help="neuron body IDs or query filter")
    parser.add_argument("--out", required=True, help="path for the adjacency JSON export")
    parser.add_argument("--name", default="hemibrain-MB-v1", help="export name")
    parser.add_argument(
        "--live",
        action="store_true",
        help="issue the real neuPrint query (offline dry-run otherwise)",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="neuPrint API base URL",
    )
    args = parser.parse_args(argv)

    if not args.live:
        # Dry run: no network, no file. Real usage requires --live.
        return 0

    try:
        response = fetch_connectivity(args.token, args.region, args.endpoint)
        export = response_to_csr(response, name=args.name)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with open(args.out, "w", encoding="utf-8") as out:
        json.dump(export, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
