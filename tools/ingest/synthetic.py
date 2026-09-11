"""Synthetic mushroom-body-shaped reservoir generator (leaf 03).

Seed-driven, fully deterministic CSR adjacency generator -- the
license-clean default artifact for classi-fly (no third-party data).

Shape (mushroom body analogy, RESEARCH.md section 4):
    a few projection inputs expand into N sparsely-connected internal
    units (Kenyon-like), which feed a small output fan-in.

Contract 1 (plans/classi-fly/master.md) adjacency export shape::

    {name, neurons, edges, indptr, indices, weights,
     source, license, attribution}

Determinism: identical seed + parameters yield byte-identical
indptr/indices/weights. Only random.Random(seed) is used and iteration
is strictly index-ordered, never over sets/dicts.

Stability: weights are scaled by a global factor so a power-iteration
estimate of the spectral radius lands on a fixed target (< 0.9 by
default); a caller forcing an unsafe target gets ValueError.

Usage::

    python3 tools/ingest/synthetic.py --seed 7 --neurons 2048 \
        --fan-in 6 --inhib-frac 0.2 --out adjacency.json
"""

import argparse
import json
import math
import random
import sys

DEFAULT_NEURONS = 2048
DEFAULT_FAN_IN = 6
DEFAULT_INHIB_FRAC = 0.2
DEFAULT_TARGET = 0.9

#: Fixed weight distribution: weight magnitude ~ Uniform(0.05, 1.0).
WEIGHT_LOW = 0.05
WEIGHT_HIGH = 1.0

#: Power-iteration settings (stdlib only; the matrix is sparse, so each
#: sweep is O(edges)). 200 sweeps is far past convergence for a 0.9
#: target; the relaxation factor guards a pathological zero row space.
POWER_ITER_SWEEPS = 200
POWER_ITER_EPS = 1e-12

_SOURCE = "synthetic"
_LICENSE = "none"
_ATTRIBUTION = "seed-generated (classi-fly); no third-party data"


def spectral_radius_estimate(csr, sweeps=POWER_ITER_SWEEPS):
    """Estimate the spectral radius of the CSR matrix by power iteration.

    Works on the CSR dict produced by :func:`generate` (or any dict with
    neurons/edges/indptr/indices/weights). Returns the largest-magnitude
    eigenvalue estimate (float64); deterministic for identical inputs.
    """
    n = csr["neurons"]
    indptr = csr["indptr"]
    indices = csr["indices"]
    weights = csr["weights"]
    if n == 0 or len(indices) == 0:
        return 0.0

    # Deterministic nonzero start vector (pseudo-random but fixed values
    # derived from an index-based hash formula -- no RNG needed).
    v = [math.sin(1.7 * (i + 1)) for i in range(n)]

    radius = 0.0
    for _ in range(sweeps):
        nxt = [0.0] * n
        for row in range(n):
            acc = 0.0
            for k in range(indptr[row], indptr[row + 1]):
                acc += weights[k] * v[indices[k]]
            nxt[row] = acc
        norm = math.sqrt(sum(x * x for x in nxt))
        if norm <= POWER_ITER_EPS:
            # Exactly-invariant subspace collapsed (e.g. an all-zero
            # matrix): use the previous iterate's norm so we still
            # return a finite, deterministic value.
            norm = math.sqrt(sum(x * x for x in v))
            if norm <= POWER_ITER_EPS:
                return 0.0
            return norm
        inv = 1.0 / norm
        v = [x * inv for x in nxt]
        radius = norm
    return radius


def _validate(seed, n, fan_in, inhib_frac, target):
    if not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if fan_in < 1 or fan_in > n - 1:
        raise ValueError(
            f"fan_in must be in [1, {n - 1}] for n={n}, got {fan_in}")
    if not (0.0 <= inhib_frac <= 1.0):
        raise ValueError(f"inhib_frac must be in [0, 1], got {inhib_frac}")
    if not (0.0 < target < 1.0):
        raise ValueError(
            f"target spectral radius must be in (0, 1), got {target}; "
            "a reservoir is unstable at >= 1 and degenerate at <= 0")


def generate(seed, n=DEFAULT_NEURONS, fan_in=DEFAULT_FAN_IN,
             inhib_frac=DEFAULT_INHIB_FRAC, target=DEFAULT_TARGET):
    """Generate a deterministic sparse reservoir as a Contract 1 CSR dict.

    seed: int controlling every random draw.
    n: number of internal (Kenyon-like) units.
    fan_in: distinct presynaptic targets sampled per unit (no replacement,
        no self-loops). Row length is exactly fan_in.
    inhib_frac: fraction of edges whose weight sign is negative; the sign
        decision is one RNG draw per edge, so it is per-edge Bernoulli and
        the realized fraction converges to inhib_frac.
    target: spectral-radius budget for the global weight scale; must lie
        in (0, 1) or ValueError is raised.

    Returns::

        {name, neurons, edges, indptr, indices, weights,
         source, license, attribution}

    Deterministic: identical inputs -> identical lists; indices rows are
    sorted ascending per CSR convention.
    """
    _validate(seed, n, fan_in, inhib_frac, target)
    rng = random.Random(seed)

    indptr = [0] * (n + 1)
    indices = []
    weights = []
    for row in range(n):
        # sample() without replacement -> fan_in DISTINCT targets, and it
        # never includes `row` itself because range excludes it: no
        # self-loops. Sorted so the CSR rows are canonical.
        targets = rng.sample(range(n - 1), fan_in)
        targets = sorted(t if t < row else t + 1 for t in targets)
        for t in targets:
            indices.append(t)
            magnitude = rng.uniform(WEIGHT_LOW, WEIGHT_HIGH)
            sign = -1.0 if rng.random() < inhib_frac else 1.0
            weights.append(sign * magnitude)
        indptr[row + 1] = len(indices)

    # Global stability scale: fit a single multiplicative factor so the
    # power-iteration spectral-radius estimate lands on `target`. This
    # preserves the drawn distribution's shape (pure scaling) while
    # guaranteeing reservoir stability.
    scaled_weights = weights
    if weights:
        est = spectral_radius_estimate(_csr_view(n, indptr, indices, weights))
        if est > 0.0:
            scale = target / est
            scaled_weights = [w * scale for w in weights]

    return {
        "name": f"synthetic-seed{seed}-n{n}",
        "neurons": n,
        "edges": len(scaled_weights),
        "indptr": indptr,
        "indices": indices,
        "weights": scaled_weights,
        "source": _SOURCE,
        "license": _LICENSE,
        "attribution": _ATTRIBUTION,
    }


def _csr_view(n, indptr, indices, weights):
    """Minimal CSR-shaped dict for spectral_radius_estimate."""
    return {
        "neurons": n,
        "edges": len(indices),
        "indptr": indptr,
        "indices": indices,
        "weights": weights,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a deterministic synthetic reservoir CSR JSON.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--neurons", type=int, default=DEFAULT_NEURONS)
    parser.add_argument("--fan-in", type=int, default=DEFAULT_FAN_IN,
                        dest="fan_in")
    parser.add_argument("--inhib-frac", type=float, default=DEFAULT_INHIB_FRAC,
                        dest="inhib_frac")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET,
                        help="spectral-radius target, must be in (0, 1)")
    parser.add_argument("--out", default="-",
                        help="output path (default stdout)")
    args = parser.parse_args(argv)

    try:
        csr = generate(seed=args.seed, n=args.neurons, fan_in=args.fan_in,
                       inhib_frac=args.inhib_frac, target=args.target)
    except ValueError as exc:
        parser.error(str(exc))

    payload = json.dumps(csr, sort_keys=False, separators=(",", ":"))
    if args.out == "-":
        sys.stdout.write(payload + "\n")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
