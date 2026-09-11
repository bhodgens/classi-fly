"""Deterministic tiny fixtures shared by the offline trainer tests.

Everything here is hardcoded (no RNG) so fixtures are byte-stable across
runs and Python versions. Weight and input-projection values are exact
integers so int8 quantization through the .fly fixture writer is lossless.
"""

NEURONS = 6
EDGES = 9
EMBED_DIM = 4
STEPS = 5
DECAY = 0.8

INDPTR = [0, 2, 3, 5, 6, 8, 9]
INDICES = [1, 3, 2, 0, 4, 5, 1, 2, 0]
WEIGHTS = [4, -2, 3, 1, -3, 2, -1, 4, -2]
WEIGHT_SCALE = 0.05

IN_W = [
    5, -3, 2, 1, -4, 0,
    2, 4, -1, 3, 1, -2,
    -2, 1, 3, -5, 2, 4,
    1, -2, -3, 2, 5, -1,
]
IN_SCALE = 0.01

CLASSES = ["alpha", "beta"]


def tiny_artifact():
    """Return a tiny adjacency-dict artifact (the dict form states() accepts)."""
    return {
        "name": "tiny",
        "neurons": NEURONS,
        "edges": EDGES,
        "embed_dim": EMBED_DIM,
        "steps": STEPS,
        "decay": DECAY,
        "classes": list(CLASSES),
        "indptr": list(INDPTR),
        "indices": list(INDICES),
        "weights": list(WEIGHTS),
        "weight_scale": WEIGHT_SCALE,
        "in_w": list(IN_W),
        "in_scale": IN_SCALE,
        "source": "synthetic",
        "license": "none",
        "attribution": "n/a",
    }
