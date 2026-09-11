"""Deterministic generator for the eval fixture `.fly` (Contract 1 v1 shape).

DEVIATION (documented): the full spec's quantization/producer pipeline is
owned by 01-reservoir-core and does not exist yet (no Go loader). This
generator packs a hand-built minimal artifact: 8 reservoir lanes, CSR
adjacency with a single self-edge per lane, an int8 input projection with a
lane-one-hot dominant row, and an int8 readout. weight_scale=0.5, so int8
couples lanes strongly enough that dominant patterns win the tanh competition
- which is all the fixture needs to test the harness end to end.

The zstd outer container: this environment cannot `import zstandard`, so the
fixture is stored UNCOMPRESSED (raw FLYRES01 bytes, no zstd frame). flyio.py
still decompresses real zstd frames when the module is present, and readers
must accept both.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

CLASSES = ["alpha", "beta", "gamma"]
EMBED_DIM = 8
NEURONS = 8
STEPS = 4
LANE_FOCUS = 3.0
WEIGHT_SCALE = 0.5

HEADER = {
    "format": "fly-reservoir",
    "version": 1,
    "name": "eval-fixture",
    "neurons": NEURONS,
    "edges": NEURONS,
    "embed_dim": EMBED_DIM,
    "steps": STEPS,
    "classes": CLASSES,
    "weight_scale": WEIGHT_SCALE,
    "source": "synthetic-fixture",
    "license": "CC0-1.0",
    "attribution": "classi-fly eval fixtures",
    "created_utc": "2026-09-11T00:00:00Z",
}


def build_fixture_bytes() -> bytes:
    n, k, d = NEURONS, len(CLASSES), EMBED_DIM
    header_json = json.dumps(HEADER).encode("utf-8")
    out = bytearray()
    out += b"FLYRES01"
    out += struct.pack("<I", len(header_json))
    out += header_json

    # CSR: lane i has one self-edge of weight +1. Lane-coupling for lane i is
    # the readout column projection; tanh saturation at +-1 (via weight_scale)
    # separates dominant lanes.
    indptr = np.arange(n + 1, dtype="<u4")
    indices = np.arange(n, dtype="<u4")
    weights = np.ones(n, dtype=np.int8)
    out += indptr.tobytes() + indices.tobytes() + weights.tobytes()

    out += struct.pack("<B", 1)  # input_mode = matrix
    # Input projection: int8[d, n], column c = lane c gets LANE_FOCUS/0.25.
    # dequant: int8 * in_scale (0.25) => lane i sees its basis coord at 12.0.
    in_w = np.zeros((d, n), dtype=np.int8)
    for c in range(n):
        in_w[c % d, c] = int(round(LANE_FOCUS / 0.25))
    out += in_w.tobytes()
    out += struct.pack("<I", n)           # in_row_len = S
    out += struct.pack("<d", 0.25)        # in_scale
    # Readout: int8[n, k], one-hot per class lane; decays cross-lane via tanh.
    readout = np.zeros((n, k), dtype=np.int8)
    for c in range(min(k, n)):
        readout[c, c] = 8
    out += readout.tobytes()
    out += struct.pack("<d", 0.25)        # readout_scale
    out += struct.pack("<%df" % k, *([0.0] * k))   # bias f32
    thresholds = [0.0] * k                # fixture never abstains
    out += struct.pack("<%df" % k, *thresholds)    # threshold f32
    return bytes(out)


def main() -> None:
    here = Path(__file__).resolve().parent
    target = here / "eval_fixture.fly"
    target.write_bytes(build_fixture_bytes())
    print("wrote %s (%d bytes, uncompressed)" % (target, target.stat().st_size))


if __name__ == "__main__":
    main()
