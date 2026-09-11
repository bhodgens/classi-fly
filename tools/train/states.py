"""Reservoir state extraction for offline readout training (leaf 04 Task 1).

Pure-Python reimplementation of the Go forward pass from
``01-reservoir-core/01-sparse-core.md``:

    s = tanh(decay*s + W*s + win*x),  s0 = 0, run ``steps`` times

with int8 dequantization (``value = int8 * scale``). ``states`` is pure and
deterministic: no RNG, no I/O, no globals.

numpy is used when importable (vectorized SpMV is only a speedup; the
numerics below are identical in both paths because each output element is a
plain float64 sum over the row in index order). Without numpy the same
computation runs on stdlib lists - NEVER install anything.

Input sources:
- an adjacency dict: {neurons, embed_dim, steps, decay, classes,
  indptr, indices, weights, weight_scale, in_w, in_scale} (weights/in_w are
  int8 values, as produced by tools/ingest or the .fly header contract);
- a .fly file path (Contract 1 container: magic "FLYRES01", JSON header,
  packed little-endian arrays; zstd payload decompressed when a codec is
  importable, raw payload auto-detected otherwise).

The .fly container never carries a raw ``win`` array: input_mode is
``seed`` (the fixture/CLI side expands it) or ``matrix`` (in_w[D*N] +
in_scale, read here). Both config forms normalize to the same dict, and the
same embedding always produces the same state vector.
"""

import json
import struct
from pathlib import Path

try:
    import numpy as _np
except ImportError:  # stdlib fallback; never install anything
    _np = None

MAGIC = b"FLYRES01"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
# Leak term used when neither the artifact nor the caller supplies one. The
# 01 leaf documents decay as a value in (0, 1); 0.8 keeps the recurrence
# stable for any normalized reservoir.
DEFAULT_DECAY = 0.8


def _zstd_decompress(data):
    from compression import zstd as _z  # Python 3.14+ stdlib

    return _z.decompress(data)


def _has_zstd_frame(data):
    return len(data) >= 4 and data[:4] == _ZSTD_MAGIC


def _dequant_int8(raw, scale):
    """int8 bytes -> float list/tuple, value = int8 * scale."""
    if _np is not None:
        return _np.frombuffer(raw, dtype=_np.int8).astype(_np.float64) * scale
    return [v * scale for v in struct.unpack("<%db" % len(raw), raw)]


def _int8_bytes(values):
    """list of int8 values -> packed bytes (accepts negatives)."""
    return struct.pack("<%db" % len(values), *values)


def _read_fly(path, decay=None):
    """Parse a Contract 1 .fly file into the states() config dict."""
    blob = Path(path).read_bytes()
    if blob[:8] != MAGIC:
        raise ValueError("not a .fly file (bad magic)")
    (header_len,) = struct.unpack_from("<I", blob, 8)
    off = 12
    header = json.loads(blob[off : off + header_len].decode("utf-8"))
    off += header_len
    if header.get("format") != "fly-reservoir" or header.get("version") != 1:
        raise ValueError("unsupported .fly header: %r" % (header.get("format"),))
    n = int(header["neurons"])
    edges = int(header["edges"])
    embed_dim = int(header["embed_dim"])
    steps = int(header["steps"])
    weight_scale = float(header["weight_scale"])

    payload = blob[off:]
    if _has_zstd_frame(payload):
        payload = _zstd_decompress(payload)

    pos = 0
    indptr = struct.unpack_from("<%dI" % (n + 1), payload, pos)
    pos += 4 * (n + 1)
    indices = struct.unpack_from("<%dI" % edges, payload, pos)
    pos += 4 * edges
    weights = _dequant_int8(payload[pos : pos + edges], weight_scale)
    pos += edges
    (input_mode,) = struct.unpack_from("<B", payload, pos)
    pos += 1
    if input_mode == 1:
        in_w_raw = payload[pos : pos + embed_dim * n]
        pos += embed_dim * n
        (row_len,) = struct.unpack_from("<I", payload, pos)
        pos += 4
        if row_len != n:
            raise ValueError("in_row_len %d != neurons %d" % (row_len, n))
        (in_scale,) = struct.unpack_from("<d", payload, pos)
        pos += 8
        win = _dequant_int8(in_w_raw, in_scale)
    elif input_mode == 0:
        raise ValueError(
            "seed-mode .fly cannot be expanded without the generator; "
            "pack a matrix-mode artifact for training"
        )
    else:
        raise ValueError("unknown input_mode %d" % input_mode)

    pos += n * len(header["classes"])  # readout (int8) - not used by states()
    cfg_decay = (
        float(header["decay"])
        if "decay" in header
        else (float(decay) if decay is not None else DEFAULT_DECAY)
    )
    cfg = {
        "name": header.get("name", ""),
        "neurons": n,
        "edges": edges,
        "embed_dim": embed_dim,
        "steps": steps,
        "decay": cfg_decay,
        "classes": list(header["classes"]),
        "indptr": list(indptr),
        "indices": list(indices),
        "weights": weights,
        "weight_scale": weight_scale,
        "win": win,
        "in_scale": in_scale,
    }
    return cfg


def load_states_config(source, decay=None):
    """Normalize a .fly path or an adjacency dict into the states() config.

    Both forms carry int8 weights plus their scales; the config holds the
    dequantized float64 W values and input projection ``win``.

    Contract 1's frozen header has no ``decay`` field; the trainer honors an
    optional ``decay`` header extension when present (the packer writes it),
    otherwise the ``decay`` argument, otherwise DEFAULT_DECAY.
    """
    if isinstance(source, (str, Path)):
        return _read_fly(source, decay=decay)
    a = source
    n = int(a["neurons"])
    embed_dim = int(a["embed_dim"])
    win = a.get("win")
    if win is None:
        in_w = a.get("in_w")
        if in_w is None:
            raise ValueError("adjacency dict needs in_w (matrix input mode)")
        if len(in_w) != embed_dim * n:
            raise ValueError("in_w must have embed_dim*neurons entries")
        win = _dequant_int8(_int8_bytes(in_w), float(a["in_scale"]))
    return {
        "name": a.get("name", ""),
        "neurons": n,
        "edges": int(a.get("edges", len(a["indices"]))),
        "embed_dim": embed_dim,
        "steps": int(a["steps"]),
        "decay": float(a["decay"]) if a.get("decay") is not None else (float(decay) if decay is not None else DEFAULT_DECAY),
        "classes": list(a.get("classes", [])),
        "indptr": [int(v) for v in a["indptr"]],
        "indices": [int(v) for v in a["indices"]],
        "weights": _dequant_int8(_int8_bytes(a["weights"]), float(a["weight_scale"]))
        if all(isinstance(v, int) and -128 <= v <= 127 for v in a["weights"])
        else [float(v) for v in a["weights"]],
        "weight_scale": float(a.get("weight_scale", 1.0)),
        "win": win,
        "in_scale": float(a.get("in_scale", 1.0)),
    }


def states(cfg, x):
    """Run the frozen reservoir recurrence for one embedding.

    cfg: config dict from load_states_config (or the normalized adjacency
    dict). x: embedding, length embed_dim. Returns the state vector of
    length neurons, each value in (-1, 1). Deterministic; identical inputs
    give identical outputs in both the numpy and stdlib paths.
    """
    n = cfg["neurons"]
    steps = cfg["steps"]
    decay = cfg["decay"]
    indptr = cfg["indptr"]
    indices = cfg["indices"]
    vals = cfg["weights"]
    win = cfg["win"]

    if len(x) != cfg["embed_dim"]:
        raise ValueError(
            "embedding dim %d != embed_dim %d" % (len(x), cfg["embed_dim"])
        )

    if _np is not None:
        xv = _np.asarray(x, dtype=_np.float64)
        w = _np.asarray(win, dtype=_np.float64).reshape(cfg["embed_dim"], n)
        drive = xv @ w  # length n, fixed input projection
        s = _np.zeros(n, dtype=_np.float64)
        for _ in range(steps):
            recur = _np.zeros(n, dtype=_np.float64)
            for i in range(n):
                lo, hi = indptr[i], indptr[i + 1]
                if hi > lo:
                    recur[i] = _np.dot(vals[lo:hi], s[indices[lo:hi]])
            s = _np.tanh(decay * s + recur + drive)
        return [float(v) for v in s]

    # stdlib fallback: same order of operations as the numpy path
    drive = [0.0] * n
    for d in range(cfg["embed_dim"]):
        xd = x[d]
        if xd == 0.0:
            continue
        base = d * n
        for i in range(n):
            drive[i] += win[base + i] * xd
    s = [0.0] * n
    for _ in range(steps):
        new = [0.0] * n
        for i in range(n):
            acc = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                acc += vals[j] * s[indices[j]]
            new[i] = _tanh(decay * s[i] + acc + drive[i])
        s = new
    return s


def _tanh(v):
    if _np is not None:
        return float(_np.tanh(v))
    if v > 20.0:
        return 1.0
    if v < -20.0:
        return -1.0
    e2 = _exp(2.0 * v)
    return (e2 - 1.0) / (e2 + 1.0)


def _exp(v):
    # minimal stdlib exp (stdlib fallback path only)
    import math

    return math.exp(v)
