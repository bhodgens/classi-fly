"""Contract 1 ``.fly`` fixture bytes for the offline trainer tests.

The Go packer (leaves 01/02) owns the production writer. This helper exists so
the Python trainer can be exercised before that lands: it emits tiny,
license-clean .fly files matching the frozen container layout from
master.md (magic ``FLYRES01``, little-endian, JSON header, packed arrays).

The payload region after the header is zstd-compressed when a zstd codec is
importable (stdlib ``compression.zstd`` on Python 3.14+, else the
``zstandard``/``pyzstd`` packages if present). When no codec is available the
payload is stored raw; the reader auto-detects the zstd frame magic, so both
forms parse.
"""

import json
import struct
from pathlib import Path

MAGIC = b"FLYRES01"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _zstd_compress(data):
    try:
        from compression import zstd as _z  # Python 3.14+ stdlib

        return _z.compress(data)
    except ImportError:
        pass
    try:
        import zstandard as _z  # third party

        return _z.ZstdCompressor().compress(data)
    except ImportError:
        pass
    try:
        import pyzstd as _z  # third party

        return _z.compress(data)
    except ImportError:
        return data


def _pack_int8(values, what):
    out = []
    for v in values:
        q = int(v)
        if q != v or not -128 <= q <= 127:
            raise ValueError("%s value %r is not an exact int8" % (what, v))
        out.append(q)
    return struct.pack("<%db" % len(out), *out)


def encode_payload(artifact):
    """Pack the post-header payload exactly as Contract 1 lays it out."""
    n = int(artifact["neurons"])
    indptr = artifact["indptr"]
    indices = artifact["indices"]
    weights = artifact["weights"]
    if len(indptr) != n + 1:
        raise ValueError("indptr must have n+1 entries")
    edges = len(indices)
    if len(weights) != edges:
        raise ValueError("weights/indices length mismatch")

    payload = struct.pack("<%dI" % len(indptr), *[int(v) for v in indptr])
    payload += struct.pack("<%dI" % edges, *[int(v) for v in indices])
    payload += _pack_int8(weights, "weights")

    win = artifact.get("win")
    in_w = artifact.get("in_w")
    if in_w is not None:
        embed_dim = int(artifact["embed_dim"])
        if len(in_w) != embed_dim * n:
            raise ValueError("in_w must have embed_dim*neurons entries")
        payload += struct.pack("<B", 1)
        payload += _pack_int8(in_w, "in_w")
        payload += struct.pack("<Id", n, float(artifact["in_scale"]))
    elif win is not None:
        raise ValueError(
            "the .fly container has no raw win-array mode (input_mode is "
            "seed or matrix); build the fixture with in_w/in_scale"
        )
    else:
        payload += struct.pack("<B", 0)
        payload += struct.pack("<Q", int(artifact.get("seed", 0)))

    classes = artifact.get("classes", [])
    k = len(classes)
    readout = artifact.get("readout", [0] * (n * k))
    if len(readout) != n * k:
        raise ValueError("readout must have neurons*len(classes) entries")
    payload += _pack_int8(readout, "readout")
    payload += struct.pack("<d", float(artifact.get("readout_scale", 1.0)))
    bias = artifact.get("bias", [0.0] * k)
    payload += struct.pack("<%df" % k, *[float(v) for v in bias])
    threshold = artifact.get("threshold", [0.0] * k)
    payload += struct.pack("<%df" % k, *[float(v) for v in threshold])
    return payload


def encode_fly(artifact):
    """Serialize a tiny artifact dict into .fly container bytes."""
    header = {
        "format": "fly-reservoir",
        "version": 1,
        "name": artifact.get("name", "fixture"),
        "neurons": int(artifact["neurons"]),
        "edges": len(artifact["indices"]),
        "embed_dim": int(artifact["embed_dim"]),
        "steps": int(artifact["steps"]),
        "classes": list(artifact.get("classes", [])),
        "weight_scale": float(artifact["weight_scale"]),
        "source": artifact.get("source", "synthetic"),
        "license": artifact.get("license", "none"),
        "attribution": artifact.get("attribution", "n/a"),
        "created_utc": artifact.get("created_utc", "2026-09-11T00:00:00Z"),
    }
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    payload = encode_payload(artifact)
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + _zstd_compress(payload)


def write_fly(path, artifact):
    """Write a tiny artifact dict as a .fly file at ``path``."""
    Path(path).write_bytes(encode_fly(artifact))
    return Path(path)
