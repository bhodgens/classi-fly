"""Fixture-grade reader for the Contract 1 `.fly` artifact + Classify wrapper.

DEVIATION NOTES:
- Implements enough of Contract 1 to LOAD the packed eval fixture: exact
  binary layout (magic "FLYRES01", little-endian, JSON header, CSR weights,
  input projection, int8 readout, bias, per-class thresholds). The zstd outer
  container is honored when the `zstandard` module is importable; this
  environment lacks it, so the committed fixture is stored UNCOMPRESSED
  (zstd frames are detected by magic and decompressed when available).
- The forward pass is this harness's reference semantics: u = x @ Win
  (D -> S), h = tanh(u + A h) for `steps` steps, softmax(readout^T h + bias),
  abstain when top-1 probability < the winning class's threshold (Contract 2:
  abstained => zero confidence/margin). Full producer/quantization semantics
  are owned by 01-reservoir-core; consumers (Go, sidecar) must match them.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"FLYRES01"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


@dataclass
class Result:
    Class: str
    Confidence: float
    Margin: float
    Abstained: bool


@dataclass
class Info:
    Name: str = ""
    Neurons: int = 0
    Edges: int = 0
    EmbedDim: int = 0
    Steps: int = 0
    Classes: list = field(default_factory=list)
    License: str = ""
    Attribution: str = ""


def _maybe_zstd(raw: bytes) -> bytes:
    """Normalize a .fly file to (header_len + header + raw payload) form.

    Two layouts exist:
    - Python-fixture layout: EITHER the whole file is a zstd frame of a raw
      artifact, OR the raw artifact bytes directly.
    - Go packer layout (reservoir.WriteFile): clear "FLYRES01 + headerLen +
      JSON header" in front of a zstd-COMPRESSED payload. The payload frame
      is decompressed in place so Load()'s header offsets still resolve; the
      reassembled buffer keeps its original header bytes.
    """
    if raw[:8] == MAGIC:
        (hl,) = struct.unpack_from("<I", raw, 8)
        payload = raw[12 + hl:]
        if raw[12:13] == b"{" and payload[:4] == ZSTD_MAGIC:
            try:
                import zstandard
            except ImportError as exc:
                raise RuntimeError(
                    "fly: zstd-compressed .fly needs the zstandard module") from exc
            return raw[:12 + hl] + zstandard.ZstdDecompressor().decompress(payload)
        return raw
    if raw[:4] == ZSTD_MAGIC:
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError(
                "fly: zstd-compressed .fly needs the zstandard module") from exc
        raw = zstandard.ZstdDecompressor().decompress(raw)
    if raw[:8] != MAGIC:
        raise ValueError("fly: bad magic (not a FLYRES01 artifact)")
    return raw


class _Model:
    # Declared up front so static analysis sees the attribute set.
    header: dict
    n: int
    k: int
    d: int
    steps: int
    classes: list
    weight_scale: float
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    edges: int
    input_mode: int
    seed: int | None
    in_w: np.ndarray | None
    in_row_len: int
    in_scale: float
    win: np.ndarray | None
    readout: np.ndarray
    readout_scale: float
    bias: np.ndarray
    threshold: np.ndarray


class Reservoir:
    """Python mirror of Contract 2 (Load / Classify / Info), fail-closed."""

    def __init__(self, m: _Model):
        self._m = m

    @classmethod
    def Load(cls, path) -> "Reservoir":
        raw = _maybe_zstd(Path(path).read_bytes())
        m = _Model()
        (header_len,) = struct.unpack_from("<I", raw, 8)
        off = 12
        m.header = json.loads(raw[off:off + header_len].decode("utf-8"))
        off += header_len
        n = int(m.header["neurons"])
        k = len(m.header["classes"])
        d = int(m.header["embed_dim"])
        if n <= 0 or k <= 0 or d <= 0:
            raise ValueError("fly: degenerate header dimensions")
        m.n, m.k, m.d = n, k, d
        m.steps = int(m.header["steps"])
        m.classes = [str(c) for c in m.header["classes"]]
        m.weight_scale = float(m.header["weight_scale"])

        indptr = np.frombuffer(raw, dtype="<u4", count=n + 1, offset=off).astype(np.int64)
        off += 4 * (n + 1)
        e = int(indptr[-1])
        indices = np.frombuffer(raw, dtype="<u4", count=e, offset=off).astype(np.int64)
        off += 4 * e
        weights = np.frombuffer(raw, dtype=np.int8, count=e, offset=off)
        off += e
        if (indptr[:-1] > indptr[1:]).any() or (indices >= n).any():
            raise ValueError("fly: corrupt CSR adjacency")
        m.indptr, m.indices, m.weights = indptr, indices, weights
        m.edges = e

        (m.input_mode,) = struct.unpack_from("<B", raw, off)
        off += 1
        if m.input_mode == 0:
            (m.seed,) = struct.unpack_from("<Q", raw, off)
            off += 8
            m.in_w = None
            m.in_row_len = n
            m.in_scale = 0.25  # fixture-grade seed-mode dequant scale
        elif m.input_mode == 1:
            s = n  # payload is D*S int8 with row length S; S == N in this format use
            m.in_w = np.frombuffer(raw, dtype=np.int8, count=d * s, offset=off).reshape(d, s)
            off += d * s
            (m.in_row_len,) = struct.unpack_from("<I", raw, off)
            off += 4
            if m.in_row_len != s:
                raise ValueError("fly: unsupported input row length %d" % m.in_row_len)
            (m.in_scale,) = struct.unpack_from("<d", raw, off)
            off += 8
            m.seed = None
        else:
            raise ValueError("fly: unknown input_mode %d" % m.input_mode)

        m.readout = np.frombuffer(raw, dtype=np.int8, count=n * k, offset=off).reshape(n, k)
        off += n * k
        (m.readout_scale,) = struct.unpack_from("<d", raw, off)
        off += 8
        m.bias = np.frombuffer(raw, dtype="<f4", count=k, offset=off).astype(np.float64)
        off += 4 * k
        m.threshold = np.frombuffer(raw, dtype="<f4", count=k, offset=off).astype(np.float64)
        off += 4 * k
        if off != len(raw):
            raise ValueError("fly: %d trailing bytes (corrupt artifact)" % (len(raw) - off))
        if m.input_mode == 0:
            rng = np.random.default_rng(m.seed)
            m.win = rng.integers(-2, 3, size=(d, n)).astype(np.float64) * m.in_scale
        return cls(m)

    def Info(self) -> Info:
        h = self._m.header
        return Info(Name=h.get("name", ""), Neurons=self._m.n, Edges=self._m.edges,
                    EmbedDim=self._m.d, Steps=self._m.steps, Classes=list(self._m.classes),
                    License=h.get("license", ""), Attribution=h.get("attribution", ""))

    def _propagate(self, x: np.ndarray) -> np.ndarray:
        m = self._m
        if m.in_w is not None:
            u = x @ (m.in_w.astype(np.float64)) * m.in_scale
        else:
            u = x @ m.win
        n = m.n
        if len(u) < n:
            u = np.concatenate([u, np.zeros(n - len(u))])
        elif len(u) > n:
            u = u[:n]
        vals = m.weights.astype(np.float64) * m.weight_scale
        ip, ix = m.indptr, m.indices
        h = np.zeros(n)
        for _ in range(m.steps):
            ah = np.empty(n)
            for i in range(n):
                lo, hi = ip[i], ip[i + 1]
                ah[i] = float(vals[lo:hi] @ h[ix[lo:hi]])
            h = np.tanh(u + ah)
        return h

    def Classify(self, embedding) -> Result:
        m = self._m
        x = np.asarray(embedding, dtype=np.float64).ravel()
        if len(x) != m.d:
            raise ValueError("fly: embedding dim %d != artifact embed_dim %d" % (len(x), m.d))
        h = self._propagate(x)
        logits = h @ (m.readout.astype(np.float64) * m.readout_scale) + m.bias
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        order = np.argsort(-p)
        top = int(order[0])
        conf = float(p[top])
        margin = float(p[top] - p[order[1]]) if m.k > 1 else conf
        if conf < float(m.threshold[top]):
            # Contract 2: abstained => zero confidence/margin.
            return Result(Class=m.classes[top], Confidence=0.0, Margin=0.0, Abstained=True)
        return Result(Class=m.classes[top], Confidence=conf, Margin=margin, Abstained=False)
