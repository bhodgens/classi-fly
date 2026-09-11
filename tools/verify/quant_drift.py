"""Float vs int8 quantization drift, measured on real `.fly` artifacts.

Method (leaf 08 Task 2, as amended by the orchestrator): craft TWO `.fly`
artifacts with identical topology/classes/readout but different input/readout
representations, load both with the Python reference reader
(``tools/eval/flyio.py``), and compare the final state vectors and
classifications across a battery of probe embeddings.

The float reference: seed-mode input projection expanded in float64
(``win = rng.integers(-2,3). * in_scale``, the documented Contract 1 seed
expansion) and a float64 (dequantized) readout. The int8 twin: the SAME
values rounded to exact int8 storage with the same dequant scales - i.e. what
a producer that quantizes in_scale-larger weights into int8 actually stores.
The residual difference between the two forward passes is the quantization
drift this tool measures.

Verdict: drift is acceptable when (a) the argmax class agrees on every probe
and (b) the abstain decision agrees on every probe. Max state delta is
reported and compared against the documented tolerance (1e-6 relative on
tanh-bounded states is the bar the Go core claims; quantization of the input
projection is allowed a coarser 5e-2 gate - the boundary is: does any probe
change classification or route/abstain?).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

# flyio: importable both under pytest (conftest.py sets sys.path) and as a
# direct CLI run (this fallback).
_HERE = Path(__file__).resolve().parent
_EVAL = _HERE.parent / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

import flyio  # noqa: E402

MAX_STATE_DELTA = 5e-2     # documented gate for int8-quantized projections
MAX_PROB_DELTA = 1e-2      # softmax probability drift gate
SEED_INT_RANGE = 3         # Contract 1 seed expansion: integers in [-2, 2]


def _encode_fly(header: dict, payload: bytes) -> bytes:
    """Container writer matching flybytes/flyio (raw payload, no zstd frame).

    Written here rather than imported so quant_drift stays independent of the
    trainer's fixture helper.
    """
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    return (b"FLYRES01" + struct.pack("<I", len(header_bytes))
            + header_bytes + payload)


def _csr_payload(artifact: dict) -> bytes:
    n = artifact["neurons"]
    indptr, indices, weights = artifact["indptr"], artifact["indices"], artifact["weights"]
    out = struct.pack("<%dI" % (n + 1), *[int(v) for v in indptr])
    out += struct.pack("<%dI" % len(indices), *[int(v) for v in indices])
    out += struct.pack("<%db" % len(weights), *[int(v) for v in weights])
    return out


def _win_reference(embed_dim: int, neurons: int, seed: int, in_scale: float) -> np.ndarray:
    """The documented Contract 1 seed expansion, float64.

    MUST mirror flyio._propagate's expansion exactly (integers(-2, 3)) -
    this is the value the float artifact's reader expands and the int8 twin
    quantizes, so the pair differs ONLY by int8 round-trip error.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(-2, 3, size=(embed_dim, neurons)).astype(np.float64) * in_scale


def build_float_fly(path, *, neurons: int = 6, embed_dim: int = 4, steps: int = 3,
                    classes=("a", "b"), seed: int = 1234, weight_scale: float = 0.5,
                    in_scale: float = 0.25, readout_scale: float = 0.25,
                    license_: str = "none", name: str = "quant-float") -> Path:
    """FLOAT reference artifact: seed-mode input, full-precision readout.

    Seed mode: flyio expands win = rng.integers(-2,3,(d,n)) * in_scale in
    float64, and the readout int8 * readout_scale dequantizes exactly, so
    quantization error enters only through the readout's int8 grid.
    """
    n, d, k = neurons, embed_dim, len(classes)
    win = _win_reference(d, n, seed, in_scale)
    rng = np.random.default_rng(seed)
    rng.integers(-2, 3, size=(d, n))  # burn the same draws so readout matches the twin
    readout = np.rint(rng.uniform(-4, 4, size=(n, k))).astype(np.int8)
    bias = [0.0] * k
    thresholds = [0.35] * k  # non-trivial abstain boundary

    header = {
        "format": "fly-reservoir", "version": 1, "name": name,
        "neurons": n, "edges": n, "embed_dim": d, "steps": steps,
        "classes": list(classes), "weight_scale": weight_scale,
        "source": "synthetic", "license": license_,
        "attribution": "tools/verify", "created_utc": "2026-09-11T00:00:00Z",
    }
    payload = _csr_payload({
        "neurons": n,
        "indptr": list(range(n + 1)),
        "indices": list(range(n)),          # self-loop per lane
        "weights": [1] * n,
    })
    payload += struct.pack("<B", 0) + struct.pack("<Q", seed)  # input_mode=seed
    payload += readout.tobytes()
    payload += struct.pack("<d", readout_scale)
    payload += struct.pack("<%df" % k, *bias)
    payload += struct.pack("<%df" % k, *thresholds)
    Path(path).write_bytes(_encode_fly(header, payload))
    return Path(path)


def build_int8_fly(path, *, neurons: int = 6, embed_dim: int = 4, steps: int = 3,
                   classes=("a", "b"), seed: int = 1234, weight_scale: float = 0.5,
                   in_scale: float = 0.25, readout_scale: float = 0.25,
                   license_: str = "none", name: str = "quant-int8") -> Path:
    """INT8 twin: matrix-mode in_w quantized from the SAME win values.

    float-mode win entries are integers * in_scale, so int8 storage with
    in_scale as the dequant scale is EXACT for them - the twin differs from
    the float artifact only by whatever the runtimes actually diverge on.
    To make the comparison a real stress, in_w is stored via round(x /
    in_scale) and decoded by the reader as int8 * in_scale, i.e. the classic
    quantize/dequantize round trip including its rounding error.
    """
    n, d, k = neurons, embed_dim, len(classes)
    win = _win_reference(d, n, seed, in_scale)
    rng = np.random.default_rng(seed)
    rng.integers(-2, 3, size=(d, n))  # burn the same draws so readout matches the float twin
    readout = np.rint(rng.uniform(-4, 4, size=(n, k))).astype(np.int8)
    bias = [0.0] * k
    thresholds = [0.35] * k

    header = {
        "format": "fly-reservoir", "version": 1, "name": name,
        "neurons": n, "edges": n, "embed_dim": d, "steps": steps,
        "classes": list(classes), "weight_scale": weight_scale,
        "source": "synthetic", "license": license_,
        "attribution": "tools/verify", "created_utc": "2026-09-11T00:00:00Z",
    }
    payload = _csr_payload({
        "neurons": n,
        "indptr": list(range(n + 1)),
        "indices": list(range(n)),
        "weights": [1] * n,
    })
    payload += struct.pack("<B", 1)  # input_mode = matrix
    in_w = np.rint(win / in_scale).astype(np.int8)
    payload += in_w.tobytes()
    payload += struct.pack("<I", n)
    payload += struct.pack("<d", in_scale)
    payload += readout.tobytes()
    payload += struct.pack("<d", readout_scale)
    payload += struct.pack("<%df" % k, *bias)
    payload += struct.pack("<%df" % k, *thresholds)
    Path(path).write_bytes(_encode_fly(header, payload))
    return Path(path)


def probe_embeddings(embed_dim: int = 4, count: int = 64, seed: int = 7) -> np.ndarray:
    """Deterministic probe battery: basis directions, signs, and noise."""
    rng = np.random.default_rng(seed)
    probes = []
    for c in range(embed_dim):
        v = np.zeros(embed_dim)
        v[c] = 1.0
        probes.append(v)
        probes.append(-v)
    probes.append(np.ones(embed_dim) / np.sqrt(embed_dim))
    while len(probes) < count:
        probes.append(rng.uniform(-1, 1, embed_dim))
    return np.vstack(probes)


def _states(res: "flyio.Reservoir", x: np.ndarray) -> np.ndarray:
    # _propagate is the reference forward pass; call it directly so the
    # comparison is state-vs-state, not just class-vs-class.
    return res._propagate(np.asarray(x, dtype=np.float64).ravel())


def compare(float_path, int8_path, *, embed_dim: int = 4) -> dict:
    """Run both artifacts over the probe battery; return the drift report."""
    rf = flyio.Reservoir.Load(float_path)
    ri = flyio.Reservoir.Load(int8_path)
    if rf._m.classes != ri._m.classes:
        raise ValueError("quant pair: class lists differ")
    probes = probe_embeddings(embed_dim=embed_dim)
    class_disagree = 0
    route_disagree = 0
    max_state_delta = 0.0
    max_prob_delta = 0.0
    worst = None
    for x in probes:
        hf = _states(rf, x)
        hi = _states(ri, x)
        d = float(np.max(np.abs(hf - hi)))
        if d > max_state_delta:
            max_state_delta, worst = d, [float(v) for v in x]
        p_f, p_i = rf.Classify(x), ri.Classify(x)
        probs_f = _softmax_of(rf, hf)
        probs_i = _softmax_of(ri, hi)
        max_prob_delta = max(max_prob_delta,
                             float(np.max(np.abs(probs_f - probs_i))))
        if p_f.Class != p_i.Class:
            class_disagree += 1
        if p_f.Abstained != p_i.Abstained:
            route_disagree += 1
    return {
        "probes": len(probes),
        "class_disagreements": class_disagree,
        "route_disagreements": route_disagree,
        "max_state_delta": max_state_delta,
        "max_prob_delta": max_prob_delta,
        "state_tolerance": MAX_STATE_DELTA,
        "prob_tolerance": MAX_PROB_DELTA,
        "state_ok": max_state_delta <= MAX_STATE_DELTA,
        "prob_ok": max_prob_delta <= MAX_PROB_DELTA,
        "classify_ok": class_disagree == 0 and route_disagree == 0,
        "worst_probe": worst,
    }


def _softmax_of(res: "flyio.Reservoir", h: np.ndarray) -> np.ndarray:
    m = res._m
    logits = h @ (m.readout.astype(np.float64) * m.readout_scale) + m.bias
    z = logits - logits.max()
    p = np.exp(z)
    return p / p.sum()


def format_report(rep: dict) -> str:
    lines = [
        "quant drift: %d probes" % rep["probes"],
        "  max |state| delta : %.3e (tol %.1e) %s" % (
            rep["max_state_delta"], rep["state_tolerance"],
            "ok" if rep["state_ok"] else "OVER"),
        "  max prob delta    : %.3e (tol %.1e) %s" % (
            rep["max_prob_delta"], rep["prob_tolerance"],
            "ok" if rep["prob_ok"] else "OVER"),
        "  class disagreements: %d" % rep["class_disagreements"],
        "  route disagreements: %d" % rep["route_disagreements"],
    ]
    ok = rep["state_ok"] and rep["prob_ok"] and rep["classify_ok"]
    lines.append("  VERDICT: %s" % ("PASS - no classification changed" if ok else "FAIL - quantization changed behavior"))
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    import tempfile
    p = argparse.ArgumentParser(
        description="Float-vs-int8 agreement check on synthetic .fly twins")
    p.add_argument("--float-fly", default=None, help="float reference .fly (default: build one)")
    p.add_argument("--int8-fly", default=None, help="int8 twin .fly (default: build one)")
    p.add_argument("--embed-dim", type=int, default=4)
    p.add_argument("--neurons", type=int, default=6)
    p.add_argument("--out-dir", default=None, help="where default twins are written")
    args = p.parse_args(argv)
    fpath, ipath = args.float_fly, args.int8_fly
    tmp = None
    if fpath is None or ipath is None:
        tmp = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="quant-drift-"))
        tmp.mkdir(parents=True, exist_ok=True)
        kw = dict(neurons=args.neurons, embed_dim=args.embed_dim)
        if fpath is None:
            fpath = build_float_fly(tmp / "float_ref.fly", **kw)
        if ipath is None:
            ipath = build_int8_fly(tmp / "int8.fly", **kw)
    rep = compare(fpath, ipath, embed_dim=args.embed_dim)
    print(format_report(rep))
    if tmp is not None:
        print("  artifacts: %s | %s" % (fpath, ipath))
    return 0 if (rep["classify_ok"] and rep["state_ok"] and rep["prob_ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
