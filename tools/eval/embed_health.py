"""Embedding-pipeline health check (meept issue #42).

Mechanism (measured and verified, docs/MEEPT-TESTS.md test 3 /
docs/AXES-2026-09-12.md axis D): cosine distance from the incoming embedding
to the NEAREST vector in a fixed reference set of known-good embeddings.
Per-item distance to a fixed reference set detected every distribution shift
with 2.5-step median latency and 0.83% false alarms; state-novelty was worse.

Design.
  - Reference set: known-good embeddings stored as an NPZ file (key
    "vectors", shape (N, D)). Build it from data/e1_inputs.json gold
    embeddings with `build_reference` / `python3 tools/eval/embed_health.py
    build`.
  - Threshold: the `threshold_pct` percentile of the reference's known-good
    NEAREST-NEIGHBOUR cosine distances, estimated CROSS-FITTED (k-fold:
    each vector is scored against the other folds, never itself). Default
    threshold_pct=0.95 -> 95% of known-good embeddings pass. On the e1
    gold corpus this lands at ~0.265 and retains 95%+ of held-out gold
    (see tools/eval/test_embed_health.py).
  - check(): distance to nearest reference vector; healthy iff
    distance <= threshold.
  - Fail-safe: a missing/corrupt/empty reference set yields method
    "unknown" and never claims healthy. Zero-norm or NaN inputs are flagged
    as unhealthy (degenerate_input), not crashed on.

Standalone Go-sidecar contract: tools/eval/embed_health_server.py wraps this
module over loopback HTTP (stdlib only, mirroring cmd/classi-fly/serve.go).

CLI:
  python3 tools/eval/embed_health.py build   # gold JSON -> reference NPZ
  python3 tools/eval/embed_health.py selftest
Run:  python3 -m pytest tools/eval/test_embed_health.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "data" / "e1_inputs.json"
REF_PATH = REPO / "data" / "embed_health_ref.npz"

REFERENCE_KEY = "vectors"
_EPS = 1e-12


class EmbedHealthResult(TypedDict):
    healthy: bool | None   # None == unknown (fail-safe, never claim healthy)
    distance: float | None
    threshold: float | None
    method: str            # "cosine_min_ref" | "degenerate_input" | "unknown"


def _norm_rows(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), _EPS)


def calibrate(x_gold: np.ndarray, threshold_pct: float = 0.95,
              folds: int = 5, seed: int = 0) -> float:
    """Deployment-honest threshold: `threshold_pct` percentile of CROSS-FITTED
    nearest-neighbour cosine distances.

    The reference is split into `folds` deterministic folds; each vector's
    score is its distance to the nearest vector in the OTHER folds, so no
    vector is ever scored against itself. Raises ValueError on an
    empty/degenerate set (< 2 vectors, NaN/inf).
    """
    x = np.asarray(x_gold, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(
            f"calibration needs >= 2 vectors of shape (N, D), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("calibration set contains NaN/inf")
    k = max(2, min(folds, x.shape[0]))
    xn = _norm_rows(x)
    idx = np.random.default_rng(seed).permutation(x.shape[0])
    parts = np.array_split(idx, k)
    dists = []
    for part in parts:
        rest = np.setdiff1d(idx, part)
        sims = xn[part] @ xn[rest].T
        dists.append(1.0 - sims.max(axis=1))
    return float(np.quantile(np.concatenate(dists), threshold_pct))
    # NOTE: `1.0 - sims.max(axis=1)` is the NN distance (max similarity);
    # `(1.0 - sims).max(axis=1)` would be the distance to the FARTHEST vector.


class EmbedHealthCheck:
    """Cosine distance to nearest known-good reference vector."""

    def __init__(self, ref_path: str | Path,
                 threshold_pct: float = 0.95,
                 ref_vectors: np.ndarray | None = None):
        self.threshold_pct = float(threshold_pct)
        self.threshold: float | None = None
        self._ref: np.ndarray | None = None     # normalized (N, D)
        if ref_vectors is not None:
            self._load_array(np.asarray(ref_vectors, dtype=np.float64))
        else:
            self._load_file(Path(ref_path))

    # ---- loading (fail-safe) ------------------------------------------
    def _load_array(self, vecs: np.ndarray) -> None:
        try:
            if vecs.ndim != 2 or vecs.shape[0] < 2:
                raise ValueError("reference set needs shape (N>=2, D)")
            if not np.all(np.isfinite(vecs)):
                raise ValueError("reference set contains NaN/inf")
            self.threshold = calibrate(vecs, self.threshold_pct)
            self._ref = _norm_rows(vecs)
        except (ValueError, FloatingPointError):
            self._ref = None
            self.threshold = None

    def _load_file(self, path: Path) -> None:
        try:
            with np.load(path) as z:
                self._load_array(np.asarray(z[REFERENCE_KEY],
                                            dtype=np.float64))
        except Exception:
            self._ref = None
            self.threshold = None

    @property
    def loaded(self) -> bool:
        return self._ref is not None

    # ---- checking ------------------------------------------------------
    def _unknown(self) -> EmbedHealthResult:
        return {"healthy": None, "distance": None, "threshold": None,
                "method": "unknown"}

    def check(self, embedding: np.ndarray) -> EmbedHealthResult:
        ref = self._ref
        thr = self.threshold
        if ref is None or thr is None:
            return self._unknown()
        x = np.asarray(embedding, dtype=np.float64).ravel()
        if x.shape[0] != ref.shape[1]:
            return {"healthy": False, "distance": None,
                    "threshold": thr,
                    "method": "degenerate_input"}   # dim mismatch = degraded
        if not np.all(np.isfinite(x)):
            return {"healthy": False, "distance": None,
                    "threshold": thr,
                    "method": "degenerate_input"}   # NaN/inf
        n = float(np.linalg.norm(x))
        if n <= _EPS:
            return {"healthy": False, "distance": None,
                    "threshold": thr,
                    "method": "degenerate_input"}   # zero vector
        sims = ref @ (x / n)
        dist = float(1.0 - float(np.max(sims)))
        return {"healthy": dist <= thr, "distance": dist,
                "threshold": thr, "method": "cosine_min_ref"}

    def check_batch(self, embeddings: np.ndarray) -> list[EmbedHealthResult]:
        X = np.asarray(embeddings, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.loaded:
            return [self._unknown() for _ in range(X.shape[0])]
        return [self.check(X[i]) for i in range(X.shape[0])]


# ------------------------------------------------------------------ CLI
def build_reference(gold_json: Path = GOLD, out: Path = REF_PATH,
                    n: int | None = None) -> int:
    """Build the reference NPZ from e1_inputs.json gold embeddings."""
    d = json.loads(gold_json.read_text())
    gold = np.asarray(d["vecs"], dtype=np.float64)[:len(d["gold"])]
    if n is not None:
        gold = gold[:n]
    np.savez_compressed(out, vectors=gold)
    print(f"wrote {out} ({gold.shape[0]} x {gold.shape[1]}), "
          f"threshold@0.95={calibrate(gold):.4f}")
    return 0


def _selftest() -> int:
    chk = EmbedHealthCheck(REF_PATH)
    assert chk.loaded, f"reference set failed to load from {REF_PATH}"
    d = json.loads(GOLD.read_text())
    gold = np.asarray(d["vecs"], dtype=np.float64)
    holdout = gold[len(d["gold"]):]                 # gold-OOD rows: not in ref
    batch = chk.check_batch(holdout)
    ret = float(np.mean([bool(r["healthy"]) for r in batch]))
    print(f"loaded={chk.loaded} threshold={chk.threshold:.4f} "
          f"gold-OOD retention={ret:.3f} on {len(batch)} items")
    return 0 if ret >= 0.95 else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "build":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None
        raise SystemExit(build_reference(n=n))
    raise SystemExit(_selftest())
