"""Tests for the loopback health-check HTTP server.

httptest-style: each test module gets a REAL ThreadingHTTPServer bound to
127.0.0.1:0 (ephemeral port) in a daemon thread, then exercises the wire
contract with urllib — the same way the Go daemon will call it.

Pinned properties:
  1. /health/status: 200 {"status":"ok"} with a reference set; 503 without.
  2. /health/check: matches the module's EmbedHealthResult for gold,
     corrupted, zero and NaN vectors; healthy=null (fail-safe) when no
     reference set; 400 on malformed bodies; 404 on unknown paths.
  3. /health/calibrate: swaps the reference atomically and returns
     {"threshold", "reference_size"}; the new threshold matches calibrate();
     a subsequent /health/check uses the new set; 400 on bad vectors.

Run:  python3 -m pytest tools/eval/test_embed_health_server.py -q
"""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))

from embed_health import calibrate  # noqa: E402
from embed_health_server import EmbedHealthState, make_handler  # noqa: E402


def _corrupt(kind, param, X, rng):
    """Byte-identical recipe to meept_test3_degraded.corrupt (TEST-only)."""
    Xc = X.copy()
    D = X.shape[1]
    if kind == "noise":
        return Xc + param * rng.normal(0.0, 1.0, size=X.shape)
    if kind == "dropout":
        return Xc * (rng.random(X.shape) >= param)
    if kind == "trunc":
        cut = int(round(D * param))
        Xc[:, D - cut:] = 0.0
        return Xc
    raise ValueError(kind)


class Server:
    def __init__(self, ref_path):
        self.state = EmbedHealthState(ref_path)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0),
                                          make_handler(self.state))
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def get(self, path: str):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def post(self, path: str, payload=None, raw: bytes | None = None):
        data = raw if raw is not None else json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())


@pytest.fixture(scope="module")
def gold():
    d = json.loads((REPO / "data" / "e1_inputs.json").read_text())
    vecs = np.asarray(d["vecs"], dtype=np.float64)
    return vecs[:len(d["gold"])]


@pytest.fixture(scope="module")
def loaded(gold):
    s = Server(REPO / "data" / "embed_health_ref.npz")
    yield s, gold
    s.stop()


@pytest.fixture(scope="module")
def bare():
    s = Server("/nonexistent/path/ref.npz")   # fail-safe on purpose
    yield s
    s.stop()


# ------------------------------------------------------------------ status
def test_status_ok_with_reference(loaded):
    s, _ = loaded
    code, body = s.get("/health/status")
    assert code == 200 and body == {"status": "ok"}


def test_status_503_without_reference(bare):
    code, body = bare.get("/health/status")
    assert code == 503 and body == {"status": "no_reference_set"}


def test_unknown_path_404(loaded):
    s, _ = loaded
    assert s.get("/nope")[0] == 404
    assert s.post("/nope", {})[0] == 404


# ------------------------------------------------------------------ check
def test_check_gold_embedding_is_healthy(loaded):
    s, gold = loaded
    code, body = s.post("/health/check", {"embedding": gold[0].tolist()})
    assert code == 200
    assert body["healthy"] is True
    assert 0.0 <= body["distance"] <= body["threshold"]
    assert body["threshold"] == pytest.approx(calibrate(gold))


def test_check_corrupted_embedding_is_flagged(loaded):
    s, gold = loaded
    X = _corrupt("noise", 0.2, gold[:20], np.random.default_rng(7))
    for row in X:
        code, body = s.post("/health/check", {"embedding": row.tolist()})
        assert code == 200 and body["healthy"] is False


def test_check_zero_vector_flagged(loaded):
    s, _ = loaded
    code, body = s.post("/health/check", {"embedding": [0.0] * 1024})
    assert code == 200 and body["healthy"] is False
    assert body["distance"] is None


def test_check_nan_vector_flagged(loaded):
    # NaN is invalid JSON-numeric content for the wire guard, so the server
    # rejects it with 400 when loaded — the daemon sees a hard error, which
    # is the fail-closed answer. (json.dumps produces NaN literal; Python's
    # json accepts it, our finite-guard does not.)
    s, _ = loaded
    code, body = s.post("/health/check",
                        {"embedding": [float("nan")] + [1.0] * 1023})
    assert code == 400


def test_check_fail_safe_without_reference(bare):
    code, body = bare.post("/health/check", {"embedding": [1.0] * 1024})
    assert code == 200
    assert body["healthy"] is None and body["method" if "method" in body else "healthy"] is None
    assert body["distance"] is None and body["threshold"] is None


def test_check_malformed_body_400(loaded):
    s, _ = loaded
    assert s.post("/health/check", raw=b"not json")[0] == 400
    assert s.post("/health/check", {})[0] == 400
    assert s.post("/health/check", {"embedding": "nope"})[0] == 400
    assert s.post("/health/check", {"embedding": [["a"] * 1024]})[0] == 400


# ------------------------------------------------------------------ calibrate
def test_calibrate_swaps_reference(loaded, gold):
    s, gold_all = loaded
    X = gold_all[:100]
    code, body = s.post("/health/calibrate", {"vectors": X.tolist()})
    assert code == 200
    assert body["reference_size"] == 100
    assert body["threshold"] == pytest.approx(calibrate(X))
    assert s.state.checker.threshold == pytest.approx(calibrate(X))
    # a check against a vector near the NEW set is healthy with NEW threshold
    code, body = s.post("/health/check", {"embedding": X[0].tolist()})
    assert code == 200 and body["healthy"] is True
    assert body["threshold"] == pytest.approx(calibrate(X))


def test_calibrate_rejects_bad_payloads(loaded):
    s, _ = loaded
    assert s.post("/health/calibrate", {})[0] == 400
    assert s.post("/health/calibrate", {"vectors": []})[0] == 400
    assert s.post("/health/calibrate", {"vectors": [[1.0, 2.0]]})[0] == 400  # <2 vecs
    assert s.post("/health/calibrate", {"vectors": "nope"})[0] == 400
    code, _ = s.post("/health/calibrate",
                     {"vectors": [[1.0] * 8 for _ in range(5)]})
    assert code == 200   # tiny dim is fine; module only needs >=2 rows


def test_calibrate_then_status_ok(bare, gold):
    """A fail-safe server recovers via /health/calibrate."""
    code, body = bare.post("/health/calibrate",
                           {"vectors": gold[:50].tolist()})
    assert code == 200 and body["reference_size"] == 50
    assert bare.get("/health/status")[0] == 200
    code, body = bare.post("/health/check", {"embedding": gold[0].tolist()})
    assert code == 200 and body["healthy"] is True
