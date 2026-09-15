"""Loopback HTTP server wrapping the embedding-pipeline health check.

Same pattern as cmd/classi-fly/serve.go: stdlib net/http equivalents
(http.server + ThreadingHTTPServer), no framework, loopback-only bind.

Endpoints:
  POST /health/check    {"embedding": [float,...]}
                        -> {"healthy": bool|null, "distance": float|null,
                            "threshold": float|null}
  POST /health/calibrate {"vectors": [[float,...],...]}
                        -> {"threshold": float, "reference_size": int}
                        (atomically replaces the in-memory reference set)
  GET  /health/status   -> 200 "ok" if a reference set is loaded, 503 otherwise

Fail-safe contract: healthy=null means "unknown - no reference set"; the
daemon must treat null as NOT healthy. Malformed JSON / wrong types -> 400.
The check endpoint answers null-healthy (not an error) while the reference
set is missing, so the daemon can distinguish "pipeline degraded" from
"health-check itself offline".

Run:  python3 tools/eval/embed_health_server.py [--port 8390] \
          [--ref data/embed_health_ref.npz]
Build the reference first:  python3 tools/eval/embed_health.py build
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from embed_health import EmbedHealthCheck, calibrate

_MAX_BODY = 64 * 1024 * 1024   # ~1M 1024-dim float64 embeddings, hard cap


class EmbedHealthState:
    """The mutable reference set behind the server (swap-in on calibrate)."""

    def __init__(self, ref_path):
        self.checker = EmbedHealthCheck(ref_path)

    @property
    def loaded(self) -> bool:
        return self.checker.loaded


def make_handler(state: EmbedHealthState):
    class Handler(BaseHTTPRequestHandler):
        # ---- plumbing ------------------------------------------------
        def log_message(self, format, *args):    # noqa: A002 - stdlib name
            pass

        def _json(self, code: int, payload) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > _MAX_BODY:
                return None
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None

        @staticmethod
        def _vec_list(obj) -> list | None:
            """[[float,...],...] with finite floats, or None."""
            if not isinstance(obj, list) or not obj:
                return None
            out = []
            for row in obj:
                if not isinstance(row, list) or not row:
                    return None
                try:
                    vals = [float(v) for v in row]
                except (TypeError, ValueError):
                    return None
                if not all(np.isfinite(vals)):
                    return None
                out.append(vals)
            return out

        # ---- endpoints -----------------------------------------------
        def do_GET(self) -> None:
            if self.path == "/health/status":
                if state.loaded:
                    self._json(200, {"status": "ok"})
                else:
                    self._json(503, {"status": "no_reference_set"})
                return
            self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path == "/health/check":
                self._check()
            elif self.path == "/health/calibrate":
                self._calibrate()
            else:
                self._json(404, {"error": "not_found"})

        def _check(self) -> None:
            body = self._read_json()
            if not isinstance(body, dict) or "embedding" not in body:
                self._json(400, {"error": "expected {\"embedding\": [float,...]}"})
                return
            rows = self._vec_list([body["embedding"]]) \
                if isinstance(body["embedding"], list) else None
            if rows is None:
                # Not a clean finite vector list. If no reference is loaded
                # the fail-safe answer is still more useful than a 400.
                if not state.loaded:
                    self._json(200, state.checker.check(np.zeros(1)))
                else:
                    self._json(400, {"error": "embedding must be [float,...]"})
                return
            result = state.checker.check(np.asarray(rows[0], dtype=np.float64))
            self._json(200, {k: result[k] for k in
                             ("healthy", "distance", "threshold")})

        def _calibrate(self) -> None:
            body = self._read_json()
            if not isinstance(body, dict):
                self._json(400, {"error": "expected {\"vectors\": [[...],...]}"})
                return
            rows = self._vec_list(body.get("vectors"))
            if rows is None:
                self._json(400, {"error": "vectors must be non-empty [[float,...],...]"})
                return
            X = np.asarray(rows, dtype=np.float64)
            if X.shape[0] < 2:
                self._json(400, {"error": "need >= 2 vectors"})
                return
            thr = calibrate(X, threshold_pct=state.checker.threshold_pct)
            # swap in atomically only after a clean calibration
            new = EmbedHealthCheck("unused", ref_vectors=X)
            if not new.loaded:
                self._json(500, {"error": "calibration failed"})
                return
            state.checker = new
            self._json(200, {"threshold": thr, "reference_size": int(X.shape[0])})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8390)
    ap.add_argument("--ref", default="data/embed_health_ref.npz",
                    help="reference NPZ (key 'vectors'); server starts "
                         "fail-safe if missing")
    args = ap.parse_args()

    state = EmbedHealthState(args.ref)
    if not state.loaded:
        print(f"WARNING: reference set {args.ref!r} not usable - serving "
              f"fail-safe (healthy=null) until /health/calibrate",
              flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"embed_health_server on 127.0.0.1:{args.port} "
          f"(ref loaded: {state.loaded})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
