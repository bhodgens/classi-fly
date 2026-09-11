"""Embedding access for the eval harness, with a disk cache.

Privacy rule: corpus text is sent ONLY to the OpenAI-compatible embedding
endpoint whose base URL the caller passes in (``--embed-url``), or to an
in-process fake client in tests/fixtures. Corpus text is never written to
disk: the cache is keyed by ``sha256(model_id + NUL + text)`` and stores the
vector only, never the raw text.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "classi-fly-embed-cache"

# Prefixes understood by the deterministic reference vectors below. Used by
# the fixture generator and the fake endpoint ONLY - never by real inference.
FIXTURE_VOCAB = {"alpha": 0, "beta": 1, "gamma": 2}


def cache_key(model_id: str, text: str) -> str:
    """sha256(model_id + NUL + text) - the disk-cache key."""
    return hashlib.sha256((model_id + "\x00" + text).encode("utf-8")).hexdigest()


def deterministic_reference_vector(text: str, dim: int = 8) -> list:
    """Fixture/test-only deterministic stand-in for a real embedding.

    Texts whose first token is one of FIXTURE_VOCAB map to that unit basis
    vector plus hash-derived jitter; anything else maps to the neutral
    all-ones direction. Deterministic across runs; sends nothing anywhere.
    """
    digest = hashlib.sha256(("refvec\x00" + text).encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vec = rng.uniform(-1.0, 1.0, dim) * 0.02
    first = ""
    for tok in text.lower().split():
        tok = tok.strip(".,:;!?()[]{}\"'")
        if tok:
            first = tok
            break
    basis = FIXTURE_VOCAB.get(first)
    if basis is None or basis >= dim:
        vec = vec + np.ones(dim) / np.sqrt(dim)
    else:
        vec[basis] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec = np.ones(dim) / np.sqrt(dim)
        norm = 1.0
    return [float(x) for x in vec / norm]


class HTTPEndpoint:
    """OpenAI-compatible /embeddings client (the only network path)."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, texts, model_id):
        body = json.dumps({"model": model_id, "input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = sorted(payload["data"], key=lambda d: d.get("index", 0))
        return [[float(x) for x in item["embedding"]] for item in data]


class DeterministicFakeEndpoint:
    """In-process fake endpoint for tests/fixtures. Counts request batches."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.requests = 0

    def embed(self, texts, model_id):
        self.requests += 1
        return [deterministic_reference_vector(t, self.dim) for t in texts]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / (key + ".json")


def _read_cache(cache_dir, key: str):
    p = _cache_path(cache_dir, key)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    vec = rec.get("vector")
    if not isinstance(vec, list):
        return None
    return [float(x) for x in vec]


def _write_cache(cache_dir, key: str, model_id: str, vec) -> None:
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    # Stores the vector only - never the raw text.
    rec = {"model": model_id, "key": key, "dim": len(vec), "vector": [float(x) for x in vec]}
    _cache_path(cdir, key).write_text(json.dumps(rec), encoding="utf-8")


def cached_embed(text, *, model_id=DEFAULT_EMBED_MODEL, url=None, cache_dir=None,
                 client=None, timeout: float = 30.0):
    """Embed one text; the disk cache guarantees at most one endpoint call."""
    return embed_texts([text], model_id=model_id, url=url, cache_dir=cache_dir,
                       client=client, timeout=timeout)[0]


def embed_texts(texts, *, model_id=DEFAULT_EMBED_MODEL, url=None, cache_dir=None,
                client=None, timeout: float = 30.0):
    """Embed many texts; only cache misses reach the endpoint, in one batch."""
    if client is None:
        if not url:
            raise ValueError("embeddings: pass an endpoint URL (--embed-url) or a client")
        client = HTTPEndpoint(url, timeout=timeout)
    cdir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    keys = [cache_key(model_id, t) for t in texts]
    out = {}
    misses = []
    for text, key in zip(texts, keys):
        if key in out:
            continue
        hit = _read_cache(cdir, key)
        if hit is None:
            misses.append((text, key))
        else:
            out[key] = hit
    if misses:
        vectors = client.embed([t for t, _ in misses], model_id)
        if len(vectors) != len(misses):
            raise RuntimeError(
                "embeddings: endpoint returned %d vectors for %d inputs"
                % (len(vectors), len(misses)))
        for (_text, key), vec in zip(misses, vectors):
            _write_cache(cdir, key, model_id, vec)
            out[key] = [float(x) for x in vec]
    return [out[key] for key in keys]
