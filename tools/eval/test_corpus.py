import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus as corpus_mod
import embeddings as emb


def test_load_corpus_reads_fixture(tmp_path):
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps([
        {"text": "alpha fix the parser", "intent": "alpha", "ood": False, "source": "t"},
        {"text": "beta plan the sprint", "intent": "beta", "ood": True, "source": "t"},
        {"text": "gamma review the diff", "intent": "gamma", "ood": False, "source": "t"},
    ]), encoding="utf-8")
    items = corpus_mod.load_corpus(p)
    assert len(items) == 3
    assert items[0]["text"] == "alpha fix the parser"
    assert items[0]["intent"] == "alpha"
    assert items[0]["ood"] is False
    assert items[1]["ood"] is True          # OOD flag preserved
    assert items[2]["source"] == "t"


def test_load_corpus_defaults_and_json5(tmp_path):
    p = tmp_path / "corpus.json5"
    p.write_text(
        "// comment\n[\n"
        '  {"text": "alpha one", "intent": "alpha"},  // ood omitted -> false\n'
        '  {"text": "beta two", "intent": "beta", "ood": true},\n'
        "]",
        encoding="utf-8")
    items = corpus_mod.load_corpus(p)
    assert items[0]["ood"] is False
    assert items[1]["ood"] is True


def test_load_corpus_rejects_bad_items(tmp_path):
    p = tmp_path / "corpus.json"
    p.write_text('[{"text": "no intent"}]', encoding="utf-8")
    with pytest.raises(ValueError):
        corpus_mod.load_corpus(p)


def test_cached_embed_calls_endpoint_once(tmp_path):
    fake = emb.DeterministicFakeEndpoint(dim=8)
    texts = ["alpha fix the parser", "beta plan the sprint",
             "alpha fix the parser"]  # duplicate within the batch
    v1 = emb.embed_texts(texts, model_id="m1", cache_dir=tmp_path, client=fake)
    assert fake.requests == 1  # one batch for the whole first call
    assert v1[0] == v1[2]      # identical text -> identical vector

    # Re-embedding the same texts must hit ONLY the disk cache.
    v2 = emb.embed_texts(texts, model_id="m1", cache_dir=tmp_path, client=fake)
    assert fake.requests == 1  # no new endpoint calls
    assert v1 == v2

    # A different model id is a different cache key -> a new call.
    emb.embed_texts(texts, model_id="m2", cache_dir=tmp_path, client=fake)
    assert fake.requests == 2

    # Single-text convenience API also goes through the cache.
    a = emb.cached_embed("gamma review the diff", model_id="m1",
                         cache_dir=tmp_path, client=fake)
    b = emb.cached_embed("gamma review the diff", model_id="m1",
                         cache_dir=tmp_path, client=fake)
    assert fake.requests == 3
    assert a == b


def test_cache_keys_and_privacy(tmp_path):
    fake = emb.DeterministicFakeEndpoint(dim=8)
    secret = "alpha totally private corpus text"
    emb.cached_embed(secret, model_id="m", cache_dir=tmp_path, client=fake)
    # Nothing on disk contains the raw text.
    for f in Path(tmp_path).iterdir():
        assert "private corpus" not in f.read_text(encoding="utf-8")
    # Keys are sha256(model + NUL + text).
    assert emb.cache_key("m", secret) == emb.cache_key("m", secret)
    assert emb.cache_key("m", secret) != emb.cache_key("other", secret)


def test_deterministic_reference_vector_is_stable():
    a = emb.deterministic_reference_vector("alpha hello")
    b = emb.deterministic_reference_vector("alpha hello")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
