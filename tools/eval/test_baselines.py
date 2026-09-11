import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import centroid_margin, knn_unanimity, tfidf_logistic


def _separable(n_per=25, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    centers = {"a": 0, "b": 1, "c": 2}
    for name, basis in centers.items():
        for _ in range(n_per):
            v = rng.normal(0, 0.05, dim)
            v[basis] += 1.0
            X.append(v.tolist())
            y.append(name)
    return X, y


def test_centroid_margin_separable(tmp_path=None):
    X, y = _separable()
    b = centroid_margin().fit(X, y)
    lbl, conf, margin = b.predict(X[0])
    assert lbl == y[0]
    assert conf > 0.9
    assert margin > 0.1


def test_centroid_margin_clean_vs_ambiguous():
    X, y = _separable(n_per=10, seed=3)
    b = centroid_margin().fit(X, y)
    clean = [0.0] * 8
    clean[0] = 1.0                       # dead on the 'a' centroid
    _, conf_clean, m_clean = b.predict(clean)
    assert m_clean > 0                   # positive margin for a clean case
    ambiguous = [1.0 / 3] * 3 + [0.0] * 5   # equidistant from a, b, c
    _, conf_amb, m_amb = b.predict(ambiguous)
    assert m_amb < m_clean               # ambiguous case has smaller margin
    assert conf_amb <= conf_clean


def test_knn_unanimity_separable():
    X, y = _separable(n_per=20, seed=1)
    b = knn_unanimity(k=5).fit(X, y)
    lbl, conf, margin = b.predict(X[7])
    assert lbl == y[7]
    assert conf == 1.0                   # clean cluster -> kNN unanimity
    assert margin == 1.0


def test_knn_k_default_is_5():
    b = knn_unanimity()
    assert b.k == 5


def test_tfidf_logistic_text_separable():
    texts = ([("alpha " + w) for w in ["parse", "token", "lexer", "ast", "regex"]] * 6
             + [("beta " + w) for w in ["sprint", "backlog", "standup", "roadmap", "kanban"]] * 6)
    labels = ["a"] * 30 + ["b"] * 30
    b = tfidf_logistic().fit(texts, labels)
    lbl, conf, margin = b.predict("alpha lexer tokens")
    assert lbl == "a"
    assert conf > 0.8
    assert margin > 0.2
    lbl2, conf2, m2 = b.predict("beta roadmap planning")
    assert lbl2 == "b"
    assert conf2 > 0.8


def test_all_baselines_predict_before_fit_raises():
    for b in (centroid_margin(), knn_unanimity(), tfidf_logistic()):
        with pytest.raises(RuntimeError):
            b.predict([0.0])
